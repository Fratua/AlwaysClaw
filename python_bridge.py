"""
Python Bridge - JSON-RPC server for Node.js <-> Python IPC.
Reads newline-delimited JSON from stdin, writes JSON responses to stdout.
All logging goes to stderr only.
"""

import sys
import json
import os
import traceback
import logging
import sqlite3
import shutil
from typing import Any, Callable, Dict, Optional

# Force all logging to stderr so stdout stays clean for JSON-RPC
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format='[PythonBridge] %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()


class PythonBridge:
    """JSON-RPC server that dispatches to registered handlers."""

    def __init__(self):
        self.handlers: Dict[str, Callable] = {}
        self.llm_client = None
        self.memory_manager = None
        self.db_connection = None
        self._gmail_client = None
        self._gmail_import_failed = False
        self._tts_orchestrator = None
        self._tts_import_failed = False
        self._twilio_manager = None
        self._twilio_import_failed = False
        self._stopping = False
        self._register_builtin_handlers()

    def _register_builtin_handlers(self):
        """Register built-in handlers."""
        self.handlers['echo'] = self._handle_echo
        self.handlers['health'] = self._handle_health
        self.handlers['llm.complete'] = self._handle_llm_complete
        self.handlers['llm.generate'] = self._handle_llm_generate

        # Memory handlers
        self.handlers['memory.initialize'] = self._handle_memory_initialize
        self.handlers['memory.store'] = self._handle_memory_store
        self.handlers['memory.search'] = self._handle_memory_search
        self.handlers['memory.consolidate'] = self._handle_memory_consolidate
        self.handlers['memory.backup'] = self._handle_memory_backup
        self.handlers['memory.sync'] = self._handle_memory_sync

        # Gmail handlers (delegate to gmail_client_implementation)
        self.handlers['gmail.send'] = self._handle_gmail_send
        self.handlers['gmail.read'] = self._handle_gmail_read
        self.handlers['gmail.search'] = self._handle_gmail_search
        self.handlers['gmail.context'] = self._handle_gmail_context
        self.handlers['gmail.process_batch'] = self._handle_gmail_process_batch

        # Twilio handlers
        self.handlers['twilio.call'] = self._handle_twilio_call
        self.handlers['twilio.sms'] = self._handle_twilio_sms

        # TTS handlers
        self.handlers['tts.speak'] = self._handle_tts_speak

        # STT handlers
        self.handlers['stt.transcribe'] = self._handle_stt_transcribe

        # Auth handlers
        self.handlers['auth.validate'] = self._handle_auth_validate

    def _initialize_llm(self):
        """Lazy-initialize the OpenAI GPT-5.2 client."""
        if self.llm_client is None:
            try:
                from openai_client import OpenAIClient
                self.llm_client = OpenAIClient.get_instance()
                logger.info(f"OpenAI GPT-5.2 client initialized (model={self.llm_client.model})")
            except (ImportError, RuntimeError, OSError, ValueError) as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                raise

    def _initialize_memory_db(self):
        """Initialize SQLite memory database from schema."""
        if self.db_connection is not None:
            return

        db_path = os.environ.get('MEMORY_DB_PATH', './data/memory.db')
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        if db_dir and not os.access(db_dir, os.W_OK):
            raise PermissionError(
                f"Memory DB directory is not writable: {db_dir}\n"
                f"Ensure the process has write permissions, or set MEMORY_DB_PATH to a writable location."
            )

        try:
            self.db_connection = sqlite3.connect(db_path)
        except sqlite3.OperationalError as e:
            raise sqlite3.OperationalError(
                f"Cannot open memory database at {db_path}: {e}\n"
                f"Check file permissions and disk space."
            ) from e
        self.db_connection.row_factory = sqlite3.Row

        # Apply schema
        schema_path = os.path.join(os.path.dirname(__file__), 'init_memory_db.sql')
        if os.path.exists(schema_path):
            try:
                with open(schema_path, 'r') as f:
                    schema_sql = f.read()
                # Filter out the sqlite-vec virtual table (may not be available)
                lines = schema_sql.split('\n')
                filtered = []
                skip_block = False
                for line in lines:
                    if 'CREATE VIRTUAL TABLE' in line and 'vec0' in line:
                        skip_block = True
                        continue
                    if skip_block and line.strip() == ');':
                        skip_block = False
                        continue
                    if not skip_block:
                        filtered.append(line)
                filtered_sql = '\n'.join(filtered)
                self.db_connection.executescript(filtered_sql)
                logger.info(f"Memory DB initialized at {db_path}")
            except (sqlite3.Error, OSError) as e:
                logger.error(f"Failed to apply memory schema: {e}")
        else:
            logger.warning(f"Memory schema not found at {schema_path}")

    # ── Built-in handlers ────────────────────────────────────────

    def _handle_echo(self, **params):
        return params

    def _handle_health(self, **params):
        return {
            "status": "ok",
            "llm_initialized": self.llm_client is not None,
            "memory_initialized": self.db_connection is not None,
            "pid": os.getpid(),
        }

    def _handle_llm_complete(self, **params):
        self._initialize_llm()
        return self.llm_client.complete(
            messages=params.get('messages', []),
            system=params.get('system', ''),
            max_tokens=params.get('max_tokens', 4096),
            temperature=params.get('temperature', 0.7),
        )

    def _handle_llm_generate(self, **params):
        self._initialize_llm()
        text = self.llm_client.generate(
            prompt=params.get('prompt', ''),
            system=params.get('system', ''),
        )
        return {"content": text}

    # ── Memory handlers ──────────────────────────────────────────

    def _handle_memory_initialize(self, **params):
        self._initialize_memory_db()
        return {"initialized": True}

    def _handle_memory_store(self, **params):
        self._initialize_memory_db()
        import uuid
        entry_id = params.get('id', str(uuid.uuid4()))
        mem_type = params.get('type', 'episodic')
        content = params.get('content', '')
        source = params.get('source', 'bridge')
        importance = params.get('importance', 0.5)

        self.db_connection.execute(
            """INSERT OR REPLACE INTO memory_entries
               (id, type, content, source_file, importance_score)
               VALUES (?, ?, ?, ?, ?)""",
            (entry_id, mem_type, content, source, importance),
        )
        # Also insert into FTS index
        self.db_connection.execute(
            """INSERT OR REPLACE INTO memory_fts (content, memory_id)
               VALUES (?, ?)""",
            (content, entry_id),
        )
        self.db_connection.commit()

        # Tag support
        tags = params.get('tags', [])
        for tag in tags:
            self.db_connection.execute(
                "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                (entry_id, tag),
            )
        if tags:
            self.db_connection.commit()

        return {"stored": True, "id": entry_id}

    def _handle_memory_search(self, **params):
        self._initialize_memory_db()
        query = params.get('query', '')
        limit = params.get('limit', 10)
        mem_type = params.get('type')

        # Use FTS5 for text search
        sql = """
            SELECT me.id, me.type, me.content, me.source_file,
                   me.importance_score, me.created_at, me.access_count,
                   rank
            FROM memory_fts fts
            JOIN memory_entries me ON fts.memory_id = me.id
            WHERE memory_fts MATCH ?
        """
        bind_params = [query]

        if mem_type:
            sql += " AND me.type = ?"
            bind_params.append(mem_type)

        sql += " ORDER BY rank LIMIT ?"
        bind_params.append(limit)

        try:
            rows = self.db_connection.execute(sql, bind_params).fetchall()
            results = []
            for row in rows:
                results.append({
                    "id": row["id"],
                    "type": row["type"],
                    "content": row["content"],
                    "source": row["source_file"],
                    "importance": row["importance_score"],
                    "created_at": row["created_at"],
                    "access_count": row["access_count"],
                })
                # Update access count
                self.db_connection.execute(
                    "UPDATE memory_entries SET access_count = access_count + 1, "
                    "last_accessed = CURRENT_TIMESTAMP WHERE id = ?",
                    (row["id"],),
                )
            self.db_connection.commit()
            return {"results": results, "count": len(results)}
        except (sqlite3.Error, ValueError) as e:
            logger.error(f"Memory search error: {e}")
            return {"results": [], "count": 0, "error": str(e)}

    def _handle_memory_consolidate(self, **params):
        self._initialize_memory_db()
        # Simple consolidation: count entries per type
        rows = self.db_connection.execute(
            "SELECT type, COUNT(*) as cnt FROM memory_entries GROUP BY type"
        ).fetchall()
        stats = {row["type"]: row["cnt"] for row in rows}
        return {"consolidated": True, "stats": stats}

    def _handle_memory_backup(self, **params):
        self._initialize_memory_db()
        db_path = os.environ.get('MEMORY_DB_PATH', './data/memory.db')
        backup_path = db_path + '.backup'
        try:
            shutil.copy2(db_path, backup_path)
            return {"backed_up": True, "path": backup_path}
        except OSError as e:
            return {"backed_up": False, "error": str(e)}

    def _handle_memory_sync(self, **params):
        self._initialize_memory_db()
        count = self.db_connection.execute(
            "SELECT COUNT(*) as cnt FROM memory_entries"
        ).fetchone()["cnt"]
        return {"synced": True, "total_entries": count}

    # ── Lazy client initializers ────────────────────────────────

    def _get_gmail_client(self):
        """Lazy-initialize and cache the Gmail client."""
        if self._gmail_client is None and not self._gmail_import_failed:
            try:
                from gmail_client_implementation import GmailClient
                self._gmail_client = GmailClient()
                logger.info("Gmail client initialized")
            except ImportError:
                self._gmail_import_failed = True
                logger.warning("gmail_client_implementation not available")
                raise
        if self._gmail_import_failed:
            raise ImportError("Gmail client not available")
        return self._gmail_client

    def _get_tts_orchestrator(self):
        """Lazy-initialize and cache the TTS orchestrator."""
        if self._tts_orchestrator is None and not self._tts_import_failed:
            try:
                from tts_orchestrator import TTSOrchestrator
                self._tts_orchestrator = TTSOrchestrator()
                logger.info("TTS orchestrator initialized")
            except ImportError:
                self._tts_import_failed = True
                logger.warning("tts_orchestrator not available")
                raise
        if self._tts_import_failed:
            raise ImportError("TTS orchestrator not available")
        return self._tts_orchestrator

    def _get_twilio_manager(self):
        """Lazy-initialize and cache the Twilio manager."""
        if self._twilio_manager is None and not self._twilio_import_failed:
            try:
                from twilio_voice_integration import TwilioVoiceManager
                self._twilio_manager = TwilioVoiceManager()
                logger.info("Twilio manager initialized")
            except ImportError:
                self._twilio_import_failed = True
                logger.warning("twilio_voice_integration not available")
                raise
        if self._twilio_import_failed:
            raise ImportError("Twilio manager not available")
        return self._twilio_manager

    # ── Gmail handlers ───────────────────────────────────────────

    def _handle_gmail_send(self, **params):
        to_addr = params.get('to', '')
        if not to_addr or '@' not in to_addr:
            return {"sent": False, "error": f"Invalid recipient address: {to_addr!r}"}
        try:
            from gmail_client_implementation import GmailClient
            client = self._get_gmail_client()
            return client.messages.send_message(
                to=to_addr,
                subject=params.get('subject', ''),
                body=params.get('body', ''),
            )
        except ImportError:
            logger.warning("gmail_client_implementation not available")
            return {"sent": False, "error": "Gmail client not available"}

    def _handle_gmail_read(self, **params):
        try:
            client = self._get_gmail_client()
            return client.messages.list_messages(
                max_results=params.get('max_results', 10),
                query=params.get('query', 'is:unread'),
            )
        except ImportError:
            return {"emails": [], "error": "Gmail client not available"}

    def _handle_gmail_search(self, **params):
        try:
            client = self._get_gmail_client()
            return client.messages.list_messages(query=params.get('query', ''))
        except ImportError:
            return {"emails": [], "error": "Gmail client not available"}

    def _handle_gmail_context(self, **params):
        try:
            client = self._get_gmail_client()
            messages = client.messages.list_messages(query='is:unread', max_results=100)
            unread_count = len(messages) if isinstance(messages, list) else messages.get('resultSizeEstimate', 0) if isinstance(messages, dict) else 0
            return {"unread": unread_count}
        except ImportError:
            return {"unread": 0, "error": "Gmail client not available"}
        except (ConnectionError, TimeoutError, OSError, ValueError) as e:
            return {"unread": 0, "error": str(e)}

    def _handle_gmail_process_batch(self, **params):
        try:
            client = self._get_gmail_client()
            messages = client.messages.list_messages(
                max_results=params.get('max_results', 50),
                query=params.get('query', ''),
            )
            return {"processed": True, "messages": messages}
        except ImportError:
            return {"processed": False, "error": "Gmail client not available"}

    # ── Twilio handlers ──────────────────────────────────────────

    def _handle_twilio_call(self, **params):
        to_num = params.get('to', '')
        if not to_num:
            return {"called": False, "error": "No 'to' number provided"}
        try:
            manager = self._get_twilio_manager()
            return manager.make_call(
                to_number=params.get('to', ''),
                twiml_url=params.get('twiml_url', ''),
                from_number=params.get('from', ''),
            )
        except ImportError:
            return {"called": False, "error": "Twilio client not available"}
        except (ConnectionError, TimeoutError, OSError, ValueError) as e:
            return {"called": False, "error": str(e)}

    def _handle_twilio_sms(self, **params):
        to_num = params.get('to', '')
        body = params.get('body', '')
        if not to_num:
            return {"sent": False, "error": "No 'to' number provided"}
        if not body:
            return {"sent": False, "error": "No message body provided"}
        try:
            manager = self._get_twilio_manager()
            return manager.send_sms(
                to=params.get('to', ''),
                body=params.get('body', ''),
            )
        except ImportError:
            return {"sent": False, "error": "Twilio client not available"}
        except (ConnectionError, TimeoutError, OSError, ValueError) as e:
            return {"sent": False, "error": str(e)}

    # ── TTS handlers ─────────────────────────────────────────────

    def _handle_tts_speak(self, **params):
        text = params.get('text', '')
        if not text:
            return {"spoken": False, "error": "No text provided"}
        if len(text) > 10000:
            return {"spoken": False, "error": f"Text too long ({len(text)} chars, max 10000)"}
        try:
            orchestrator = self._get_tts_orchestrator()
            return orchestrator.speak(
                text=text,
                voice=params.get('voice'),
                output_path=params.get('output_path'),
            )
        except ImportError:
            return {"spoken": False, "error": "TTS orchestrator not available"}

    # ── STT handlers ─────────────────────────────────────────────

    def _handle_stt_transcribe(self, **params):
        try:
            import azure.cognitiveservices.speech as speechsdk
            speech_key = os.environ.get('AZURE_SPEECH_KEY', '')
            speech_region = os.environ.get('AZURE_SPEECH_REGION', '')
            if not speech_key or not speech_region:
                return {"text": "", "error": "Azure Speech credentials not configured (set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION)"}

            audio_path = params.get('audio_path', '')
            if not audio_path:
                return {"text": "", "error": "No audio_path provided"}

            speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
            audio_config = speechsdk.AudioConfig(filename=audio_path)
            recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
            result = recognizer.recognize_once()

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return {"text": result.text, "confidence": 1.0}
            elif result.reason == speechsdk.ResultReason.NoMatch:
                return {"text": "", "error": "No speech recognized"}
            else:
                return {"text": "", "error": f"Speech recognition failed: {result.reason}"}
        except ImportError:
            return {"text": "", "error": "Azure Speech SDK not installed (pip install azure-cognitiveservices-speech)"}
        except (OSError, RuntimeError, ValueError) as e:
            return {"text": "", "error": str(e)}

    # ── Auth handlers ────────────────────────────────────────────

    def _handle_auth_validate(self, **params):
        token = params.get('token', '')
        if not token:
            return {"valid": False, "error": "No token provided"}

        # Try JWT validation if pyjwt is available
        try:
            import jwt
            secret = os.environ.get('JWT_SECRET', '')
            if secret:
                decoded = jwt.decode(token, secret, algorithms=['HS256'])
                return {"valid": True, "claims": decoded}
        except jwt.ExpiredSignatureError:
            return {"valid": False, "error": "Token expired"}
        except jwt.InvalidTokenError as e:
            return {"valid": False, "error": f"Invalid token: {e}"}
        except ImportError:
            pass

        # No JWT_SECRET configured and no JWT library available - cannot validate
        return {"valid": False, "error": "JWT_SECRET not configured; cannot validate tokens"}

    # ── Loop handler registration (called from loop_adapters.py) ─

    def register_loop_handlers(self):
        """Register all cognitive loop handlers."""
        try:
            from loop_adapters import get_loop_handlers
            loop_handlers = get_loop_handlers(self.llm_client)
            self.handlers.update(loop_handlers)
            logger.info(f"Registered {len(loop_handlers)} loop handlers")
        except (ImportError, RuntimeError) as e:
            logger.error(f"Failed to register loop handlers: {e}")

    # ── Main event loop ──────────────────────────────────────────

    def _check_env_vars(self):
        """Check required and optional env vars at startup and log what's missing."""
        required_vars = {
            'OPENAI_API_KEY': 'LLM completions (llm.complete, llm.generate, cognitive loops)',
        }
        optional_vars = {
            'MEMORY_DB_PATH': 'Memory persistence (default: ./data/memory.db)',
            'GMAIL_CREDENTIALS_PATH': 'Gmail integration (gmail.send, gmail.read, gmail.search)',
            'TWILIO_ACCOUNT_SID': 'Twilio voice/SMS (twilio.call, twilio.sms)',
            'TWILIO_AUTH_TOKEN': 'Twilio voice/SMS',
            'AZURE_SPEECH_KEY': 'Speech-to-text (stt.transcribe)',
            'AZURE_SPEECH_REGION': 'Speech-to-text',
            'JWT_SECRET': 'JWT token validation (auth.validate)',
        }
        for var, feature in required_vars.items():
            if not os.environ.get(var):
                logger.error(f"REQUIRED env var missing: {var} ({feature})")
        missing_optional = []
        for var, feature in optional_vars.items():
            if not os.environ.get(var):
                missing_optional.append(f"  {var}: {feature}")
        if missing_optional:
            logger.warning(
                "Optional environment variables not set (features will be disabled):\n"
                + "\n".join(missing_optional)
            )

    def run(self):
        """Main loop: read JSON-RPC requests from stdin, write responses to stdout."""
        logger.info("Python bridge starting...")

        # Check env vars and report what's missing
        self._check_env_vars()

        # Initialize shared resources
        try:
            self._initialize_llm()
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            logger.warning(f"OpenAI client init deferred: {e}")

        try:
            self._initialize_memory_db()
        except (sqlite3.Error, OSError) as e:
            logger.warning(f"Memory DB init deferred: {e}")

        # Register loop handlers
        self.register_loop_handlers()

        logger.info(f"Bridge ready. {len(self.handlers)} handlers registered.")

        # Signal readiness to Node.js
        self._send_response({"jsonrpc": "2.0", "method": "ready", "params": {
            "handlers": list(self.handlers.keys()),
        }})

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                self._send_error(None, -32700, f"Parse error: {e}")
                continue

            req_id = request.get('id')
            method = request.get('method', '')
            params = request.get('params', {})

            logger.debug(f"Request: {method} (id={req_id})")

            handler = self.handlers.get(method)
            if not handler:
                self._send_error(req_id, -32601, f"Method not found: {method}")
                continue

            try:
                if isinstance(params, dict):
                    result = handler(**params)
                elif isinstance(params, list):
                    result = handler(*params)
                else:
                    result = handler()

                self._send_response({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": result,
                })
            except (RuntimeError, OSError, ValueError, KeyError, TypeError, ImportError) as e:
                logger.error(f"Handler error for {method}: {e}\n{traceback.format_exc()}")
                self._send_error(req_id, -32000, str(e))

    def _send_response(self, response: dict):
        """Write a JSON response to stdout (newline-delimited)."""
        try:
            line = json.dumps(response, default=str)
            sys.stdout.write(line + '\n')
            sys.stdout.flush()
        except (OSError, ValueError, TypeError) as e:
            logger.error(f"Failed to send response: {e}")

    def _send_error(self, req_id: Optional[Any], code: int, message: str):
        self._send_response({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        })


if __name__ == '__main__':
    import atexit
    bridge = PythonBridge()

    def _cleanup():
        if bridge.db_connection is not None:
            try:
                bridge.db_connection.close()
            except Exception:
                pass

    atexit.register(_cleanup)
    bridge.run()
