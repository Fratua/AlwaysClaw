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
                logger.info("OpenAI GPT-5.2 client initialized")
            except Exception as e:
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

        self.db_connection = sqlite3.connect(db_path)
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
            except Exception as e:
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
        except Exception as e:
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
        except Exception as e:
            return {"backed_up": False, "error": str(e)}

    def _handle_memory_sync(self, **params):
        self._initialize_memory_db()
        count = self.db_connection.execute(
            "SELECT COUNT(*) as cnt FROM memory_entries"
        ).fetchone()["cnt"]
        return {"synced": True, "total_entries": count}

    # ── Gmail handlers ───────────────────────────────────────────

    def _handle_gmail_send(self, **params):
        try:
            from gmail_client_implementation import GmailClient
            client = GmailClient()
            return client.send_message(
                to=params.get('to', ''),
                subject=params.get('subject', ''),
                body=params.get('body', ''),
            )
        except ImportError:
            logger.warning("gmail_client_implementation not available")
            return {"sent": False, "error": "Gmail client not available"}

    def _handle_gmail_read(self, **params):
        try:
            from gmail_client_implementation import GmailClient
            client = GmailClient()
            return client.read_messages(
                max_results=params.get('max_results', 10),
                query=params.get('query', 'is:unread'),
            )
        except ImportError:
            return {"emails": [], "error": "Gmail client not available"}

    def _handle_gmail_search(self, **params):
        try:
            from gmail_client_implementation import GmailClient
            client = GmailClient()
            return client.search_messages(query=params.get('query', ''))
        except ImportError:
            return {"emails": [], "error": "Gmail client not available"}

    def _handle_gmail_context(self, **params):
        try:
            from gmail_client_implementation import GmailClient
            client = GmailClient()
            return client.get_context()
        except ImportError:
            return {"unread": 0, "error": "Gmail client not available"}

    def _handle_gmail_process_batch(self, **params):
        try:
            from gmail_client_implementation import GmailClient
            client = GmailClient()
            return client.process_batch(params)
        except ImportError:
            return {"processed": False, "error": "Gmail client not available"}

    # ── Twilio handlers ──────────────────────────────────────────

    def _handle_twilio_call(self, **params):
        try:
            from twilio_voice_integration import TwilioVoiceClient
            client = TwilioVoiceClient()
            return client.make_call(
                to=params.get('to', ''),
                message=params.get('message', ''),
            )
        except ImportError:
            return {"called": False, "error": "Twilio client not available"}

    def _handle_twilio_sms(self, **params):
        try:
            from twilio_voice_integration import TwilioVoiceClient
            client = TwilioVoiceClient()
            return client.send_sms(
                to=params.get('to', ''),
                body=params.get('body', ''),
            )
        except ImportError:
            return {"sent": False, "error": "Twilio client not available"}

    # ── TTS handlers ─────────────────────────────────────────────

    def _handle_tts_speak(self, **params):
        try:
            from tts_orchestrator import TTSOrchestrator
            orchestrator = TTSOrchestrator()
            return orchestrator.speak(
                text=params.get('text', ''),
                voice=params.get('voice'),
                output_path=params.get('output_path'),
            )
        except ImportError:
            return {"spoken": False, "error": "TTS orchestrator not available"}

    # ── STT handlers ─────────────────────────────────────────────

    def _handle_stt_transcribe(self, **params):
        return {"text": "", "error": "STT not yet implemented via bridge"}

    # ── Auth handlers ────────────────────────────────────────────

    def _handle_auth_validate(self, **params):
        return {"valid": True}

    # ── Loop handler registration (called from loop_adapters.py) ─

    def register_loop_handlers(self):
        """Register all cognitive loop handlers."""
        try:
            from loop_adapters import get_loop_handlers
            loop_handlers = get_loop_handlers(self.llm_client)
            self.handlers.update(loop_handlers)
            logger.info(f"Registered {len(loop_handlers)} loop handlers")
        except Exception as e:
            logger.error(f"Failed to register loop handlers: {e}")

    # ── Main event loop ──────────────────────────────────────────

    def run(self):
        """Main loop: read JSON-RPC requests from stdin, write responses to stdout."""
        logger.info("Python bridge starting...")

        # Initialize shared resources
        try:
            self._initialize_llm()
        except Exception as e:
            logger.warning(f"OpenAI client init deferred: {e}")

        try:
            self._initialize_memory_db()
        except Exception as e:
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
            except Exception as e:
                logger.error(f"Handler error for {method}: {e}\n{traceback.format_exc()}")
                self._send_error(req_id, -32000, str(e))

    def _send_response(self, response: dict):
        """Write a JSON response to stdout (newline-delimited)."""
        try:
            line = json.dumps(response, default=str)
            sys.stdout.write(line + '\n')
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"Failed to send response: {e}")

    def _send_error(self, req_id: Optional[Any], code: int, message: str):
        self._send_response({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        })


if __name__ == '__main__':
    bridge = PythonBridge()
    bridge.run()
