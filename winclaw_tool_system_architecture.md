# WinClaw Tool System & Plugin Architecture Specification
## Windows 10 OpenClaw-Inspired AI Agent Framework

**Version:** 1.0.0  
**Date:** 2026  
**Status:** Technical Specification

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Tool Registry & Discovery](#3-tool-registry--discovery)
4. [Tool Schema Definitions](#4-tool-schema-definitions)
5. [Tool Execution Pipeline](#5-tool-execution-pipeline)
6. [Dynamic Tool Loading & Hot-Reloading](#6-dynamic-tool-loading--hot-reloading)
7. [Permission System & Capability Declarations](#7-permission-system--capability-declarations)
8. [Self-Improving Tool Generation](#8-self-improving-tool-generation)
9. [Tool Marketplace Architecture](#9-tool-marketplace-architecture)
10. [Integration Patterns](#10-integration-patterns)
11. [Security Model](#11-security-model)
12. [Implementation Examples](#12-implementation-examples)

---

## 1. Executive Summary

WinClaw is a Windows 10-focused AI agent framework inspired by OpenClaw, designed to provide autonomous, 24/7 operation with deep system integration. The tool/skill system is the core extensibility mechanism that enables the agent to interact with external services, APIs, and system resources.

### Key Design Principles

| Principle | Description |
|-----------|-------------|
| **MCP-Compliant** | Full compatibility with Model Context Protocol for interoperability |
| **Self-Improving** | Agent can generate, validate, and deploy new tools autonomously |
| **Security-First** | Multi-layer sandboxing with capability-based permissions |
| **Hot-Reloadable** | Zero-downtime tool updates and dynamic loading |
| **Marketplace-Ready** | Built-in skill distribution and verification system |

---

## 2. Architecture Overview

### 2.1 High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           WINCLAW AGENT SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   GPT-5.2 Core  │◄──►│  Agent Runtime  │◄──►│  Memory System  │         │
│  │  (Reasoning)    │    │  (Orchestrator) │    │  (Context/State)│         │
│  └─────────────────┘    └────────┬────────┘    └─────────────────┘         │
│                                  │                                          │
│                        ┌─────────┴─────────┐                                │
│                        │   TOOL SYSTEM     │                                │
│                        │    (This Spec)    │                                │
│                        └─────────┬─────────┘                                │
│                                  │                                          │
│  ┌───────────────────────────────┼───────────────────────────────┐         │
│  │                               │                               │         │
│  ▼                               ▼                               ▼         │
│ ┌──────────────┐    ┌──────────────────────┐    ┌──────────────────────┐  │
│ │Tool Registry │    │  Execution Pipeline  │    │  Permission Engine   │  │
│ │& Discovery   │    │  & Sandboxing        │    │  & Capability Mgmt   │  │
│ └──────────────┘    └──────────────────────┘    └──────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      TOOL ECOSYSTEM                                  │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │   │
│  │  │  Gmail  │ │ Browser │ │  TTS    │ │  STT    │ │ Twilio  │ ...    │   │
│  │  │  Skill  │ │ Control │ │ Engine  │ │ Engine  │ │ Voice   │        │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SKILL MARKETPLACE (ClawHub)                       │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │   │
│  │  │  Registry   │ │  Scanner    │ │  Validator  │ │  Installer  │    │   │
│  │  │  Service    │ │  Service    │ │  Service    │ │  Service    │    │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Core System Components

| Component | Responsibility | Technology Stack |
|-----------|----------------|------------------|
| **Agent Runtime** | Orchestrates agentic loops, manages state | Python 3.12+, asyncio |
| **Tool Registry** | Central catalog of all available tools | SQLite + In-Memory Cache |
| **Execution Engine** | Sandboxed tool execution | Docker + seccomp |
| **Permission System** | Capability-based access control | Casbin + Custom Policies |
| **Skill Generator** | AI-powered tool creation | GPT-5.2 + Code Validation |
| **Marketplace Client** | Skill discovery and installation | REST API + GitHub Integration |

---

## 3. Tool Registry & Discovery

### 3.1 Registry Architecture

The Tool Registry implements a hierarchical, distributed discovery pattern inspired by MCP and A2A protocols.

```python
# Core Registry Interface
class ToolRegistry(ABC):
    """Abstract base for tool registry implementations."""
    
    @abstractmethod
    async def register(self, tool: ToolDefinition) -> RegistrationResult:
        """Register a new tool in the registry."""
        pass
    
    @abstractmethod
    async def discover(self, query: DiscoveryQuery) -> List[ToolDefinition]:
        """Discover tools matching the query criteria."""
        pass
    
    @abstractmethod
    async def get_tool(self, tool_id: str) -> Optional[ToolDefinition]:
        """Retrieve a specific tool by ID."""
        pass
    
    @abstractmethod
    async def unregister(self, tool_id: str) -> bool:
        """Remove a tool from the registry."""
        pass
```

### 3.2 Registry Data Model

```python
@dataclass
class ToolDefinition:
    """Complete definition of a tool/skill."""
    
    # Identity
    id: str                          # Unique identifier (UUID)
    name: str                        # Human-readable name
    version: str                     # Semantic version
    namespace: str                   # Organization/author namespace
    
    # Metadata
    description: str                 # LLM-visible description
    long_description: Optional[str]  # Detailed documentation
    tags: List[str]                  # Searchable tags
    category: ToolCategory           # Functional category
    
    # Schema
    input_schema: JSONSchema         # Input parameter schema
    output_schema: JSONSchema        # Output result schema
    error_schema: JSONSchema         # Error response schema
    
    # Implementation
    implementation: ImplementationSpec  # How to execute the tool
    dependencies: List[Dependency]      # Required dependencies
    
    # Permissions
    required_capabilities: List[Capability]  # Required permissions
    risk_level: RiskLevel                     # Security risk classification
    
    # Lifecycle
    created_at: datetime
    updated_at: datetime
    author: AuthorInfo
    signature: Optional[str]         # Cryptographic signature
    
    # Runtime
    state: ToolState                 # ACTIVE, DEPRECATED, BETA, etc.
    stats: ToolStatistics            # Usage statistics

@dataclass
class ImplementationSpec:
    """Specification for tool implementation."""
    
    type: ImplementationType         # PYTHON, JAVASCRIPT, DOCKER, etc.
    entry_point: str                 # Module/function path
    code: Optional[str]              # Inline code (for generated tools)
    source_url: Optional[str]        # GitHub/repo URL
    checksum: Optional[str]          # SHA-256 of code
    
    # Sandbox configuration
    sandbox_config: SandboxConfig
    
    # Resource limits
    resource_limits: ResourceLimits
```

### 3.3 Discovery Query Interface

```python
@dataclass
class DiscoveryQuery:
    """Query for discovering tools."""
    
    # Text search
    keywords: Optional[List[str]] = None
    semantic_query: Optional[str] = None  # Natural language search
    
    # Filters
    categories: Optional[List[ToolCategory]] = None
    tags: Optional[List[str]] = None
    risk_levels: Optional[List[RiskLevel]] = None
    capabilities: Optional[List[Capability]] = None
    
    # Author/Trust
    namespaces: Optional[List[str]] = None
    trusted_only: bool = False
    verified_only: bool = False
    
    # Sorting
    sort_by: SortField = SortField.RELEVANCE
    sort_order: SortOrder = SortOrder.DESCENDING
    
    # Pagination
    limit: int = 20
    offset: int = 0

class ToolRegistryService:
    """Production registry implementation with caching and indexing."""
    
    def __init__(self):
        self._local_registry: Dict[str, ToolDefinition] = {}
        self._semantic_index: VectorIndex = VectorIndex()
        self._capability_graph: CapabilityGraph = CapabilityGraph()
        self._cache: TTLCache = TTLCache(maxsize=1000, ttl=300)
        
    async def discover(self, query: DiscoveryQuery) -> DiscoveryResult:
        """Multi-strategy tool discovery."""
        
        results = []
        
        # 1. Keyword/Tag search (inverted index)
        if query.keywords or query.tags:
            keyword_results = await self._keyword_search(query)
            results.extend(keyword_results)
        
        # 2. Semantic search (vector similarity)
        if query.semantic_query:
            semantic_results = await self._semantic_search(query.semantic_query)
            results.extend(semantic_results)
        
        # 3. Capability-based discovery
        if query.capabilities:
            capability_results = await self._capability_search(query.capabilities)
            results.extend(capability_results)
        
        # 4. Apply filters and ranking
        filtered = self._apply_filters(results, query)
        ranked = self._rank_results(filtered, query)
        
        return DiscoveryResult(
            tools=ranked[query.offset:query.offset + query.limit],
            total=len(ranked),
            query_id=str(uuid.uuid4())
        )
```

### 3.4 MCP-Compliant Discovery Endpoint

```python
class McpDiscoveryHandler:
    """Implements MCP protocol tools/list endpoint."""
    
    async def handle_tools_list(self, request: McpRequest) -> McpResponse:
        """Return list of available tools in MCP format."""
        
        tools = await self.registry.get_all_active_tools()
        
        return McpResponse(
            jsonrpc="2.0",
            result={
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.input_schema.to_dict(),
                        "annotations": {
                            "title": tool.name,
                            "readOnlyHint": tool.risk_level == RiskLevel.READONLY,
                            "destructiveHint": tool.risk_level == RiskLevel.DESTRUCTIVE,
                            "idempotentHint": tool.is_idempotent(),
                            "openWorldHint": tool.makes_external_calls()
                        }
                    }
                    for tool in tools
                ]
            }
        )
```

---

## 4. Tool Schema Definitions

### 4.1 JSON Schema Specification

Tools use JSON Schema 2020-12 for parameter and response validation.

```python
class ToolSchemaBuilder:
    """Builder for creating tool schemas with validation."""
    
    @staticmethod
    def create_schema(
        name: str,
        description: str,
        parameters: Dict[str, ParameterSpec],
        required: List[str] = None,
        returns: ReturnSpec = None
    ) -> ToolSchema:
        """Create a complete tool schema."""
        
        properties = {}
        for param_name, spec in parameters.items():
            properties[param_name] = {
                "type": spec.type.value,
                "description": spec.description,
                **spec.extra_schema
            }
        
        schema = {
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False
        }
        
        return ToolSchema(
            name=name,
            description=description,
            input_schema=schema,
            output_schema=returns.to_schema() if returns else {"type": "object"}
        )

# Example: Gmail Send Email Tool Schema
GMAIL_SEND_SCHEMA = ToolSchemaBuilder.create_schema(
    name="gmail_send_email",
    description="Send an email using Gmail API",
    parameters={
        "to": ParameterSpec(
            type=ParameterType.STRING,
            description="Recipient email address(es), comma-separated for multiple",
            extra_schema={"format": "email"}
        ),
        "subject": ParameterSpec(
            type=ParameterType.STRING,
            description="Email subject line",
            extra_schema={"maxLength": 998}
        ),
        "body": ParameterSpec(
            type=ParameterType.STRING,
            description="Email body content (HTML or plain text)"
        ),
        "cc": ParameterSpec(
            type=ParameterType.STRING,
            description="CC recipients, comma-separated",
            optional=True
        ),
        "bcc": ParameterSpec(
            type=ParameterType.STRING,
            description="BCC recipients, comma-separated",
            optional=True
        ),
        "attachments": ParameterSpec(
            type=ParameterType.ARRAY,
            description="List of file paths to attach",
            optional=True,
            extra_schema={
                "items": {"type": "string"},
                "maxItems": 25
            }
        ),
        "is_html": ParameterSpec(
            type=ParameterType.BOOLEAN,
            description="Whether body is HTML content",
            optional=True,
            default=False
        )
    },
    required=["to", "subject", "body"],
    returns=ReturnSpec(
        type=ReturnType.OBJECT,
        description="Email send result",
        fields={
            "message_id": {"type": "string", "description": "Gmail message ID"},
            "thread_id": {"type": "string", "description": "Gmail thread ID"},
            "sent_at": {"type": "string", "format": "date-time"},
            "success": {"type": "boolean"}
        }
    )
)
```

### 4.2 Schema Validation Pipeline

```python
class SchemaValidator:
    """Multi-layer schema validation for tool inputs/outputs."""
    
    def __init__(self):
        self._jsonschema_validator = Draft202012Validator
        self._custom_validators: List[CustomValidator] = []
        
    async def validate_input(
        self,
        tool: ToolDefinition,
        arguments: Dict[str, Any],
        context: ValidationContext
    ) -> ValidationResult:
        """Validate tool input arguments."""
        
        errors = []
        
        # 1. JSON Schema validation
        schema_errors = self._validate_json_schema(
            tool.input_schema,
            arguments
        )
        errors.extend(schema_errors)
        
        # 2. Semantic validation
        semantic_errors = await self._validate_semantics(
            tool,
            arguments,
            context
        )
        errors.extend(semantic_errors)
        
        # 3. Security validation
        security_errors = await self._validate_security(
            tool,
            arguments,
            context
        )
        errors.extend(security_errors)
        
        # 4. Custom validators
        for validator in self._custom_validators:
            custom_errors = await validator.validate(tool, arguments, context)
            errors.extend(custom_errors)
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            sanitized_args=self._sanitize_arguments(arguments, tool)
        )
    
    def _validate_json_schema(
        self,
        schema: Dict,
        data: Dict
    ) -> List[ValidationError]:
        """Validate against JSON Schema."""
        validator = self._jsonschema_validator(schema)
        return [
            ValidationError(
                field=error.json_path,
                message=error.message,
                severity=ErrorSeverity.ERROR
            )
            for error in validator.iter_errors(data)
        ]
    
    async def _validate_security(
        self,
        tool: ToolDefinition,
        arguments: Dict,
        context: ValidationContext
    ) -> List[ValidationError]:
        """Security-focused validation."""
        errors = []
        
        # Check for path traversal attempts
        for key, value in arguments.items():
            if isinstance(value, str):
                if ".." in value or value.startswith("/"):
                    errors.append(ValidationError(
                        field=key,
                        message=f"Potential path traversal detected in {key}",
                        severity=ErrorSeverity.CRITICAL
                    ))
        
        # Check for command injection patterns
        dangerous_patterns = [
            r";\s*rm\s+-rf",
            r"`.*`",
            r"\$\(.*\)",
            r"\|\s*sh",
            r">\s*/dev",
        ]
        
        for key, value in arguments.items():
            if isinstance(value, str):
                for pattern in dangerous_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        errors.append(ValidationError(
                            field=key,
                            message=f"Potential command injection in {key}",
                            severity=ErrorSeverity.CRITICAL
                        ))
        
        return errors
```

---

## 5. Tool Execution Pipeline

### 5.1 Execution Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TOOL EXECUTION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Request   │───►│  Validation │───►│  Permission │───►│  Resource   │  │
│  │   Receive   │    │   Layer     │    │    Check    │    │  Allocate   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘  │
│                                                                    │        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────▼──────┐  │
│  │   Result    │◄───│   Output    │◄───│   Sandbox   │◄───│  Execution  │  │
│  │   Return    │    │  Transform  │    │   Monitor   │    │   Engine    │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Execution Engine Implementation

```python
class ToolExecutionEngine:
    """Core execution engine with sandboxing and monitoring."""
    
    def __init__(self):
        self._sandbox_factory: SandboxFactory = DockerSandboxFactory()
        self._permission_engine: PermissionEngine = PermissionEngine()
        self._monitor: ExecutionMonitor = ExecutionMonitor()
        self._result_transformer: ResultTransformer = ResultTransformer()
        
    async def execute(
        self,
        tool: ToolDefinition,
        arguments: Dict[str, Any],
        context: ExecutionContext
    ) -> ExecutionResult:
        """Execute a tool with full sandboxing and monitoring."""
        
        execution_id = str(uuid.uuid4())
        start_time = time.monotonic()
        
        try:
            # 1. Validate input
            validation = await self._validate_input(tool, arguments, context)
            if not validation.valid:
                return ExecutionResult(
                    success=False,
                    error=ExecutionError(
                        type=ErrorType.VALIDATION,
                        message="Input validation failed",
                        details=validation.errors
                    ),
                    execution_id=execution_id
                )
            
            # 2. Check permissions
            permission_check = await self._permission_engine.check(
                tool,
                context.user,
                context.session
            )
            if not permission_check.allowed:
                return ExecutionResult(
                    success=False,
                    error=ExecutionError(
                        type=ErrorType.PERMISSION_DENIED,
                        message=f"Permission denied: {permission_check.reason}"
                    ),
                    execution_id=execution_id
                )
            
            # 3. Create sandbox
            sandbox = await self._sandbox_factory.create(
                tool.implementation.sandbox_config,
                execution_id
            )
            
            # 4. Execute in sandbox
            with self._monitor.track_execution(execution_id, tool):
                raw_result = await self._execute_in_sandbox(
                    sandbox,
                    tool,
                    validation.sanitized_args
                )
            
            # 5. Transform output
            transformed_result = await self._result_transformer.transform(
                raw_result,
                tool.output_schema
            )
            
            # 6. Cleanup
            await sandbox.destroy()
            
            return ExecutionResult(
                success=True,
                data=transformed_result,
                execution_id=execution_id,
                duration_ms=(time.monotonic() - start_time) * 1000
            )
            
        except Exception as e:
            logger.exception(f"Tool execution failed: {tool.name}")
            return ExecutionResult(
                success=False,
                error=ExecutionError(
                    type=ErrorType.EXECUTION,
                    message=str(e),
                    stack_trace=traceback.format_exc()
                ),
                execution_id=execution_id,
                duration_ms=(time.monotonic() - start_time) * 1000
            )
    
    async def _execute_in_sandbox(
        self,
        sandbox: Sandbox,
        tool: ToolDefinition,
        arguments: Dict[str, Any]
    ) -> Any:
        """Execute tool code within sandbox environment."""
        
        if tool.implementation.type == ImplementationType.PYTHON:
            return await self._execute_python(sandbox, tool, arguments)
        elif tool.implementation.type == ImplementationType.JAVASCRIPT:
            return await self._execute_javascript(sandbox, tool, arguments)
        elif tool.implementation.type == ImplementationType.DOCKER:
            return await self._execute_docker(sandbox, tool, arguments)
        else:
            raise ValueError(f"Unsupported implementation type: {tool.implementation.type}")
    
    async def _execute_python(
        self,
        sandbox: Sandbox,
        tool: ToolDefinition,
        arguments: Dict[str, Any]
    ) -> Any:
        """Execute Python tool in sandbox."""
        
        # Prepare execution script
        script = f"""
import json
import sys
sys.path.insert(0, '/tool')

# Import tool module
from {tool.implementation.entry_point} import run

# Parse arguments
args = json.loads('{json.dumps(arguments)}')

# Execute
result = run(**args)

# Output result
print(json.dumps({{"result": result}}))
"""
        
        # Write script to sandbox
        await sandbox.write_file("/tmp/execute.py", script)
        
        # Execute with resource limits
        result = await sandbox.execute_command(
            ["python3", "/tmp/execute.py"],
            timeout=tool.implementation.resource_limits.max_execution_time,
            memory_limit=tool.implementation.resource_limits.max_memory_mb
        )
        
        if result.returncode != 0:
            raise ExecutionError(
                type=ErrorType.EXECUTION,
                message=f"Tool execution failed: {result.stderr}"
            )
        
        return json.loads(result.stdout)["result"]
```

### 5.3 Sandboxing Implementation

```python
class DockerSandboxFactory(SandboxFactory):
    """Docker-based sandbox implementation for Windows 10."""
    
    def __init__(self):
        self._docker_client = docker.from_env()
        self._image_cache: Dict[str, str] = {}
        
    async def create(
        self,
        config: SandboxConfig,
        execution_id: str
    ) -> Sandbox:
        """Create a new isolated sandbox container."""
        
        # Get or build base image
        image = await self._get_base_image(config)
        
        # Create container with security constraints
        container = self._docker_client.containers.run(
            image,
            command=["sleep", "3600"],  # Keep alive
            detach=True,
            name=f"winclaw-sandbox-{execution_id}",
            
            # Security options
            security_opt=[
                "no-new-privileges:true",
                "seccomp=winclaw-seccomp.json"
            ],
            cap_drop=["ALL"],
            cap_add=config.allowed_capabilities or [],
            
            # Resource limits
            mem_limit=f"{config.memory_limit_mb}m",
            cpu_quota=int(config.cpu_limit * 100000),
            cpu_period=100000,
            pids_limit=config.max_processes,
            
            # Network (default: isolated)
            network_mode="none" if not config.allow_network else "bridge",
            
            # Filesystem (read-only root, tmpfs for writes)
            read_only=True,
            tmpfs={
                "/tmp": "rw,noexec,nosuid,size=100m",
                "/var/tmp": "rw,noexec,nosuid,size=100m"
            },
            
            # Volume mounts (read-only)
            volumes={
                config.tool_code_path: {
                    "bind": "/tool",
                    "mode": "ro"
                }
            } if config.tool_code_path else {}
        )
        
        return DockerSandbox(container, execution_id)

class DockerSandbox(Sandbox):
    """Docker container sandbox instance."""
    
    def __init__(self, container, execution_id: str):
        self._container = container
        self._execution_id = execution_id
        self._start_time = time.monotonic()
        
    async def execute_command(
        self,
        command: List[str],
        timeout: int = 60,
        memory_limit: int = 512
    ) -> CommandResult:
        """Execute a command within the sandbox."""
        
        exec_result = self._container.exec_run(
            command,
            stdout=True,
            stderr=True,
            demux=True,
            timeout=timeout
        )
        
        stdout = exec_result.output[0].decode() if exec_result.output[0] else ""
        stderr = exec_result.output[1].decode() if exec_result.output[1] else ""
        
        return CommandResult(
            returncode=exec_result.exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=(time.monotonic() - self._start_time) * 1000
        )
    
    async def write_file(self, path: str, content: str) -> None:
        """Write file to sandbox."""
        # Use docker cp or exec with heredoc
        encoded = base64.b64encode(content.encode()).decode()
        await self.execute_command([
            "sh", "-c", f"echo {encoded} | base64 -d > {path}"
        ])
    
    async def read_file(self, path: str) -> str:
        """Read file from sandbox."""
        result = await self.execute_command(["cat", path])
        if result.returncode != 0:
            raise FileNotFoundError(f"File not found: {path}")
        return result.stdout
    
    async def destroy(self) -> None:
        """Destroy the sandbox container."""
        try:
            self._container.stop(timeout=5)
            self._container.remove(force=True)
        except Exception as e:
            logger.warning(f"Error destroying sandbox: {e}")
```

---

## 6. Dynamic Tool Loading & Hot-Reloading

### 6.1 Hot-Reload Architecture

```python
class HotReloadManager:
    """Manages dynamic tool loading and hot-reloading."""
    
    def __init__(self, registry: ToolRegistry):
        self._registry = registry
        self._loaded_tools: Dict[str, LoadedTool] = {}
        self._watchers: Dict[str, FileWatcher] = {}
        self._event_bus: EventBus = EventBus()
        
    async def initialize(self) -> None:
        """Initialize hot-reload system."""
        # Load all tools from disk
        await self._load_all_tools()
        
        # Start file watchers
        await self._start_watchers()
        
        # Subscribe to events
        self._event_bus.subscribe(ToolEvent.CREATED, self._on_tool_created)
        self._event_bus.subscribe(ToolEvent.MODIFIED, self._on_tool_modified)
        self._event_bus.subscribe(ToolEvent.DELETED, self._on_tool_deleted)
        
    async def load_tool(self, tool_path: Path) -> ToolDefinition:
        """Load a tool from filesystem."""
        
        # Parse tool manifest
        manifest_path = tool_path / "skill.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        # Load implementation
        impl_path = tool_path / manifest.get("entry_point", "index.py")
        with open(impl_path) as f:
            code = f.read()
        
        # Create tool definition
        tool_def = ToolDefinition(
            id=manifest.get("id") or str(uuid.uuid4()),
            name=manifest["name"],
            version=manifest["version"],
            namespace=manifest.get("namespace", "local"),
            description=manifest["description"],
            input_schema=manifest["parameters"],
            implementation=ImplementationSpec(
                type=ImplementationType(manifest.get("type", "python")),
                entry_point=manifest.get("entry_point", "index"),
                code=code,
                checksum=hashlib.sha256(code.encode()).hexdigest()
            )
        )
        
        # Register in registry
        await self._registry.register(tool_def)
        
        # Track loaded tool
        self._loaded_tools[tool_def.id] = LoadedTool(
            definition=tool_def,
            source_path=tool_path,
            loaded_at=datetime.now()
        )
        
        return tool_def
    
    async def hot_reload(self, tool_id: str) -> ReloadResult:
        """Hot-reload a tool without stopping the agent."""
        
        loaded_tool = self._loaded_tools.get(tool_id)
        if not loaded_tool:
            return ReloadResult(success=False, error="Tool not found")
        
        try:
            # 1. Load new version
            new_tool = await self.load_tool(loaded_tool.source_path)
            
            # 2. Validate new version
            validation = await self._validate_tool(new_tool)
            if not validation.valid:
                return ReloadResult(
                    success=False,
                    error=f"Validation failed: {validation.errors}"
                )
            
            # 3. Create shadow registration
            shadow_id = f"{tool_id}@shadow"
            shadow_tool = dataclasses.replace(new_tool, id=shadow_id)
            await self._registry.register(shadow_tool)
            
            # 4. Test shadow version
            test_result = await self._test_tool(shadow_tool)
            if not test_result.success:
                await self._registry.unregister(shadow_id)
                return ReloadResult(
                    success=False,
                    error=f"Shadow test failed: {test_result.error}"
                )
            
            # 5. Atomic swap
            async with self._registry.transaction():
                await self._registry.unregister(tool_id)
                await self._registry.register(dataclasses.replace(new_tool, id=tool_id))
                await self._registry.unregister(shadow_id)
            
            # 6. Update tracking
            self._loaded_tools[tool_id] = LoadedTool(
                definition=new_tool,
                source_path=loaded_tool.source_path,
                loaded_at=datetime.now()
            )
            
            # 7. Notify subscribers
            self._event_bus.publish(ToolEvent.RELOADED, {
                "tool_id": tool_id,
                "old_version": loaded_tool.definition.version,
                "new_version": new_tool.version
            })
            
            return ReloadResult(success=True, tool=new_tool)
            
        except Exception as e:
            logger.exception(f"Hot reload failed for {tool_id}")
            return ReloadResult(success=False, error=str(e))
```

### 6.2 File Watcher Implementation

```python
class ToolFileWatcher:
    """Watches tool directories for changes."""
    
    def __init__(self, hot_reload_manager: HotReloadManager):
        self._manager = hot_reload_manager
        self._observer = Observer()
        self._watch_handlers: Dict[str, ToolEventHandler] = {}
        
    async def start_watching(self, tools_dir: Path) -> None:
        """Start watching tool directories."""
        
        handler = ToolEventHandler(self._manager)
        
        # Watch each tool subdirectory
        for tool_dir in tools_dir.iterdir():
            if tool_dir.is_dir():
                watch = self._observer.schedule(
                    handler,
                    str(tool_dir),
                    recursive=True
                )
                self._watch_handlers[str(tool_dir)] = watch
        
        self._observer.start()
        
    class ToolEventHandler(FileSystemEventHandler):
        """Handles filesystem events for tools."""
        
        def __init__(self, manager: HotReloadManager):
            self._manager = manager
            self._debounce_timers: Dict[str, asyncio.TimerHandle] = {}
            
        def on_modified(self, event):
            if event.is_directory:
                return
            
            tool_path = Path(event.src_path).parent
            tool_id = self._get_tool_id(tool_path)
            
            # Debounce rapid changes
            if tool_id in self._debounce_timers:
                self._debounce_timers[tool_id].cancel()
            
            loop = asyncio.get_event_loop()
            self._debounce_timers[tool_id] = loop.call_later(
                1.0,  # 1 second debounce
                lambda: asyncio.create_task(
                    self._manager.hot_reload(tool_id)
                )
            )
        
        def on_created(self, event):
            if not event.is_directory:
                return
            
            # New tool directory created
            tool_path = Path(event.src_path)
            asyncio.create_task(self._manager.load_tool(tool_path))
```

---

## 7. Permission System & Capability Declarations

### 7.1 Capability-Based Security Model

```python
class Capability(Enum):
    """System capabilities that tools can request."""
    
    # File System
    FILE_READ = "file:read"
    FILE_WRITE = "file:write"
    FILE_DELETE = "file:delete"
    FILE_EXECUTE = "file:execute"
    
    # Network
    NETWORK_OUTBOUND = "network:outbound"
    NETWORK_INBOUND = "network:inbound"
    NETWORK_LOCAL = "network:local"
    
    # System
    SYSTEM_SHELL = "system:shell"
    SYSTEM_PROCESS = "system:process"
    SYSTEM_REGISTRY = "system:registry"
    SYSTEM_SERVICE = "system:service"
    
    # APIs
    API_GMAIL = "api:gmail"
    API_CALENDAR = "api:calendar"
    API_DRIVE = "api:drive"
    API_SLACK = "api:slack"
    API_GITHUB = "api:github"
    API_TWILIO = "api:twilio"
    
    # Hardware
    HW_MICROPHONE = "hw:microphone"
    HW_SPEAKER = "hw:speaker"
    HW_CAMERA = "hw:camera"
    
    # Sensitive
    SENSITIVE_CREDENTIALS = "sensitive:credentials"
    SENSITIVE_PII = "sensitive:pii"
    SENSITIVE_PAYMENT = "sensitive:payment"

class RiskLevel(Enum):
    """Security risk classification for tools."""
    
    READONLY = "readonly"           # Read-only operations, no side effects
    SAFE = "safe"                   # Safe operations with minimal risk
    NORMAL = "normal"               # Standard operations requiring approval
    ELEVATED = "elevated"           # Higher risk, explicit approval required
    DESTRUCTIVE = "destructive"     # Destructive operations, multi-step approval
    CRITICAL = "critical"           # Critical system operations, admin approval
```

### 7.2 Permission Engine

```python
class PermissionEngine:
    """Capability-based permission system."""
    
    def __init__(self):
        self._enforcer: casbin.Enforcer = self._init_enforcer()
        self._user_caps: Dict[str, Set[Capability]] = {}
        self._session_caps: Dict[str, Set[Capability]] = {}
        
    def _init_enforcer(self) -> casbin.Enforcer:
        """Initialize Casbin enforcer with policy model."""
        
        model_text = """
        [request_definition]
        r = sub, obj, act
        
        [policy_definition]
        p = sub, obj, act, eft
        
        [role_definition]
        g = _, _
        g2 = _, _
        
        [policy_effect]
        e = some(where (p.eft == allow)) && !some(where (p.eft == deny))
        
        [matchers]
        m = g(r.sub, p.sub) && g2(r.obj, p.obj) && r.act == p.act
        """
        
        return casbin.Enforcer(
            casbin.Model(model_text),
            self._load_policies()
        )
    
    async def check(
        self,
        tool: ToolDefinition,
        user: User,
        session: Session
    ) -> PermissionResult:
        """Check if tool execution is permitted."""
        
        required_caps = set(tool.required_capabilities)
        
        # Check user capabilities
        user_caps = self._user_caps.get(user.id, set())
        missing_user_caps = required_caps - user_caps
        
        if missing_user_caps:
            return PermissionResult(
                allowed=False,
                reason=f"User lacks capabilities: {missing_user_caps}"
            )
        
        # Check session capabilities
        session_caps = self._session_caps.get(session.id, set())
        missing_session_caps = required_caps - session_caps
        
        if missing_session_caps:
            return PermissionResult(
                allowed=False,
                reason=f"Session lacks capabilities: {missing_session_caps}",
                requires_reauthorization=True
            )
        
        # Check risk level approval
        risk_check = await self._check_risk_approval(tool, user, session)
        if not risk_check.approved:
            return PermissionResult(
                allowed=False,
                reason=f"Risk level approval required: {tool.risk_level.value}"
            )
        
        # Check Casbin policies
        for cap in required_caps:
            if not self._enforcer.enforce(user.id, tool.id, cap.value):
                return PermissionResult(
                    allowed=False,
                    reason=f"Policy denied: {cap.value}"
                )
        
        return PermissionResult(allowed=True)
    
    async def grant_capability(
        self,
        user_id: str,
        capability: Capability,
        expires_at: Optional[datetime] = None,
        constraints: Optional[Dict] = None
    ) -> GrantResult:
        """Grant a capability to a user."""
        
        if user_id not in self._user_caps:
            self._user_caps[user_id] = set()
        
        self._user_caps[user_id].add(capability)
        
        # Store grant with metadata
        grant = CapabilityGrant(
            user_id=user_id,
            capability=capability,
            granted_at=datetime.now(),
            expires_at=expires_at,
            constraints=constraints
        )
        
        await self._persist_grant(grant)
        
        return GrantResult(success=True, grant=grant)
    
    async def request_approval(
        self,
        tool: ToolDefinition,
        user: User,
        context: ApprovalContext
    ) -> ApprovalResult:
        """Request user approval for elevated-risk tool execution."""
        
        # Generate approval request
        request = ApprovalRequest(
            id=str(uuid.uuid4()),
            tool_id=tool.id,
            tool_name=tool.name,
            risk_level=tool.risk_level,
            requested_capabilities=tool.required_capabilities,
            justification=context.justification,
            requested_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=5)
        )
        
        # Send approval notification
        await self._send_approval_notification(user, request)
        
        # Wait for response
        response = await self._wait_for_approval(request.id, timeout=300)
        
        if response and response.approved:
            # Grant temporary capability
            await self.grant_capability(
                user_id=user.id,
                capability=Capability.SENSITIVE_CREDENTIALS,
                expires_at=datetime.now() + timedelta(hours=1)
            )
            
            return ApprovalResult(
                approved=True,
                temporary_grant=True
            )
        
        return ApprovalResult(approved=False, reason="Approval denied or timed out")
```

---

## 8. Self-Improving Tool Generation

### 8.1 Tool Generation Pipeline

```python
class SkillGenerator:
    """AI-powered tool/skill generation system."""
    
    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client
        self._validator = ToolValidator()
        self._test_runner = ToolTestRunner()
        self._registry: ToolRegistry = None
        
    async def generate_skill(
        self,
        request: SkillGenerationRequest
    ) -> GenerationResult:
        """Generate a new skill based on natural language description."""
        
        generation_id = str(uuid.uuid4())
        
        try:
            # 1. Analyze requirements
            analysis = await self._analyze_requirements(request)
            
            # 2. Generate skill specification
            spec = await self._generate_spec(analysis)
            
            # 3. Generate implementation
            implementation = await self._generate_implementation(spec)
            
            # 4. Validate generated code
            validation = await self._validator.validate(implementation)
            if not validation.valid:
                # Retry with error feedback
                implementation = await self._regenerate_with_feedback(
                    spec, validation.errors
                )
                validation = await self._validator.validate(implementation)
            
            # 5. Generate tests
            tests = await self._generate_tests(spec, implementation)
            
            # 6. Run tests in sandbox
            test_results = await self._test_runner.run_tests(
                implementation,
                tests
            )
            
            if not test_results.all_passed:
                # Regenerate with test feedback
                implementation = await self._regenerate_with_test_feedback(
                    spec, implementation, test_results
                )
                test_results = await self._test_runner.run_tests(
                    implementation,
                    tests
                )
            
            # 7. Create tool definition
            tool_def = self._create_tool_definition(
                spec,
                implementation,
                request
            )
            
            # 8. Security scan
            security_scan = await self._security_scan(tool_def)
            if security_scan.risk_score > 0.7:
                return GenerationResult(
                    success=False,
                    error=f"Security risk too high: {security_scan.findings}"
                )
            
            # 9. Register if auto-deploy enabled
            if request.auto_deploy and test_results.all_passed:
                await self._registry.register(tool_def)
                
                return GenerationResult(
                    success=True,
                    tool=tool_def,
                    deployed=True
                )
            
            return GenerationResult(
                success=True,
                tool=tool_def,
                deployed=False,
                requires_approval=True
            )
            
        except Exception as e:
            logger.exception("Skill generation failed")
            return GenerationResult(
                success=False,
                error=str(e),
                generation_id=generation_id
            )
    
    async def _generate_implementation(
        self,
        spec: SkillSpecification
    ) -> SkillImplementation:
        """Generate code implementation from specification."""
        
        prompt = f"""
You are an expert Python developer creating a tool/skill for an AI agent system.

SKILL SPECIFICATION:
Name: {spec.name}
Description: {spec.description}
Purpose: {spec.purpose}

INPUT PARAMETERS:
{json.dumps(spec.parameters, indent=2)}

OUTPUT SCHEMA:
{json.dumps(spec.returns, indent=2)}

EXTERNAL DEPENDENCIES:
{', '.join(spec.dependencies)}

REQUIREMENTS:
1. Implement a single `run()` function that accepts the parameters
2. Include proper error handling with descriptive messages
3. Return results matching the output schema exactly
4. Add docstrings following Google style
5. Include type hints
6. Follow security best practices
7. No hardcoded credentials or secrets
8. Validate all inputs before processing

Generate the complete Python implementation:
"""
        
        response = await self._llm.generate(
            prompt=prompt,
            model="gpt-5.2",
            temperature=0.2,
            max_tokens=4000
        )
        
        # Extract code from response
        code = self._extract_code(response.text)
        
        return SkillImplementation(
            code=code,
            language="python",
            entry_point="run"
        )
    
    async def _generate_tests(
        self,
        spec: SkillSpecification,
        implementation: SkillImplementation
    ) -> List[TestCase]:
        """Generate comprehensive test cases."""
        
        prompt = f"""
Generate pytest test cases for the following skill:

SKILL: {spec.name}
DESCRIPTION: {spec.description}

IMPLEMENTATION:
```python
{implementation.code}
```

INPUT SCHEMA:
{json.dumps(spec.parameters, indent=2)}

Generate tests covering:
1. Happy path with valid inputs
2. Edge cases (empty strings, zero values, etc.)
3. Error cases (invalid inputs, missing required fields)
4. Boundary conditions
5. Type validation

Output only the test code:
"""
        
        response = await self._llm.generate(
            prompt=prompt,
            model="gpt-5.2",
            temperature=0.1
        )
        
        test_code = self._extract_code(response.text)
        
        return [TestCase(
            name=f"test_{spec.name}",
            code=test_code,
            type=TestType.UNIT
        )]
```

### 8.2 Self-Improvement Loop

```python
class SelfImprovementLoop:
    """Continuous self-improvement through tool generation."""
    
    def __init__(self, generator: SkillGenerator, registry: ToolRegistry):
        self._generator = generator
        self._registry = registry
        self._improvement_queue: asyncio.Queue = asyncio.Queue()
        
    async def start(self) -> None:
        """Start the self-improvement loop."""
        while True:
            try:
                # Get next improvement opportunity
                opportunity = await self._improvement_queue.get()
                
                # Analyze and generate improvement
                await self._process_improvement(opportunity)
                
            except Exception as e:
                logger.exception("Self-improvement loop error")
                await asyncio.sleep(60)
    
    async def identify_opportunities(
        self,
        conversation_history: List[Message]
    ) -> List[ImprovementOpportunity]:
        """Identify opportunities for new skills based on conversation."""
        
        prompt = f"""
Analyze the following conversation history and identify:
1. Tasks the agent struggled with or couldn't complete
2. Repeated patterns that suggest a missing tool
3. User requests that required manual workarounds

CONVERSATION HISTORY:
{self._format_history(conversation_history)}

For each opportunity, provide:
- Description of the gap
- Proposed skill name and purpose
- Expected parameters and return values
- Priority (high/medium/low)

Output as JSON array.
"""
        
        response = await self._generator._llm.generate(prompt=prompt)
        
        opportunities = json.loads(response.text)
        
        return [
            ImprovementOpportunity(**opp)
            for opp in opportunities
        ]
    
    async def _process_improvement(
        self,
        opportunity: ImprovementOpportunity
    ) -> None:
        """Process a single improvement opportunity."""
        
        # Generate skill
        request = SkillGenerationRequest(
            description=opportunity.description,
            name=opportunity.proposed_name,
            priority=opportunity.priority,
            auto_deploy=False  # Require approval for self-generated skills
        )
        
        result = await self._generator.generate_skill(request)
        
        if result.success:
            # Queue for human approval
            await self._queue_for_approval(result.tool, opportunity)
        else:
            # Log failure for analysis
            logger.warning(f"Skill generation failed: {result.error}")
```

---

## 9. Tool Marketplace Architecture

### 9.1 ClawHub-Style Marketplace

```python
class SkillMarketplace:
    """Decentralized skill marketplace with verification."""
    
    def __init__(self):
        self._registry: ToolRegistry = None
        self._scanner: SecurityScanner = SecurityScanner()
        self._validator: SkillValidator = SkillValidator()
        self._git_client: GitHubClient = GitHubClient()
        
    async def publish_skill(
        self,
        skill: ToolDefinition,
        publisher: PublisherInfo
    ) -> PublishResult:
        """Publish a skill to the marketplace."""
        
        # 1. Validate skill structure
        validation = await self._validator.validate_structure(skill)
        if not validation.valid:
            return PublishResult(
                success=False,
                error=f"Validation failed: {validation.errors}"
            )
        
        # 2. Security scan
        scan_result = await self._scanner.scan(skill)
        if scan_result.threats_found:
            return PublishResult(
                success=False,
                error=f"Security threats found: {scan_result.threats}",
                blocked=True
            )
        
        # 3. Reputation check
        publisher_rep = await self._check_publisher_reputation(publisher)
        if publisher_rep.score < 0.3:
            # Require additional verification for low-reputation publishers
            await self._queue_for_manual_review(skill, publisher)
            return PublishResult(
                success=False,
                error="Publisher reputation too low, queued for review",
                pending_review=True
            )
        
        # 4. Sign skill
        signed_skill = await self._sign_skill(skill, publisher)
        
        # 5. Publish to registry
        await self._registry.register(signed_skill)
        
        # 6. Update search index
        await self._update_search_index(signed_skill)
        
        return PublishResult(
            success=True,
            skill_id=signed_skill.id,
            verification_badge=scan_result.badge
        )
    
    async def install_skill(
        self,
        skill_id: str,
        user: User,
        version: Optional[str] = None
    ) -> InstallResult:
        """Install a skill from the marketplace."""
        
        # 1. Fetch skill metadata
        skill = await self._registry.get_tool(skill_id)
        if not skill:
            return InstallResult(success=False, error="Skill not found")
        
        # 2. Verify signature
        if not await self._verify_signature(skill):
            return InstallResult(
                success=False,
                error="Signature verification failed"
            )
        
        # 3. Check compatibility
        compatibility = await self._check_compatibility(skill)
        if not compatibility.compatible:
            return InstallResult(
                success=False,
                error=f"Incompatible: {compatibility.issues}"
            )
        
        # 4. Download implementation
        if skill.implementation.source_url:
            code = await self._git_client.download(
                skill.implementation.source_url,
                ref=version or skill.version
            )
            skill = dataclasses.replace(
                skill,
                implementation=dataclasses.replace(
                    skill.implementation,
                    code=code
                )
            )
        
        # 5. Local installation
        install_path = await self._install_locally(skill)
        
        # 6. Load into registry
        await self._registry.register(skill)
        
        return InstallResult(
            success=True,
            skill_id=skill.id,
            install_path=install_path
        )

class SecurityScanner:
    """Multi-layer security scanner for skills."""
    
    def __init__(self):
        self._static_analyzer = StaticAnalyzer()
        self._behavioral_analyzer = BehavioralAnalyzer()
        self._pattern_matcher = PatternMatcher()
        
    async def scan(self, skill: ToolDefinition) -> ScanResult:
        """Perform comprehensive security scan."""
        
        findings = []
        
        # 1. Static code analysis
        static_findings = await self._static_analyzer.analyze(
            skill.implementation.code
        )
        findings.extend(static_findings)
        
        # 2. Behavioral analysis (sandbox)
        behavioral_findings = await self._behavioral_analyzer.analyze(skill)
        findings.extend(behavioral_findings)
        
        # 3. Pattern matching for known threats
        pattern_findings = self._pattern_matcher.scan(
            skill.implementation.code
        )
        findings.extend(pattern_findings)
        
        # 4. Dependency scanning
        dep_findings = await self._scan_dependencies(skill.dependencies)
        findings.extend(dep_findings)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(findings)
        
        return ScanResult(
            findings=findings,
            risk_score=risk_score,
            threats_found=any(f.severity == Severity.CRITICAL for f in findings),
            badge=self._assign_badge(risk_score, findings)
        )
    
    def _calculate_risk_score(self, findings: List[Finding]) -> float:
        """Calculate overall risk score from findings."""
        
        severity_weights = {
            Severity.INFO: 0.1,
            Severity.LOW: 0.3,
            Severity.MEDIUM: 0.5,
            Severity.HIGH: 0.8,
            Severity.CRITICAL: 1.0
        }
        
        if not findings:
            return 0.0
        
        total_score = sum(
            severity_weights[f.severity]
            for f in findings
        )
        
        return min(total_score / len(findings), 1.0)
```

---

## 10. Integration Patterns

### 10.1 Gmail Integration Example

```python
class GmailSkill:
    """Gmail integration skill for WinClaw."""
    
    def __init__(self):
        self._credentials: GmailCredentials = None
        self._service = None
        
    @tool_schema({
        "name": "gmail_send_email",
        "description": "Send an email via Gmail API",
        "parameters": {
            "to": {"type": "string", "description": "Recipient email"},
            "subject": {"type": "string", "description": "Email subject"},
            "body": {"type": "string", "description": "Email body"},
            "cc": {"type": "string", "optional": True},
            "attachments": {"type": "array", "optional": True}
        },
        "required_capabilities": ["api:gmail", "network:outbound"],
        "risk_level": "normal"
    })
    async def send_email(self, **kwargs) -> Dict:
        """Send an email using Gmail API."""
        
        # Build message
        message = self._create_message(**kwargs)
        
        # Send via Gmail API
        sent = self._service.users().messages().send(
            userId='me',
            body=message
        ).execute()
        
        return {
            "message_id": sent['id'],
            "thread_id": sent['threadId'],
            "success": True
        }
    
    @tool_schema({
        "name": "gmail_search",
        "description": "Search emails in Gmail",
        "parameters": {
            "query": {"type": "string", "description": "Gmail search query"},
            "max_results": {"type": "integer", "default": 10}
        },
        "required_capabilities": ["api:gmail", "network:outbound"],
        "risk_level": "readonly"
    })
    async def search_emails(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search emails in Gmail."""
        
        results = self._service.users().messages().list(
            userId='me',
            q=query,
            maxResults=max_results
        ).execute()
        
        messages = []
        for msg in results.get('messages', []):
            detail = self._service.users().messages().get(
                userId='me',
                id=msg['id']
            ).execute()
            messages.append(self._parse_message(detail))
        
        return messages
```

### 10.2 Browser Control Integration

```python
class BrowserSkill:
    """Browser automation skill using Playwright."""
    
    def __init__(self):
        self._playwright = None
        self._browser = None
        self._context = None
        
    async def initialize(self):
        """Initialize browser instance."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch()
        self._context = await self._browser.new_context()
    
    @tool_schema({
        "name": "browser_navigate",
        "description": "Navigate to a URL",
        "parameters": {
            "url": {"type": "string", "description": "URL to navigate to"},
            "wait_for": {"type": "string", "optional": True, "description": "Selector to wait for"}
        },
        "required_capabilities": ["network:outbound"],
        "risk_level": "safe"
    })
    async def navigate(self, url: str, wait_for: Optional[str] = None) -> Dict:
        """Navigate browser to URL."""
        
        page = await self._context.new_page()
        response = await page.goto(url)
        
        if wait_for:
            await page.wait_for_selector(wait_for)
        
        return {
            "url": page.url,
            "status": response.status,
            "title": await page.title()
        }
    
    @tool_schema({
        "name": "browser_extract",
        "description": "Extract data from current page",
        "parameters": {
            "selector": {"type": "string", "description": "CSS selector for elements"},
            "attribute": {"type": "string", "optional": True, "description": "Attribute to extract"},
            "multiple": {"type": "boolean", "default": False}
        },
        "required_capabilities": ["network:outbound"],
        "risk_level": "readonly"
    })
    async def extract(self, selector: str, attribute: Optional[str] = None, multiple: bool = False) -> Union[str, List[str]]:
        """Extract data from current page."""
        
        page = self._context.pages[0]
        
        if multiple:
            elements = await page.query_selector_all(selector)
            results = []
            for el in elements:
                if attribute:
                    results.append(await el.get_attribute(attribute))
                else:
                    results.append(await el.text_content())
            return results
        else:
            element = await page.query_selector(selector)
            if attribute:
                return await element.get_attribute(attribute)
            return await element.text_content()
```

### 10.3 Twilio Voice/SMS Integration

```python
class TwilioSkill:
    """Twilio voice and SMS integration."""
    
    def __init__(self):
        self._client = None
        
    async def initialize(self, account_sid: str, auth_token: str):
        """Initialize Twilio client."""
        from twilio.rest import Client
        self._client = Client(account_sid, auth_token)
    
    @tool_schema({
        "name": "twilio_send_sms",
        "description": "Send SMS message",
        "parameters": {
            "to": {"type": "string", "description": "Recipient phone number"},
            "body": {"type": "string", "description": "Message body"},
            "from_number": {"type": "string", "optional": True}
        },
        "required_capabilities": ["api:twilio", "network:outbound"],
        "risk_level": "normal"
    })
    async def send_sms(self, to: str, body: str, from_number: Optional[str] = None) -> Dict:
        """Send SMS via Twilio."""
        
        message = self._client.messages.create(
            to=to,
            from_=from_number or os.getenv('TWILIO_PHONE_NUMBER'),
            body=body
        )
        
        return {
            "message_sid": message.sid,
            "status": message.status,
            "success": True
        }
    
    @tool_schema({
        "name": "twilio_make_call",
        "description": "Make voice call with TTS",
        "parameters": {
            "to": {"type": "string", "description": "Recipient phone number"},
            "message": {"type": "string", "description": "Message to speak"},
            "voice": {"type": "string", "default": "alice", "enum": ["alice", "man", "woman"]}
        },
        "required_capabilities": ["api:twilio", "network:outbound", "hw:speaker"],
        "risk_level": "elevated"
    })
    async def make_call(self, to: str, message: str, voice: str = "alice") -> Dict:
        """Make voice call with TTS message."""
        
        # Generate TwiML for TTS
        twiml = f"""
        <?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Say voice="{voice}">{message}</Say>
        </Response>
        """
        
        call = self._client.calls.create(
            to=to,
            from_=os.getenv('TWILIO_PHONE_NUMBER'),
            twiml=twiml
        )
        
        return {
            "call_sid": call.sid,
            "status": call.status,
            "success": True
        }
```

---

## 11. Security Model

### 11.1 Defense in Depth

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SECURITY LAYERS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Layer 1: Code Signing & Verification                                       │
│  ├── All tools must be cryptographically signed                             │
│  ├── Signature verification before execution                                │
│  └── Publisher identity validation                                          │
│                                                                             │
│  Layer 2: Static Analysis                                                   │
│  ├── AST-based code analysis                                                │
│  ├── Dangerous pattern detection                                            │
│  ├── Dependency vulnerability scanning                                      │
│  └── Secret/credential detection                                            │
│                                                                             │
│  Layer 3: Permission System                                                 │
│  ├── Capability-based access control                                        │
│  ├── Risk-level classification                                              │
│  ├── User approval workflows                                                │
│  └── Session-based capability grants                                        │
│                                                                             │
│  Layer 4: Sandbox Execution                                                 │
│  ├── Docker containerization                                                │
│  ├── seccomp-bpf syscall filtering                                          │
│  ├── Resource limits (CPU, memory, network)                                 │
│  └── Filesystem isolation                                                   │
│                                                                             │
│  Layer 5: Runtime Monitoring                                                │
│  ├── Syscall monitoring                                                     │
│  ├── Network traffic analysis                                               │
│  ├── File access auditing                                                   │
│  └── Anomaly detection                                                      │
│                                                                             │
│  Layer 6: Audit & Compliance                                                │
│  ├── Complete execution logging                                             │
│  ├── Immutable audit trails                                                 │
│  └── Compliance reporting                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 11.2 Security Configuration

```yaml
# security-config.yaml
security:
  # Code signing
  signing:
    required: true
    algorithms: ["ed25519", "rsa-pss-sha256"]
    trusted_keys_path: "./trusted_keys"
    
  # Static analysis
  static_analysis:
    enabled: true
    tools:
      - bandit
      - semgrep
      - safety
    fail_on:
      - CRITICAL
      - HIGH
      
  # Sandbox configuration
  sandbox:
    type: "docker"
    image: "winclaw-sandbox:latest"
    security_opts:
      - "no-new-privileges:true"
      - "seccomp=./seccomp-profile.json"
    cap_drop: ["ALL"]
    cap_add: ["NET_BIND_SERVICE"]
    resource_limits:
      memory: "512m"
      cpus: "1.0"
      pids: 50
      
  # Permission defaults
  permissions:
    default_risk_level: "normal"
    require_approval_for:
      - "destructive"
      - "critical"
    session_timeout: 3600
    
  # Monitoring
  monitoring:
    syscall_logging: true
    network_monitoring: true
    file_access_audit: true
    anomaly_detection: true
```

---

## 12. Implementation Examples

### 12.1 Complete Tool Definition Example

```json
{
  "id": "winclaw.gmail.send_email.v1",
  "name": "gmail_send_email",
  "version": "1.2.0",
  "namespace": "winclaw.gmail",
  "description": "Send emails through Gmail API with support for attachments and HTML formatting",
  "long_description": "This skill allows the agent to send emails via Gmail. It supports...",
  "tags": ["email", "gmail", "communication", "productivity"],
  "category": "communication",
  
  "input_schema": {
    "type": "object",
    "properties": {
      "to": {
        "type": "string",
        "description": "Recipient email address(es), comma-separated for multiple",
        "format": "email"
      },
      "subject": {
        "type": "string",
        "description": "Email subject line",
        "maxLength": 998
      },
      "body": {
        "type": "string",
        "description": "Email body content"
      },
      "cc": {
        "type": "string",
        "description": "CC recipients"
      },
      "attachments": {
        "type": "array",
        "items": {"type": "string"},
        "maxItems": 25,
        "description": "File paths to attach"
      },
      "is_html": {
        "type": "boolean",
        "default": false,
        "description": "Whether body is HTML"
      }
    },
    "required": ["to", "subject", "body"]
  },
  
  "output_schema": {
    "type": "object",
    "properties": {
      "message_id": {"type": "string"},
      "thread_id": {"type": "string"},
      "sent_at": {"type": "string", "format": "date-time"},
      "success": {"type": "boolean"}
    }
  },
  
  "implementation": {
    "type": "python",
    "entry_point": "gmail_skill.send_email",
    "source_url": "https://github.com/winclaw/skills/gmail",
    "checksum": "sha256:abc123...",
    "sandbox_config": {
      "network_access": true,
      "allowed_hosts": ["*.googleapis.com", "*.google.com"],
      "memory_limit_mb": 256,
      "max_execution_time": 30
    }
  },
  
  "dependencies": [
    {"name": "google-api-python-client", "version": ">=2.0.0"},
    {"name": "google-auth-httplib2", "version": ">=0.1.0"}
  ],
  
  "required_capabilities": [
    "api:gmail",
    "network:outbound",
    "file:read"
  ],
  "risk_level": "normal",
  
  "author": {
    "name": "WinClaw Team",
    "email": "team@winclaw.ai",
    "url": "https://winclaw.ai"
  },
  
  "signature": "-----BEGIN SIGNATURE-----\n...\n-----END SIGNATURE-----"
}
```

### 12.2 Agentic Loop Integration

```python
class AgenticLoop:
    """15 hardcoded agentic loops for WinClaw."""
    
    def __init__(self, tool_registry: ToolRegistry, llm: LLMClient):
        self._registry = tool_registry
        self._llm = llm
        self._loops: Dict[str, Callable] = {
            "conversation": self._conversation_loop,
            "task_execution": self._task_execution_loop,
            "monitoring": self._monitoring_loop,
            "research": self._research_loop,
            "coding": self._coding_loop,
            "communication": self._communication_loop,
            "scheduling": self._scheduling_loop,
            "data_analysis": self._data_analysis_loop,
            "system_admin": self._system_admin_loop,
            "learning": self._learning_loop,
            "creative": self._creative_loop,
            "planning": self._planning_loop,
            "troubleshooting": self._troubleshooting_loop,
            "optimization": self._optimization_loop,
            "security": self._security_loop
        }
        
    async def execute_loop(
        self,
        loop_name: str,
        context: LoopContext
    ) -> LoopResult:
        """Execute a specific agentic loop."""
        
        loop = self._loops.get(loop_name)
        if not loop:
            raise ValueError(f"Unknown loop: {loop_name}")
        
        return await loop(context)
    
    async def _task_execution_loop(self, context: LoopContext) -> LoopResult:
        """Execute a multi-step task with tool orchestration."""
        
        # 1. Plan task
        plan = await self._plan_task(context.goal)
        
        # 2. Execute steps
        results = []
        for step in plan.steps:
            # Discover appropriate tools
            tools = await self._registry.discover(
                DiscoveryQuery(
                    semantic_query=step.description,
                    capabilities=step.required_capabilities
                )
            )
            
            # Select best tool
            tool = await self._select_tool(tools, step)
            
            # Execute with LLM guidance
            result = await self._execute_with_llm(tool, step, context)
            results.append(result)
            
            # Check if we need to adjust plan
            if not result.success:
                plan = await self._replan(plan, step, result)
        
        return LoopResult(
            success=all(r.success for r in results),
            results=results
        )
```

---

## Appendix A: Configuration Reference

### A.1 Tool Registry Configuration

```yaml
registry:
  backend: "sqlite"  # sqlite, postgresql, redis
  connection_string: "./data/registry.db"
  cache:
    enabled: true
    ttl: 300
    max_size: 1000
  indexing:
    semantic_search: true
    vector_dimension: 1536
    embedding_model: "text-embedding-3-large"
```

### A.2 Execution Engine Configuration

```yaml
execution:
  sandbox:
    type: "docker"
    default_image: "winclaw-sandbox:latest"
    resource_limits:
      default_memory: "512m"
      default_cpu: "1.0"
      default_timeout: 60
  queue:
    type: "asyncio"
    max_concurrent: 10
  retry:
    max_attempts: 3
    backoff: "exponential"
```

---

## Appendix B: API Reference

### B.1 REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/tools` | GET | List all tools |
| `/api/v1/tools` | POST | Register new tool |
| `/api/v1/tools/{id}` | GET | Get tool details |
| `/api/v1/tools/{id}` | PUT | Update tool |
| `/api/v1/tools/{id}` | DELETE | Unregister tool |
| `/api/v1/tools/{id}/execute` | POST | Execute tool |
| `/api/v1/tools/discover` | POST | Discover tools |
| `/api/v1/marketplace/search` | GET | Search marketplace |
| `/api/v1/marketplace/install` | POST | Install from marketplace |

### B.2 MCP Protocol Endpoints

| Method | Description |
|--------|-------------|
| `tools/list` | List available tools |
| `tools/call` | Execute a tool |
| `tools/validate` | Validate tool schema |

---

**End of Specification**

*This document provides the complete technical specification for the WinClaw Tool System and Plugin Architecture. For implementation details, refer to the accompanying code repository and developer documentation.*
