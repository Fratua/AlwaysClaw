# Windows 10 File System & Registry Access Technical Specification
## OpenClaw-Inspired AI Agent Framework

**Version:** 1.0.0  
**Platform:** Windows 10/11 (NTFS)  
**Framework:** OpenClaw AI Agent System  
**Date:** 2025-01-20

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [NTFS File Operations](#2-ntfs-file-operations)
3. [Windows Path Handling](#3-windows-path-handling)
4. [File Permission & ACL Management](#4-file-permission--acl-management)
5. [Registry Hive Structure](#5-registry-hive-structure)
6. [Registry Operations](#6-registry-operations)
7. [File Watching & Notifications](#7-file-watching--notifications)
8. [Temporary File Management](#8-temporary-file-management)
9. [Recycle Bin Operations](#9-recycle-bin-operations)
10. [Security & Sandboxing](#10-security--sandboxing)
11. [Error Handling](#11-error-handling)
12. [Implementation Examples](#12-implementation-examples)

---

## 1. Architecture Overview

### 1.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Agent Core (GPT-5.2)                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   File Ops   │  │  Registry    │  │   Security   │           │
│  │   Module     │  │   Module     │  │   Module     │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
├─────────┼─────────────────┼─────────────────┼───────────────────┤
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌──────┴───────┐           │
│  │  NTFS API    │  │  WinReg API  │  │  ACL API     │           │
│  │  (Kernel32)  │  │  (Advapi32)  │  │  (Advapi32)  │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
├─────────┴─────────────────┴─────────────────┴───────────────────┤
│                    Windows NT Kernel                              │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Components

| Component | DLL | Purpose |
|-----------|-----|---------|
| File Operations | `kernel32.dll`, `ntdll.dll` | NTFS read/write/operations |
| Registry Access | `advapi32.dll` | Registry hive operations |
| ACL Management | `advapi32.dll` | Permission control |
| Path Handling | `shlwapi.dll`, `shell32.dll` | Special folder resolution |
| File Watching | `kernel32.dll` | Change notifications |

---

## 2. NTFS File Operations

### 2.1 File Operation Types

```python
class NTFSFileOperation(Enum):
    READ = 0x0001          # GENERIC_READ
    WRITE = 0x0002         # GENERIC_WRITE
    EXECUTE = 0x0020       # GENERIC_EXECUTE
    DELETE = 0x10000       # DELETE
    READ_ATTRIBUTES = 0x0080
    WRITE_ATTRIBUTES = 0x0100
```

### 2.2 File Access API

```python
class NTFSFileManager:
    """
    NTFS File Operations Manager
    Provides atomic, transactional file operations
    """
    
    # Windows API Constants
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    FILE_SHARE_DELETE = 0x00000004
    
    CREATE_NEW = 1
    CREATE_ALWAYS = 2
    OPEN_EXISTING = 3
    OPEN_ALWAYS = 4
    TRUNCATE_EXISTING = 5
    
    FILE_ATTRIBUTE_NORMAL = 0x80
    FILE_ATTRIBUTE_HIDDEN = 0x02
    FILE_ATTRIBUTE_SYSTEM = 0x04
    FILE_ATTRIBUTE_READONLY = 0x01
    FILE_ATTRIBUTE_TEMPORARY = 0x100
    
    FILE_FLAG_DELETE_ON_CLOSE = 0x04000000
    FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
    FILE_FLAG_RANDOM_ACCESS = 0x10000000
    FILE_FLAG_OVERLAPPED = 0x40000000
    
    def __init__(self, agent_context: AgentContext):
        self.context = agent_context
        self.transaction_manager = TransactionManager()
        self.permission_manager = PermissionManager()
        self.logger = logging.getLogger("NTFSFileManager")
        
    def read_file(
        self, 
        path: str, 
        encoding: str = 'utf-8',
        binary: bool = False,
        offset: int = 0,
        size: int = -1
    ) -> Union[str, bytes]:
        """
        Read file contents with optional range support
        
        Args:
            path: Full NTFS path (supports UNC)
            encoding: Text encoding for non-binary reads
            binary: Read as binary data
            offset: Starting byte offset
            size: Bytes to read (-1 for all)
            
        Returns:
            File contents as string or bytes
            
        Raises:
            FileNotFoundError: Path doesn't exist
            PermissionError: Insufficient access rights
            IOError: I/O operation failed
        """
        pass
        
    def write_file(
        self,
        path: str,
        content: Union[str, bytes],
        encoding: str = 'utf-8',
        append: bool = False,
        atomic: bool = True,
        create_dirs: bool = True
    ) -> bool:
        """
        Write content to file with atomic option
        
        Args:
            path: Target file path
            content: Data to write
            encoding: Text encoding
            append: Append to existing file
            atomic: Use transactional write
            create_dirs: Create parent directories
            
        Returns:
            True if successful
        """
        pass
        
    def delete_file(
        self,
        path: str,
        secure: bool = False,
        move_to_recycle: bool = True
    ) -> bool:
        """
        Delete file with optional secure deletion
        
        Args:
            path: File to delete
            secure: Overwrite before deletion
            move_to_recycle: Move to recycle bin instead
            
        Returns:
            True if successful
        """
        pass
        
    def copy_file(
        self,
        source: str,
        destination: str,
        overwrite: bool = False,
        preserve_acl: bool = True,
        preserve_timestamps: bool = True
    ) -> bool:
        """
        Copy file with ACL and timestamp preservation
        
        Args:
            source: Source file path
            destination: Destination path
            overwrite: Allow overwriting
            preserve_acl: Copy access control list
            preserve_timestamps: Copy creation/modification times
            
        Returns:
            True if successful
        """
        pass
        
    def move_file(
        self,
        source: str,
        destination: str,
        overwrite: bool = False,
        allow_cross_volume: bool = True
    ) -> bool:
        """
        Move/rename file within or across volumes
        
        Args:
            source: Source file path
            destination: Destination path
            overwrite: Allow overwriting destination
            allow_cross_volume: Allow cross-volume moves
            
        Returns:
            True if successful
        """
        pass
        
    def get_file_info(self, path: str) -> FileInfo:
        """
        Get comprehensive file information
        
        Returns:
            FileInfo object with metadata
        """
        pass
```

### 2.3 File Information Structure

```python
@dataclass
class FileInfo:
    """Comprehensive NTFS file information"""
    
    # Basic Info
    name: str
    full_path: str
    size: int
    
    # NTFS Timestamps (100-nanosecond intervals since 1601-01-01)
    creation_time: datetime
    last_access_time: datetime
    last_write_time: datetime
    change_time: datetime
    
    # Attributes
    attributes: FileAttributes
    is_directory: bool
    is_reparse_point: bool
    is_compressed: bool
    is_encrypted: bool
    
    # NTFS Specific
    file_index: int           # MFT entry number
    hard_link_count: int      # Number of hard links
    volume_serial: int
    
    # Streams
    streams: List[StreamInfo]  # ADS (Alternate Data Streams)
    
    # Security
    owner: str
    group: str
    acl: ACL
    
    # Extended
    reparse_tag: Optional[int]
    ea_size: int              # Extended attributes size

@dataclass
class StreamInfo:
    """NTFS Alternate Data Stream information"""
    name: str
    size: int
    allocation_size: int
```

### 2.4 Directory Operations

```python
class DirectoryManager:
    """NTFS Directory Operations"""
    
    def create_directory(
        self,
        path: str,
        create_parents: bool = True,
        security_descriptor: Optional[SecurityDescriptor] = None
    ) -> bool:
        """Create directory with optional security descriptor"""
        pass
        
    def delete_directory(
        self,
        path: str,
        recursive: bool = False,
        move_to_recycle: bool = True
    ) -> bool:
        """Delete directory, optionally recursively"""
        pass
        
    def list_directory(
        self,
        path: str,
        pattern: str = "*",
        include_hidden: bool = False,
        include_system: bool = False,
        recursive: bool = False
    ) -> Iterator[FileInfo]:
        """List directory contents with filtering"""
        pass
        
    def get_directory_size(self, path: str) -> int:
        """Calculate total directory size recursively"""
        pass
        
    def ensure_directory(self, path: str) -> bool:
        """Create directory if it doesn't exist"""
        pass
```

### 2.5 Advanced NTFS Features

```python
class NTFSAdvancedFeatures:
    """Advanced NTFS-specific operations"""
    
    # Alternate Data Streams (ADS)
    def create_stream(
        self,
        file_path: str,
        stream_name: str,
        content: bytes
    ) -> bool:
        """Create alternate data stream"""
        pass
        
    def read_stream(
        self,
        file_path: str,
        stream_name: str
    ) -> bytes:
        """Read alternate data stream"""
        pass
        
    def delete_stream(self, file_path: str, stream_name: str) -> bool:
        """Delete alternate data stream"""
        pass
        
    def list_streams(self, file_path: str) -> List[StreamInfo]:
        """List all streams including ADS"""
        pass
        
    # Hard Links
    def create_hard_link(self, source: str, link_path: str) -> bool:
        """Create hard link to file"""
        pass
        
    def get_hard_links(self, file_path: str) -> List[str]:
        """Get all hard links to a file"""
        pass
        
    # Symbolic Links & Junctions
    def create_symbolic_link(
        self,
        link_path: str,
        target: str,
        directory: bool = False
    ) -> bool:
        """Create symbolic link (requires elevation)"""
        pass
        
    def create_junction(self, junction_path: str, target: str) -> bool:
        """Create directory junction"""
        pass
        
    def get_reparse_point(self, path: str) -> Optional[str]:
        """Get reparse point target"""
        pass
        
    # Compression
    def set_compression(self, path: str, compress: bool = True) -> bool:
        """Enable/disable NTFS compression"""
        pass
        
    def is_compressed(self, path: str) -> bool:
        """Check if file/directory is compressed"""
        pass
        
    # Encryption (EFS)
    def encrypt_file(self, path: str) -> bool:
        """Encrypt file using EFS"""
        pass
        
    def decrypt_file(self, path: str) -> bool:
        """Decrypt EFS-encrypted file"""
        pass
        
    def is_encrypted(self, path: str) -> bool:
        """Check if file is EFS-encrypted"""
        pass
        
    # Sparse Files
    def set_sparse(self, path: str) -> bool:
        """Mark file as sparse"""
        pass
        
    def set_sparse_range(
        self,
        path: str,
        offset: int,
        length: int,
        allocate: bool
    ) -> bool:
        """Set sparse file range allocation"""
        pass
```

---

## 3. Windows Path Handling

### 3.1 Special Folder Constants

```python
class WindowsSpecialFolders:
    """Windows 10 Special Folder Identifiers"""
    
    # User Profile
    USER_PROFILE = "{userprofile}"
    APPDATA = "{appdata}"
    LOCAL_APPDATA = "{localappdata}"
    TEMP = "{temp}"
    
    # Documents
    DOCUMENTS = "{documents}"
    DOWNLOADS = "{downloads}"
    DESKTOP = "{desktop}"
    PICTURES = "{pictures}"
    MUSIC = "{music}"
    VIDEOS = "{videos}"
    
    # System
    WINDOWS = "{windows}"
    SYSTEM32 = "{system32}"
    SYSWOW64 = "{syswow64}"
    PROGRAM_FILES = "{programfiles}"
    PROGRAM_FILES_X86 = "{programfiles(x86)}"
    PROGRAM_DATA = "{programdata}"
    PUBLIC = "{public}"
    
    # Application Data
    COMMON_APPDATA = "{commonappdata}"
    COMMON_DOCUMENTS = "{commondocuments}"
    COMMON_DESKTOP = "{commondesktop}"
    
    # Recent & History
    RECENT = "{recent}"
    SENDTO = "{sendto}"
    STARTUP = "{startup}"
    COMMON_STARTUP = "{commonstartup}"
    
    # Shell Folders (CSIDL equivalents)
    CSIDL_PERSONAL = 0x0005
    CSIDL_APPDATA = 0x001a
    CSIDL_LOCAL_APPDATA = 0x001c
    CSIDL_WINDOWS = 0x0024
    CSIDL_SYSTEM = 0x0025
    CSIDL_PROGRAM_FILES = 0x0026
    CSIDL_MYPICTURES = 0x0027
    CSIDL_PROFILE = 0x0028
    CSIDL_PROGRAM_FILES_COMMON = 0x002b
    CSIDL_COMMON_APPDATA = 0x0023
    CSIDL_COMMON_DOCUMENTS = 0x002e

class KnownFolderIDs:
    """Windows Known Folder IDs (Vista+)"""
    
    # FOLDERID constants as GUIDs
    FOLDERID_Desktop = "B4BFCC3A-DB2C-424C-B029-7FE99A87C641"
    FOLDERID_Documents = "FDD39AD0-238F-46AF-ADB4-6C85480369C7"
    FOLDERID_Downloads = "374DE290-123F-4565-9164-39C4925E467B"
    FOLDERID_Pictures = "33E28130-4E1E-4676-835A-98395C3BC3BB"
    FOLDERID_Music = "4BD8D571-6D19-48D3-BE97-422220080E43"
    FOLDERID_Videos = "18989B1D-99B5-455B-841C-AB7C74E4DDFC"
    FOLDERID_RoamingAppData = "3EB685DB-65F9-4CF6-A03A-E3EF65729F3D"
    FOLDERID_LocalAppData = "F1B32785-6FBA-4FCF-9D55-7B8E7F157091"
    FOLDERID_LocalAppDataLow = "A520A1A4-1780-4FF6-BD18-167343C5AF16"
    FOLDERID_ProgramData = "62AB5D82-FDC1-4DC3-A9DD-070D1D495D97"
    FOLDERID_ProgramFiles = "905E63B6-C1BF-494E-B29C-65B732D3D21A"
    FOLDERID_ProgramFilesX86 = "7C5A40EF-A0FB-4BFC-874A-C0F2E0B9FA8E"
    FOLDERID_Windows = "F38BF404-1D43-42F2-9305-67DE0B28FC23"
    FOLDERID_System = "1AC14E77-02E7-4E5D-B744-2EB1AE5198B7"
    FOLDERID_SystemX86 = "D65231B0-B2F1-4857-A4CE-A8E7C6EA7D27"
```

### 3.2 Path Manager

```python
class WindowsPathManager:
    """
    Windows Path Resolution and Management
    Handles special folders, environment variables, and path normalization
    """
    
    # Path prefixes
    LONG_PATH_PREFIX = "\\\\?\\"          # Extended-length path
    UNC_PREFIX = "\\\\?\\UNC\\"         # Extended UNC path
    DEVICE_PATH_PREFIX = "\\\\.\\"      # Device path
    
    # Reserved names (cannot be used as filenames)
    RESERVED_NAMES = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
    }
    
    # Invalid characters
    INVALID_CHARS = '<>:"/\\|?*'
    INVALID_CHARS_REGEX = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
    
    def __init__(self):
        self._folder_cache: Dict[str, str] = {}
        self._load_known_folders()
        
    def _load_known_folders(self):
        """Pre-load known folder paths"""
        # Implementation uses SHGetKnownFolderPath
        pass
        
    def resolve_path(self, path: str, expand_env: bool = True) -> str:
        """
        Resolve path with environment variables and special folders
        
        Args:
            path: Path string (may contain env vars and special folder tokens)
            expand_env: Expand environment variables
            
        Returns:
            Absolute, normalized path
            
        Examples:
            "{appdata}\\MyApp\\config.json" -> "C:\\Users\\User\\AppData\\Roaming\\MyApp\\config.json"
            "%TEMP%\\tempfile.tmp" -> "C:\\Users\\User\\AppData\\Local\\Temp\\tempfile.tmp"
        """
        pass
        
    def get_special_folder(self, folder_id: Union[str, int]) -> str:
        """
        Get special folder path by ID or name
        
        Args:
            folder_id: Folder identifier (CSIDL, GUID, or token)
            
        Returns:
            Absolute folder path
        """
        pass
        
    def normalize_path(self, path: str) -> str:
        """
        Normalize path for Windows
        - Convert forward slashes to backslashes
        - Remove redundant separators
        - Resolve . and ..
        - Add long path prefix if needed
        
        Args:
            path: Input path
            
        Returns:
            Normalized path
        """
        pass
        
    def to_long_path(self, path: str) -> str:
        """
        Convert to extended-length path (\\?\\ prefix)
        Required for paths > 260 characters
        
        Args:
            path: Normal path
            
        Returns:
            Extended-length path
        """
        pass
        
    def from_long_path(self, path: str) -> str:
        """Remove extended-length path prefix"""
        pass
        
    def is_valid_filename(self, name: str) -> bool:
        """
        Check if filename is valid for Windows
        
        Checks:
        - Not empty or whitespace
        - Not a reserved name
        - No invalid characters
        - Not too long (>255 chars)
        - Doesn't end with space or period
        """
        pass
        
    def sanitize_filename(self, name: str, replacement: str = "_") -> str:
        """
        Sanitize filename for Windows
        
        Args:
            name: Original filename
            replacement: Character to replace invalid chars with
            
        Returns:
            Sanitized filename
        """
        pass
        
    def join_path(self, *parts: str) -> str:
        """Join path components correctly for Windows"""
        pass
        
    def split_path(self, path: str) -> Tuple[str, str]:
        """Split path into (directory, filename)"""
        pass
        
    def get_relative_path(self, path: str, base: str) -> str:
        """Get relative path from base"""
        pass
        
    def get_drive_type(self, path: str) -> DriveType:
        """Get drive type (fixed, removable, network, etc.)"""
        pass
        
    def get_volume_info(self, path: str) -> VolumeInfo:
        """Get volume information"""
        pass
```

### 3.3 Path Resolution Examples

```python
# Path Resolution Mapping
PATH_RESOLUTION_EXAMPLES = {
    # Environment Variables
    "%USERPROFILE%": r"C:\Users\{username}",
    "%APPDATA%": r"C:\Users\{username}\AppData\Roaming",
    "%LOCALAPPDATA%": r"C:\Users\{username}\AppData\Local",
    "%TEMP%": r"C:\Users\{username}\AppData\Local\Temp",
    "%SYSTEMROOT%": r"C:\Windows",
    "%PROGRAMFILES%": r"C:\Program Files",
    "%PROGRAMDATA%": r"C:\ProgramData",
    
    # Special Folder Tokens
    "{userprofile}": r"C:\Users\{username}",
    "{appdata}": r"C:\Users\{username}\AppData\Roaming",
    "{localappdata}": r"C:\Users\{username}\AppData\Local",
    "{temp}": r"C:\Users\{username}\AppData\Local\Temp",
    "{documents}": r"C:\Users\{username}\Documents",
    "{downloads}": r"C:\Users\{username}\Downloads",
    "{desktop}": r"C:\Users\{username}\Desktop",
    "{pictures}": r"C:\Users\{username}\Pictures",
    "{music}": r"C:\Users\{username}\Music",
    "{videos}": r"C:\Users\{username}\Videos",
    "{windows}": r"C:\Windows",
    "{system32}": r"C:\Windows\System32",
    "{programfiles}": r"C:\Program Files",
    "{programfiles(x86)}": r"C:\Program Files (x86)",
    "{programdata}": r"C:\ProgramData",
    
    # Agent-specific paths
    "{agent_root}": r"{localappdata}\OpenClawAgent",
    "{agent_config}": r"{localappdata}\OpenClawAgent\config",
    "{agent_data}": r"{localappdata}\OpenClawAgent\data",
    "{agent_logs}": r"{localappdata}\OpenClawAgent\logs",
    "{agent_cache}": r"{localappdata}\OpenClawAgent\cache",
    "{agent_temp}": r"{localappdata}\OpenClawAgent\temp",
    "{agent_registry}": r"HKCU\Software\OpenClawAgent",
}
```

---

## 4. File Permission & ACL Management

### 4.1 Security Descriptor Structure

```python
@dataclass
class SecurityDescriptor:
    """Windows Security Descriptor"""
    
    owner: str                      # SID string
    group: str                      # SID string
    dacl: Optional[DACL]            # Discretionary ACL
    sacl: Optional[SACL]            # System ACL
    control_flags: int              # SE_DACL_PRESENT, etc.

@dataclass
class ACL:
    """Access Control List"""
    
    entries: List[ACE]
    revision: int = 2
    
    def add_entry(self, ace: ACE) -> None:
        """Add ACE to ACL"""
        pass
        
    def remove_entry(self, index: int) -> None:
        """Remove ACE by index"""
        pass
        
    def find_entries(
        self,
        sid: Optional[str] = None,
        access_mask: Optional[int] = None
    ) -> List[ACE]:
        """Find matching ACEs"""
        pass

@dataclass
class ACE:
    """Access Control Entry"""
    
    type: ACEType                   # ALLOW, DENY, AUDIT
    flags: ACEFlags                 # CONTAINER_INHERIT, etc.
    access_mask: int                # Permission bits
    sid: str                        # Security Identifier
    
    # For object ACEs
    object_type: Optional[str] = None
    inherited_object_type: Optional[str] = None

class ACEType(Enum):
    ACCESS_ALLOWED = 0x00
    ACCESS_DENIED = 0x01
    SYSTEM_AUDIT = 0x02
    ACCESS_ALLOWED_OBJECT = 0x05
    ACCESS_DENIED_OBJECT = 0x06
    SYSTEM_AUDIT_OBJECT = 0x07

class ACEFlags(IntFlag):
    OBJECT_INHERIT = 0x01
    CONTAINER_INHERIT = 0x02
    NO_PROPAGATE_INHERIT = 0x04
    INHERIT_ONLY = 0x08
    INHERITED = 0x10
    SUCCESSFUL_ACCESS = 0x40
    FAILED_ACCESS = 0x80
```

### 4.2 Access Mask Constants

```python
class FileAccessMask:
    """NTFS File/Directory Access Rights"""
    
    # Generic Rights
    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    GENERIC_EXECUTE = 0x20000000
    GENERIC_ALL = 0x10000000
    
    # Standard Rights
    DELETE = 0x00010000
    READ_CONTROL = 0x00020000
    WRITE_DAC = 0x00040000
    WRITE_OWNER = 0x00080000
    SYNCHRONIZE = 0x00100000
    
    STANDARD_RIGHTS_READ = READ_CONTROL
    STANDARD_RIGHTS_WRITE = READ_CONTROL
    STANDARD_RIGHTS_EXECUTE = READ_CONTROL
    STANDARD_RIGHTS_ALL = 0x001F0000
    
    # File/Directory Specific Rights
    FILE_READ_DATA = 0x0001           # FILE_LIST_DIRECTORY
    FILE_WRITE_DATA = 0x0002          # FILE_ADD_FILE
    FILE_APPEND_DATA = 0x0004         # FILE_ADD_SUBDIRECTORY
    FILE_READ_EA = 0x0008
    FILE_WRITE_EA = 0x0010
    FILE_EXECUTE = 0x0020             # FILE_TRAVERSE
    FILE_DELETE_CHILD = 0x0040
    FILE_READ_ATTRIBUTES = 0x0080
    FILE_WRITE_ATTRIBUTES = 0x0100
    
    FILE_ALL_ACCESS = STANDARD_RIGHTS_ALL | 0x1FF
    FILE_GENERIC_READ = (
        STANDARD_RIGHTS_READ |
        FILE_READ_DATA |
        FILE_READ_ATTRIBUTES |
        FILE_READ_EA |
        SYNCHRONIZE
    )
    FILE_GENERIC_WRITE = (
        STANDARD_RIGHTS_WRITE |
        FILE_WRITE_DATA |
        FILE_WRITE_ATTRIBUTES |
        FILE_WRITE_EA |
        FILE_APPEND_DATA |
        SYNCHRONIZE
    )
    FILE_GENERIC_EXECUTE = (
        STANDARD_RIGHTS_EXECUTE |
        FILE_READ_ATTRIBUTES |
        FILE_EXECUTE |
        SYNCHRONIZE
    )
```

### 4.3 Permission Manager

```python
class PermissionManager:
    """
    Windows File Permission & ACL Management
    """
    
    # Well-known SIDs
    WELL_KNOWN_SIDS = {
        "S-1-5-18": "SYSTEM",
        "S-1-5-19": "LOCAL SERVICE",
        "S-1-5-20": "NETWORK SERVICE",
        "S-1-5-32-544": "Administrators",
        "S-1-5-32-545": "Users",
        "S-1-5-32-546": "Guests",
        "S-1-1-0": "Everyone",
        "S-1-5-11": "Authenticated Users",
        "S-1-5-32-547": "Power Users",
        "S-1-5-32-548": "Account Operators",
        "S-1-5-32-549": "Server Operators",
        "S-1-5-32-550": "Print Operators",
        "S-1-5-32-551": "Backup Operators",
        "S-1-5-32-552": "Replicators",
    }
    
    def __init__(self):
        self._sid_cache: Dict[str, str] = {}
        
    def get_security_descriptor(self, path: str) -> SecurityDescriptor:
        """
        Get security descriptor for file/directory
        
        Args:
            path: File or directory path
            
        Returns:
            SecurityDescriptor object
        """
        pass
        
    def set_security_descriptor(
        self,
        path: str,
        sd: SecurityDescriptor
    ) -> bool:
        """
        Set security descriptor for file/directory
        
        Args:
            path: File or directory path
            sd: Security descriptor to apply
            
        Returns:
            True if successful
        """
        pass
        
    def get_owner(self, path: str) -> str:
        """Get owner SID of file/directory"""
        pass
        
    def set_owner(self, path: str, owner_sid: str) -> bool:
        """
        Set owner of file/directory
        Note: Requires SE_TAKE_OWNERSHIP or WRITE_OWNER privilege
        """
        pass
        
    def get_dacl(self, path: str) -> ACL:
        """Get discretionary ACL"""
        pass
        
    def set_dacl(
        self,
        path: str,
        acl: ACL,
        protected: bool = False
    ) -> bool:
        """
        Set discretionary ACL
        
        Args:
            path: Target path
            acl: New DACL
            protected: Prevent inheritance from parent
        """
        pass
        
    def add_permission(
        self,
        path: str,
        sid: str,
        access_mask: int,
        ace_type: ACEType = ACEType.ACCESS_ALLOWED,
        inherit: bool = True
    ) -> bool:
        """
        Add permission for user/group
        
        Args:
            path: Target path
            sid: User/group SID or name
            access_mask: Permission bits
            ace_type: ALLOW or DENY
            inherit: Apply to children
        """
        pass
        
    def remove_permission(
        self,
        path: str,
        sid: str,
        access_mask: Optional[int] = None
    ) -> bool:
        """Remove permission for user/group"""
        pass
        
    def check_access(
        self,
        path: str,
        desired_access: int,
        token: Optional[int] = None
    ) -> bool:
        """
        Check if current/effective token has access
        
        Args:
            path: Target path
            desired_access: Requested access rights
            token: Optional token handle (None = current process)
            
        Returns:
            True if access is granted
        """
        pass
        
    def name_to_sid(self, name: str) -> str:
        """
        Convert account name to SID string
        
        Args:
            name: Account name (DOMAIN\user or user)
            
        Returns:
            SID string
        """
        pass
        
    def sid_to_name(self, sid: str) -> str:
        """Convert SID to account name"""
        pass
        
    def get_effective_permissions(
        self,
        path: str,
        sid: str
    ) -> int:
        """
        Get effective permissions for user on path
        Considers group membership and inheritance
        
        Args:
            path: Target path
            sid: User SID
            
        Returns:
            Effective access mask
        """
        pass
        
    def take_ownership(self, path: str) -> bool:
        """
        Take ownership of file/directory
        Requires SE_TAKE_OWNERSHIP privilege
        """
        pass
        
    def reset_permissions(
        self,
        path: str,
        recursive: bool = False
    ) -> bool:
        """
        Reset permissions to inherited defaults
        
        Args:
            path: Target path
            recursive: Apply to all children
        """
        pass
        
    def copy_permissions(
        self,
        source: str,
        destination: str,
        recursive: bool = False
    ) -> bool:
        """Copy permissions from source to destination"""
        pass
```

### 4.4 Predefined Permission Sets

```python
class PermissionSets:
    """Common permission configurations"""
    
    # Full control for owner only
    PRIVATE = {
        "owner": FileAccessMask.FILE_ALL_ACCESS,
        "users": 0,
        "everyone": 0
    }
    
    # Read-only for everyone
    READ_ONLY = {
        "owner": FileAccessMask.FILE_ALL_ACCESS,
        "users": FileAccessMask.FILE_GENERIC_READ,
        "everyone": FileAccessMask.FILE_GENERIC_READ
    }
    
    # Read/write for users
    USER_READ_WRITE = {
        "owner": FileAccessMask.FILE_ALL_ACCESS,
        "users": FileAccessMask.FILE_GENERIC_READ | FileAccessMask.FILE_GENERIC_WRITE,
        "everyone": FileAccessMask.FILE_GENERIC_READ
    }
    
    # Executable
    EXECUTABLE = {
        "owner": FileAccessMask.FILE_ALL_ACCESS,
        "users": (
            FileAccessMask.FILE_GENERIC_READ |
            FileAccessMask.FILE_GENERIC_EXECUTE
        ),
        "everyone": (
            FileAccessMask.FILE_GENERIC_READ |
            FileAccessMask.FILE_GENERIC_EXECUTE
        )
    }
    
    # System protected
    SYSTEM_PROTECTED = {
        "SYSTEM": FileAccessMask.FILE_ALL_ACCESS,
        "Administrators": FileAccessMask.FILE_ALL_ACCESS,
        "users": 0,
        "everyone": 0
    }
```

---

## 5. Registry Hive Structure

### 5.1 Registry Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Windows Registry Structure                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  HKEY_LOCAL_MACHINE (HKLM)                               │   │
│  │  ├─ SAM (Security Accounts Manager)                      │   │
│  │  ├─ SECURITY (LSA secrets)                               │   │
│  │  ├─ SOFTWARE (Application settings)                      │   │
│  │  │   └─ OpenClawAgent                                    │   │
│  │  │       ├─ Config                                       │   │
│  │  │       ├─ Loops                                        │   │
│  │  │       ├─ Users                                        │   │
│  │  │       └─ System                                       │   │
│  │  ├─ SYSTEM (OS configuration)                            │   │
│  │  └─ HARDWARE (Hardware profiles)                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  HKEY_CURRENT_USER (HKCU)                                │   │
│  │  ├─ Software                                             │   │
│  │  │   └─ OpenClawAgent                                    │   │
│  │  │       ├─ Preferences                                  │   │
│  │  │       ├─ History                                      │   │
│  │  │       └─ Cache                                        │   │
│  │  ├─ Control Panel                                        │   │
│  │  ├─ Environment                                          │   │
│  │  └─ AppEvents                                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  HKEY_CLASSES_ROOT (HKCR)                                │   │
│  │  └─ (File associations and COM objects)                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  HKEY_USERS (HKU)                                        │   │
│  │  └─ (Loaded user profiles)                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  HKEY_CURRENT_CONFIG (HKCC)                              │   │
│  │  └─ (Current hardware profile)                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Hive File Locations

```python
class RegistryHiveLocations:
    """Physical locations of registry hive files"""
    
    # System hives
    SYSTEM_ROOT = r"%SystemRoot%\System32\config"
    
    SAM = r"%SystemRoot%\System32\config\SAM"
    SECURITY = r"%SystemRoot%\System32\config\SECURITY"
    SOFTWARE = r"%SystemRoot%\System32\config\SOFTWARE"
    SYSTEM = r"%SystemRoot%\System32\config\SYSTEM"
    DEFAULT = r"%SystemRoot%\System32\config\DEFAULT"
    
    # User hives
    NTUSER_DAT = r"%USERPROFILE%\NTUSER.DAT"
    USRCLASS_DAT = r"%USERPROFILE%\AppData\Local\Microsoft\Windows\UsrClass.dat"
    
    # Component hives
    COMPONENTS = r"%SystemRoot%\System32\config\COMPONENTS"
    BCD = r"\Boot\BCD"  # Boot Configuration Data
    
    # Backup locations
    REG_BACKUP = r"%SystemRoot%\System32\config\RegBack"
```

### 5.3 Agent Registry Structure

```python
AGENT_REGISTRY_STRUCTURE = {
    "HKLM\\SOFTWARE\\OpenClawAgent": {
        "description": "Machine-wide agent configuration",
        "permissions": "Administrators: Full, Users: Read",
        "keys": {
            "Config": {
                "Version": "REG_SZ",
                "InstallPath": "REG_EXPAND_SZ",
                "LogLevel": "REG_DWORD",
                "AutoStart": "REG_DWORD",
                "MaxMemory": "REG_QWORD",
            },
            "Security": {
                "RequireElevation": "REG_DWORD",
                "AllowedUsers": "REG_MULTI_SZ",
                "EncryptionKey": "REG_BINARY",
            },
            "Services": {
                "Gmail": {
                    "Enabled": "REG_DWORD",
                    "PollInterval": "REG_DWORD",
                },
                "Twilio": {
                    "Enabled": "REG_DWORD",
                    "VoiceEnabled": "REG_DWORD",
                    "SMSEnabled": "REG_DWORD",
                },
                "TTS": {
                    "Enabled": "REG_DWORD",
                    "Voice": "REG_SZ",
                    "Rate": "REG_DWORD",
                },
                "STT": {
                    "Enabled": "REG_DWORD",
                    "Language": "REG_SZ",
                },
            },
            "Loops": {
                f"Loop{i:02d}": {
                    "Enabled": "REG_DWORD",
                    "Interval": "REG_DWORD",
                    "Priority": "REG_DWORD",
                }
                for i in range(1, 16)
            },
        },
    },
    "HKCU\\Software\\OpenClawAgent": {
        "description": "User-specific agent settings",
        "permissions": "Current User: Full",
        "keys": {
            "Preferences": {
                "Theme": "REG_SZ",
                "Notifications": "REG_DWORD",
                "SoundEnabled": "REG_DWORD",
            },
            "Identity": {
                "AgentName": "REG_SZ",
                "Personality": "REG_SZ",
                "Voice": "REG_SZ",
            },
            "History": {
                "Commands": "REG_BINARY",  # Encrypted
                "Conversations": "REG_BINARY",  # Encrypted
            },
            "Cache": {
                "LastSync": "REG_QWORD",
                "DataVersion": "REG_DWORD",
            },
        },
    },
    "HKLM\\SYSTEM\\CurrentControlSet\\Services\\OpenClawAgent": {
        "description": "Windows service configuration",
        "keys": {
            "Type": "REG_DWORD",
            "Start": "REG_DWORD",
            "ErrorControl": "REG_DWORD",
            "ImagePath": "REG_EXPAND_SZ",
            "DisplayName": "REG_SZ",
            "Description": "REG_SZ",
            "ObjectName": "REG_SZ",
            "DependOnService": "REG_MULTI_SZ",
        },
    },
}
```

---

## 6. Registry Operations

### 6.1 Registry Value Types

```python
class RegistryValueType(Enum):
    """Windows Registry Value Types"""
    
    REG_NONE = 0                    # No value type
    REG_SZ = 1                      # Unicode string
    REG_EXPAND_SZ = 2               # Expandable string (%VAR%)
    REG_BINARY = 3                  # Binary data
    REG_DWORD = 4                   # 32-bit number
    REG_DWORD_BIG_ENDIAN = 5        # 32-bit big-endian
    REG_LINK = 6                    # Symbolic link
    REG_MULTI_SZ = 7                # Multiple strings
    REG_RESOURCE_LIST = 8           # Resource list
    REG_FULL_RESOURCE_DESCRIPTOR = 9
    REG_RESOURCE_REQUIREMENTS_LIST = 10
    REG_QWORD = 11                  # 64-bit number

class RegistryAccessRights(IntFlag):
    """Registry Key Access Rights"""
    
    # Standard Rights
    DELETE = 0x00010000
    READ_CONTROL = 0x00020000
    WRITE_DAC = 0x00040000
    WRITE_OWNER = 0x00080000
    SYNCHRONIZE = 0x00100000
    
    # Key Specific Rights
    KEY_QUERY_VALUE = 0x0001
    KEY_SET_VALUE = 0x0002
    KEY_CREATE_SUB_KEY = 0x0004
    KEY_ENUMERATE_SUB_KEYS = 0x0008
    KEY_NOTIFY = 0x0010
    KEY_CREATE_LINK = 0x0020
    
    # Combined Rights
    KEY_READ = (
        STANDARD_RIGHTS_READ |
        KEY_QUERY_VALUE |
        KEY_ENUMERATE_SUB_KEYS |
        KEY_NOTIFY
    ) & ~SYNCHRONIZE
    
    KEY_WRITE = (
        STANDARD_RIGHTS_WRITE |
        KEY_SET_VALUE |
        KEY_CREATE_SUB_KEY
    ) & ~SYNCHRONIZE
    
    KEY_EXECUTE = KEY_READ
    
    KEY_ALL_ACCESS = (
        STANDARD_RIGHTS_ALL |
        KEY_QUERY_VALUE |
        KEY_SET_VALUE |
        KEY_CREATE_SUB_KEY |
        KEY_ENUMERATE_SUB_KEYS |
        KEY_NOTIFY |
        KEY_CREATE_LINK
    ) & ~SYNCHRONIZE
```

### 6.2 Registry Manager

```python
class RegistryManager:
    """
    Windows Registry Access Manager
    Provides safe, transactional registry operations
    """
    
    # Predefined key handles
    HKEY_CLASSES_ROOT = 0x80000000
    HKEY_CURRENT_USER = 0x80000001
    HKEY_LOCAL_MACHINE = 0x80000002
    HKEY_USERS = 0x80000003
    HKEY_PERFORMANCE_DATA = 0x80000004
    HKEY_CURRENT_CONFIG = 0x80000005
    
    # Key creation options
    REG_OPTION_RESERVED = 0x0000
    REG_OPTION_NON_VOLATILE = 0x0000
    REG_OPTION_VOLATILE = 0x0001
    REG_OPTION_CREATE_LINK = 0x0002
    REG_OPTION_BACKUP_RESTORE = 0x0004
    
    # Disposition values
    REG_CREATED_NEW_KEY = 1
    REG_OPENED_EXISTING_KEY = 2
    
    def __init__(self, transaction_manager: Optional[TransactionManager] = None):
        self.transaction_manager = transaction_manager
        self._key_cache: Dict[str, int] = {}
        self.logger = logging.getLogger("RegistryManager")
        
    def open_key(
        self,
        root: Union[int, str],
        subkey: str,
        access: int = RegistryAccessRights.KEY_READ,
        create: bool = False
    ) -> int:
        """
        Open or create registry key
        
        Args:
            root: Root key handle or name (HKLM, HKCU, etc.)
            subkey: Subkey path
            access: Access rights
            create: Create if doesn't exist
            
        Returns:
            Key handle
        """
        pass
        
    def close_key(self, key_handle: int) -> None:
        """Close registry key handle"""
        pass
        
    def read_value(
        self,
        root: Union[int, str],
        subkey: str,
        value_name: str,
        default: Any = None
    ) -> Tuple[Any, RegistryValueType]:
        """
        Read registry value
        
        Args:
            root: Root key or handle
            subkey: Subkey path
            value_name: Value name (empty for default)
            default: Default if not found
            
        Returns:
            (value, type) tuple
        """
        pass
        
    def write_value(
        self,
        root: Union[int, str],
        subkey: str,
        value_name: str,
        value: Any,
        value_type: RegistryValueType = RegistryValueType.REG_SZ
    ) -> bool:
        """
        Write registry value
        
        Args:
            root: Root key or handle
            subkey: Subkey path
            value_name: Value name
            value: Value to write
            value_type: Registry value type
            
        Returns:
            True if successful
        """
        pass
        
    def delete_value(
        self,
        root: Union[int, str],
        subkey: str,
        value_name: str
    ) -> bool:
        """Delete registry value"""
        pass
        
    def delete_key(
        self,
        root: Union[int, str],
        subkey: str,
        recursive: bool = False
    ) -> bool:
        """
        Delete registry key
        
        Args:
            root: Root key or handle
            subkey: Subkey path
            recursive: Delete subkeys recursively
        """
        pass
        
    def enum_keys(
        self,
        root: Union[int, str],
        subkey: str = ""
    ) -> Iterator[str]:
        """Enumerate subkey names"""
        pass
        
    def enum_values(
        self,
        root: Union[int, str],
        subkey: str = ""
    ) -> Iterator[Tuple[str, Any, RegistryValueType]]:
        """Enumerate value names and data"""
        pass
        
    def key_exists(self, root: Union[int, str], subkey: str) -> bool:
        """Check if registry key exists"""
        pass
        
    def value_exists(
        self,
        root: Union[int, str],
        subkey: str,
        value_name: str
    ) -> bool:
        """Check if registry value exists"""
        pass
        
    def copy_key_tree(
        self,
        source_root: Union[int, str],
        source_subkey: str,
        dest_root: Union[int, str],
        dest_subkey: str
    ) -> bool:
        """Copy entire key tree"""
        pass
        
    def get_key_info(
        self,
        root: Union[int, str],
        subkey: str
    ) -> KeyInfo:
        """Get key information"""
        pass
        
    def save_key(
        self,
        root: Union[int, str],
        subkey: str,
        file_path: str
    ) -> bool:
        """Save key to file (requires backup privilege)"""
        pass
        
    def load_key(
        self,
        root: Union[int, str],
        subkey: str,
        file_path: str
    ) -> bool:
        """Load key from file"""
        pass
        
    def connect_registry(self, machine_name: str) -> int:
        """
        Connect to remote registry
        
        Args:
            machine_name: \\machine_name or IP
            
        Returns:
            Root key handle for remote registry
        """
        pass
        
    def disconnect_registry(self, root_key: int) -> None:
        """Disconnect from remote registry"""
        pass
        
    def watch_key(
        self,
        root: Union[int, str],
        subkey: str,
        callback: Callable,
        filter: int = REG_NOTIFY_CHANGE_LAST_SET
    ) -> WatchHandle:
        """
        Watch registry key for changes
        
        Args:
            root: Root key or handle
            subkey: Subkey path
            callback: Function to call on change
            filter: Change notification filter
            
        Returns:
            Watch handle for unwatching
        """
        pass
        
    def unwatch_key(self, handle: WatchHandle) -> None:
        """Stop watching registry key"""
        pass

@dataclass
class KeyInfo:
    """Registry key information"""
    subkey_count: int
    max_subkey_len: int
    value_count: int
    max_value_name_len: int
    max_value_len: int
    last_write_time: datetime
    class_name: str
```

### 6.3 Registry Helper Methods

```python
class RegistryHelper:
    """Helper methods for common registry operations"""
    
    @staticmethod
    def parse_path(full_path: str) -> Tuple[int, str]:
        """
        Parse full registry path into root and subkey
        
        Args:
            full_path: Full path like "HKLM\SOFTWARE\MyApp"
            
        Returns:
            (root_handle, subkey_path)
        """
        pass
        
    @staticmethod
    def expand_string(value: str) -> str:
        """Expand environment variables in registry string"""
        pass
        
    @staticmethod
    def encode_multi_string(values: List[str]) -> bytes:
        """Encode list of strings for REG_MULTI_SZ"""
        pass
        
    @staticmethod
    def decode_multi_string(data: bytes) -> List[str]:
        """Decode REG_MULTI_SZ to list of strings"""
        pass
        
    @staticmethod
    def dword_to_bytes(value: int) -> bytes:
        """Convert DWORD to little-endian bytes"""
        pass
        
    @staticmethod
    def qword_to_bytes(value: int) -> bytes:
        """Convert QWORD to little-endian bytes"""
        pass
```

---

## 7. File Watching & Notifications

### 7.1 File Watcher Architecture

```python
class FileWatcherManager:
    """
    Windows File System Change Notification Manager
    Uses ReadDirectoryChangesW API
    """
    
    # Notification filters
    FILE_NOTIFY_CHANGE_FILE_NAME = 0x00000001    # Create, delete, rename
    FILE_NOTIFY_CHANGE_DIR_NAME = 0x00000002     # Directory create, delete
    FILE_NOTIFY_CHANGE_ATTRIBUTES = 0x00000004   # Attribute changes
    FILE_NOTIFY_CHANGE_SIZE = 0x00000008         # File size changes
    FILE_NOTIFY_CHANGE_LAST_WRITE = 0x00000010   # Last write time
    FILE_NOTIFY_CHANGE_LAST_ACCESS = 0x00000020  # Last access time
    FILE_NOTIFY_CHANGE_CREATION = 0x00000040     # Creation time
    FILE_NOTIFY_CHANGE_SECURITY = 0x00000100     # Security descriptor
    
    # Action types
    FILE_ACTION_ADDED = 0x00000001
    FILE_ACTION_REMOVED = 0x00000002
    FILE_ACTION_MODIFIED = 0x00000003
    FILE_ACTION_RENAMED_OLD_NAME = 0x00000004
    FILE_ACTION_RENAMED_NEW_NAME = 0x00000005
    
    def __init__(self):
        self._watchers: Dict[str, FileWatcher] = {}
        self._callback_queue: Queue = Queue()
        self._executor = ThreadPoolExecutor(max_workers=10)
        self.logger = logging.getLogger("FileWatcherManager")
        
    def add_watch(
        self,
        path: str,
        callback: Callable[[FileChangeEvent], None],
        recursive: bool = True,
        filter: int = FILE_NOTIFY_CHANGE_LAST_WRITE | FILE_NOTIFY_CHANGE_FILE_NAME,
        include_subdirectories: bool = True
    ) -> WatchHandle:
        """
        Add file system watch
        
        Args:
            path: Directory to watch
            callback: Function to call on changes
            recursive: Watch subdirectories
            filter: Change notification filter
            include_subdirectories: Include subdirectories
            
        Returns:
            Watch handle
        """
        pass
        
    def remove_watch(self, handle: WatchHandle) -> bool:
        """Remove file system watch"""
        pass
        
    def pause_watch(self, handle: WatchHandle) -> bool:
        """Pause watching (without removing)"""
        pass
        
    def resume_watch(self, handle: WatchHandle) -> bool:
        """Resume paused watch"""
        pass
        
    def get_watched_paths(self) -> List[str]:
        """Get list of watched paths"""
        pass

@dataclass
class FileChangeEvent:
    """File system change event"""
    action: FileAction
    path: str
    old_path: Optional[str]  # For rename events
    timestamp: datetime
    watcher_handle: WatchHandle

class FileAction(Enum):
    CREATED = "created"
    DELETED = "deleted"
    MODIFIED = "modified"
    RENAMED = "renamed"
    ATTRIBUTES_CHANGED = "attributes_changed"
    SIZE_CHANGED = "size_changed"
    SECURITY_CHANGED = "security_changed"

class FileWatcher:
    """Individual file watcher instance"""
    
    def __init__(
        self,
        path: str,
        callback: Callable,
        filter: int,
        recursive: bool
    ):
        self.path = path
        self.callback = callback
        self.filter = filter
        self.recursive = recursive
        self._handle: Optional[int] = None
        self._overlapped: Optional[PyOVERLAPPED] = None
        self._buffer: bytes = b""
        self._running = False
        self._thread: Optional[Thread] = None
        
    def start(self) -> None:
        """Start watching"""
        pass
        
    def stop(self) -> None:
        """Stop watching"""
        pass
        
    def _watch_loop(self) -> None:
        """Main watch loop using ReadDirectoryChangesW"""
        pass
        
    def _process_notification(self, buffer: bytes) -> None:
        """Process notification buffer"""
        pass
```

### 7.2 FindFirstChangeNotification Alternative

```python
class LegacyFileWatcher:
    """
    Legacy file watcher using FindFirstChangeNotification
    Simpler but less efficient than ReadDirectoryChangesW
    """
    
    def __init__(self):
        self._notifications: Dict[str, int] = {}
        
    def create_notification(
        self,
        path: str,
        watch_subtree: bool = True,
        filter: int = FileWatcherManager.FILE_NOTIFY_CHANGE_LAST_WRITE
    ) -> int:
        """
        Create change notification handle
        
        Args:
            path: Directory to watch
            watch_subtree: Watch subdirectories
            filter: Notification filter
            
        Returns:
            Notification handle
        """
        pass
        
    def wait_for_notification(
        self,
        handle: int,
        timeout_ms: int = INFINITE
    ) -> bool:
        """
        Wait for notification (blocking)
        
        Args:
            handle: Notification handle
            timeout_ms: Timeout in milliseconds
            
        Returns:
            True if notification received, False if timeout
        """
        pass
        
    def find_next_notification(self, handle: int) -> bool:
        """Request next notification"""
        pass
        
    def close_notification(self, handle: int) -> None:
        """Close notification handle"""
        pass
```

---

## 8. Temporary File Management

### 8.1 Temp File Manager

```python
class TempFileManager:
    """
    Windows Temporary File Management
    Handles creation, tracking, and cleanup of temporary files
    """
    
    # Default temp directories
    SYSTEM_TEMP = os.environ.get("TEMP", r"C:\Windows\Temp")
    USER_TEMP = os.environ.get("LOCALAPPDATA", "") + r"\Temp"
    
    def __init__(
        self,
        base_path: Optional[str] = None,
        prefix: str = "ocl_",
        suffix: str = ".tmp",
        cleanup_on_exit: bool = True
    ):
        self.base_path = base_path or self.USER_TEMP
        self.prefix = prefix
        self.suffix = suffix
        self.cleanup_on_exit = cleanup_on_exit
        self._tracked_files: Set[str] = set()
        self._tracked_dirs: Set[str] = set()
        self._lock = threading.Lock()
        self.logger = logging.getLogger("TempFileManager")
        
        # Register cleanup handler
        if cleanup_on_exit:
            atexit.register(self._cleanup_all)
            
    def create_temp_file(
        self,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        dir: Optional[str] = None,
        text: bool = True,
        delete_on_close: bool = False
    ) -> Tuple[int, str]:
        """
        Create temporary file
        
        Args:
            prefix: Filename prefix
            suffix: Filename suffix
            dir: Directory (default: base_path)
            text: Open in text mode
            delete_on_close: Delete when handle closed
            
        Returns:
            (file_handle, file_path) tuple
        """
        pass
        
    def create_temp_directory(
        self,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        dir: Optional[str] = None
    ) -> str:
        """
        Create temporary directory
        
        Args:
            prefix: Directory name prefix
            suffix: Directory name suffix
            dir: Parent directory
            
        Returns:
            Directory path
        """
        pass
        
    def create_named_temp_file(
        self,
        name: Optional[str] = None,
        dir: Optional[str] = None,
        content: Optional[bytes] = None
    ) -> str:
        """
        Create named temporary file with optional content
        
        Args:
            name: Specific filename (random if None)
            dir: Directory
            content: Initial content
            
        Returns:
            File path
        """
        pass
        
    def track_file(self, path: str) -> None:
        """Add file to tracking for cleanup"""
        with self._lock:
            self._tracked_files.add(path)
            
    def untrack_file(self, path: str) -> None:
        """Remove file from tracking"""
        with self._lock:
            self._tracked_files.discard(path)
            
    def delete_temp_file(self, path: str) -> bool:
        """Delete tracked temp file"""
        pass
        
    def delete_temp_directory(self, path: str, recursive: bool = True) -> bool:
        """Delete tracked temp directory"""
        pass
        
    def cleanup_all(self) -> Tuple[int, int]:
        """
        Clean up all tracked temp files and directories
        
        Returns:
            (files_deleted, dirs_deleted) tuple
        """
        pass
        
    def _cleanup_all(self) -> None:
        """Internal cleanup handler for atexit"""
        pass
        
    def get_temp_path(self) -> str:
        """Get system temp path"""
        pass
        
    def get_temp_file_name(
        self,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None
    ) -> str:
        """Generate unique temp filename (without creating)"""
        pass
        
    def get_disk_free_space(self, path: Optional[str] = None) -> int:
        """Get free space on temp drive"""
        pass
```

### 8.2 Secure Temp File Operations

```python
class SecureTempFileManager(TempFileManager):
    """
    Secure temporary file management with encryption
    """
    
    def __init__(
        self,
        encryption_key: Optional[bytes] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encryption_key = encryption_key or self._generate_key()
        
    def create_secure_temp_file(
        self,
        content: Optional[bytes] = None,
        **kwargs
    ) -> Tuple[int, str]:
        """
        Create encrypted temporary file
        
        Args:
            content: Optional initial content (will be encrypted)
            **kwargs: Additional options
            
        Returns:
            (handle, path) tuple
        """
        pass
        
    def read_secure_file(self, path: str) -> bytes:
        """Read and decrypt secure temp file"""
        pass
        
    def write_secure_file(self, path: str, content: bytes) -> None:
        """Encrypt and write to secure temp file"""
        pass
        
    def secure_delete(self, path: str, passes: int = 3) -> bool:
        """
        Securely delete file with overwriting
        
        Args:
            path: File to delete
            passes: Number of overwrite passes
            
        Returns:
            True if successful
        """
        pass
        
    def _generate_key(self) -> bytes:
        """Generate encryption key"""
        pass
```

---

## 9. Recycle Bin Operations

### 9.1 Recycle Bin Manager

```python
class RecycleBinManager:
    """
    Windows Recycle Bin Operations
    Uses SHFileOperation or IFileOperation
    """
    
    # SHFileOperation flags
    FOF_SILENT = 0x0004
    FOF_RENAMEONCOLLISION = 0x0008
    FOF_NOCONFIRMATION = 0x0010
    FOF_ALLOWUNDO = 0x0040
    FOF_FILESONLY = 0x0080
    FOF_SIMPLEPROGRESS = 0x0100
    FOF_NOCONFIRMMKDIR = 0x0200
    FOF_NOERRORUI = 0x0400
    FOF_NOCOPYSECURITYATTRIBS = 0x0800
    FOF_NORECURSION = 0x1000
    FOF_NO_CONNECTED_ELEMENTS = 0x2000
    FOF_WANTNUKEWARNING = 0x4000
    
    # FO_FUNC constants
    FO_MOVE = 0x0001
    FO_COPY = 0x0002
    FO_DELETE = 0x0003
    FO_RENAME = 0x0004
    
    def __init__(self):
        self.logger = logging.getLogger("RecycleBinManager")
        
    def delete_to_recycle_bin(
        self,
        path: str,
        silent: bool = True,
        no_confirmation: bool = True,
        show_progress: bool = False
    ) -> bool:
        """
        Move file/directory to recycle bin
        
        Args:
            path: File or directory to delete
            silent: No UI except progress
            no_confirmation: Don't ask for confirmation
            show_progress: Show progress dialog
            
        Returns:
            True if successful
        """
        pass
        
    def delete_multiple_to_recycle_bin(
        self,
        paths: List[str],
        **kwargs
    ) -> Tuple[bool, List[str]]:
        """
        Move multiple items to recycle bin
        
        Args:
            paths: List of paths to delete
            **kwargs: Options for delete_to_recycle_bin
            
        Returns:
            (success, failed_paths) tuple
        """
        pass
        
    def empty_recycle_bin(
        self,
        drive: Optional[str] = None,
        no_confirmation: bool = True,
        no_progress: bool = True,
        no_sound: bool = True
    ) -> bool:
        """
        Empty recycle bin
        
        Args:
            drive: Specific drive (None = all)
            no_confirmation: Skip confirmation
            no_progress: Don't show progress
            no_sound: Don't play sound
            
        Returns:
            True if successful
        """
        pass
        
    def get_recycle_bin_info(
        self,
        drive: Optional[str] = None
    ) -> RecycleBinInfo:
        """
        Get recycle bin information
        
        Args:
            drive: Drive letter (None = all)
            
        Returns:
            RecycleBinInfo object
        """
        pass
        
    def enum_recycle_bin_items(
        self,
        drive: Optional[str] = None
    ) -> Iterator[RecycleBinItem]:
        """
        Enumerate items in recycle bin
        
        Args:
            drive: Drive letter (None = all)
            
        Yields:
            RecycleBinItem objects
        """
        pass
        
    def restore_item(
        self,
        item_id: str,
        destination: Optional[str] = None
    ) -> bool:
        """
        Restore item from recycle bin
        
        Args:
            item_id: Item identifier
            destination: Restore to different location
            
        Returns:
            True if successful
        """
        pass
        
    def delete_from_recycle_bin(
        self,
        item_id: str,
        permanent: bool = True
    ) -> bool:
        """
        Permanently delete item from recycle bin
        
        Args:
            item_id: Item identifier
            permanent: Actually delete (not just mark)
            
        Returns:
            True if successful
        """
        pass

@dataclass
class RecycleBinInfo:
    """Recycle bin information"""
    drive: str
    total_size: int
    item_count: int
    max_size: int
    percent_full: float

@dataclass
class RecycleBinItem:
    """Recycle bin item"""
    id: str
    original_path: str
    deleted_time: datetime
    size: int
    drive: str
    attributes: int
```

---

## 10. Security & Sandboxing

### 10.1 Security Context

```python
class SecurityContext:
    """
    Windows Security Context Manager
    Handles impersonation, privileges, and access tokens
    """
    
    # Privilege constants
    SE_CREATE_TOKEN = "SeCreateTokenPrivilege"
    SE_ASSIGNPRIMARYTOKEN = "SeAssignPrimaryTokenPrivilege"
    SE_LOCK_MEMORY = "SeLockMemoryPrivilege"
    SE_INCREASE_QUOTA = "SeIncreaseQuotaPrivilege"
    SE_MACHINE_ACCOUNT = "SeMachineAccountPrivilege"
    SE_TCB = "SeTcbPrivilege"
    SE_SECURITY = "SeSecurityPrivilege"
    SE_TAKE_OWNERSHIP = "SeTakeOwnershipPrivilege"
    SE_LOAD_DRIVER = "SeLoadDriverPrivilege"
    SE_SYSTEM_PROFILE = "SeSystemProfilePrivilege"
    SE_SYSTEMTIME = "SeSystemtimePrivilege"
    SE_PROF_SINGLE_PROCESS = "SeProfileSingleProcessPrivilege"
    SE_INC_BASE_PRIORITY = "SeIncreaseBasePriorityPrivilege"
    SE_CREATE_PAGEFILE = "SeCreatePagefilePrivilege"
    SE_CREATE_PERMANENT = "SeCreatePermanentPrivilege"
    SE_BACKUP = "SeBackupPrivilege"
    SE_RESTORE = "SeRestorePrivilege"
    SE_SHUTDOWN = "SeShutdownPrivilege"
    SE_DEBUG = "SeDebugPrivilege"
    SE_AUDIT = "SeAuditPrivilege"
    SE_SYSTEM_ENVIRONMENT = "SeSystemEnvironmentPrivilege"
    SE_CHANGE_NOTIFY = "SeChangeNotifyPrivilege"
    SE_REMOTE_SHUTDOWN = "SeRemoteShutdownPrivilege"
    SE_UNDOCK = "SeUndockPrivilege"
    SE_SYNC_AGENT = "SeSyncAgentPrivilege"
    SE_ENABLE_DELEGATION = "SeEnableDelegationPrivilege"
    SE_MANAGE_VOLUME = "SeManageVolumePrivilege"
    SE_IMPERSONATE = "SeImpersonatePrivilege"
    SE_CREATE_GLOBAL = "SeCreateGlobalPrivilege"
    SE_TRUSTED_CREDMAN_ACCESS = "SeTrustedCredManAccessPrivilege"
    SE_RELABEL = "SeRelabelPrivilege"
    SE_INC_WORKING_SET = "SeIncreaseWorkingSetPrivilege"
    SE_TIME_ZONE = "SeTimeZonePrivilege"
    SE_CREATE_SYMBOLIC_LINK = "SeCreateSymbolicLinkPrivilege"
    
    def __init__(self):
        self._token: Optional[int] = None
        self._original_token: Optional[int] = None
        self._impersonating = False
        
    def enable_privilege(self, privilege: str) -> bool:
        """
        Enable privilege for current process
        
        Args:
            privilege: Privilege name (e.g., SE_BACKUP)
            
        Returns:
            True if enabled
        """
        pass
        
    def disable_privilege(self, privilege: str) -> bool:
        """Disable privilege"""
        pass
        
    def check_privilege(self, privilege: str) -> bool:
        """Check if privilege is enabled"""
        pass
        
    def impersonate_user(
        self,
        username: str,
        domain: str = "",
        password: str = ""
    ) -> bool:
        """
        Impersonate user
        
        Args:
            username: Username
            domain: Domain name
            password: User password
            
        Returns:
            True if successful
        """
        pass
        
    def impersonate_logged_on_user(self, token: int) -> bool:
        """Impersonate using existing token"""
        pass
        
    def revert_to_self(self) -> bool:
        """End impersonation"""
        pass
        
    def create_restricted_token(
        self,
        base_token: int,
        flags: int,
        delete_privileges: List[str],
        delete_sids: List[str],
        restrict_sids: List[str]
    ) -> int:
        """
        Create restricted access token
        
        Args:
            base_token: Base token to restrict
            flags: Restriction flags
            delete_privileges: Privileges to remove
            delete_sids: SIDs to disable
            restrict_sids: SIDs to restrict to
            
        Returns:
            Restricted token handle
        """
        pass
        
    def get_process_token(self, process_id: int = 0) -> int:
        """
        Get process access token
        
        Args:
            process_id: Process ID (0 = current)
            
        Returns:
            Token handle
        """
        pass
        
    def get_token_info(self, token: int) -> TokenInfo:
        """Get token information"""
        pass

@dataclass
class TokenInfo:
    """Access token information"""
    user: str
    groups: List[str]
    privileges: List[Tuple[str, int]]  # (name, attributes)
    type: str  # Primary or Impersonation
    impersonation_level: Optional[str]
    session_id: int
    elevation_type: Optional[str]
    is_elevated: bool
```

### 10.2 Sandboxing

```python
class SandboxManager:
    """
    Windows Job Object & Sandbox Manager
    Creates isolated execution environments
    """
    
    def __init__(self):
        self._jobs: Dict[str, int] = {}
        
    def create_job_object(self, name: str) -> int:
        """
        Create job object for resource control
        
        Args:
            name: Job object name
            
        Returns:
            Job handle
        """
        pass
        
    def set_job_limits(
        self,
        job: int,
        limits: JobLimits
    ) -> bool:
        """
        Set job object limits
        
        Args:
            job: Job handle
            limits: Limit configuration
            
        Returns:
            True if successful
        """
        pass
        
    def assign_process_to_job(self, job: int, process: int) -> bool:
        """Assign process to job object"""
        pass
        
    def terminate_job(self, job: int, exit_code: int = 1) -> bool:
        """Terminate all processes in job"""
        pass

@dataclass
class JobLimits:
    """Job object limits"""
    max_working_set: Optional[Tuple[int, int]] = None
    max_job_time: Optional[int] = None  # 100-nanosecond units
    max_process_count: Optional[int] = None
    max_processor_time: Optional[int] = None
    min_max_processor_rate: Optional[Tuple[int, int]] = None
    max_committed_memory: Optional[int] = None
    kill_on_job_close: bool = True
    breakaway_ok: bool = False
    silent_breakaway_ok: bool = False
    die_on_unhandled_exception: bool = True
```

---

## 11. Error Handling

### 11.1 Error Codes & Exceptions

```python
class WindowsError(Exception):
    """Base Windows error exception"""
    
    def __init__(
        self,
        message: str,
        error_code: int = 0,
        function: str = ""
    ):
        super().__init__(message)
        self.error_code = error_code
        self.function = function
        
    @property
    def error_name(self) -> str:
        """Get error code name"""
        return ERROR_CODES.get(self.error_code, "UNKNOWN")

class FileSystemError(WindowsError):
    """File system operation error"""
    pass

class RegistryError(WindowsError):
    """Registry operation error"""
    pass

class PermissionError(WindowsError):
    """Permission/ACL error"""
    pass

class SecurityError(WindowsError):
    """Security context error"""
    pass

# Common Windows Error Codes
ERROR_CODES = {
    0: "ERROR_SUCCESS",
    1: "ERROR_INVALID_FUNCTION",
    2: "ERROR_FILE_NOT_FOUND",
    3: "ERROR_PATH_NOT_FOUND",
    5: "ERROR_ACCESS_DENIED",
    6: "ERROR_INVALID_HANDLE",
    13: "ERROR_INVALID_DATA",
    18: "ERROR_NO_MORE_FILES",
    32: "ERROR_SHARING_VIOLATION",
    87: "ERROR_INVALID_PARAMETER",
    122: "ERROR_INSUFFICIENT_BUFFER",
    123: "ERROR_INVALID_NAME",
    1314: "ERROR_PRIVILEGE_NOT_HELD",
    161: "ERROR_BAD_PATHNAME",
    183: "ERROR_ALREADY_EXISTS",
    259: "ERROR_NO_MORE_ITEMS",
    995: "ERROR_OPERATION_ABORTED",
    120: "ERROR_CALL_NOT_IMPLEMENTED",
    997: "ERROR_IO_PENDING",
    1223: "ERROR_CANCELLED",
    1225: "ERROR_CONNECTION_REFUSED",
    1235: "ERROR_REQUEST_ABORTED",
    1450: "ERROR_NO_SYSTEM_RESOURCES",
    160: "ERROR_BAD_ARGUMENTS",
    317: "ERROR_MR_MID_NOT_FOUND",
    1008: "ERROR_NO_TOKEN",
    1018: "ERROR_KEY_DELETED",
    1019: "ERROR_NO_LOG_SPACE",
    1020: "ERROR_KEY_HAS_CHILDREN",
    1021: "ERROR_CHILD_MUST_BE_VOLATILE",
    1022: "ERROR_NOTIFY_ENUM_DIR",
    1056: "ERROR_SERVICE_ALREADY_RUNNING",
    1060: "ERROR_SERVICE_DOES_NOT_EXIST",
    1062: "ERROR_SERVICE_NOT_ACTIVE",
    1066: "ERROR_SERVICE_SPECIFIC_ERROR",
    1069: "ERROR_SERVICE_LOGON_FAILED",
    1072: "ERROR_SERVICE_MARKED_FOR_DELETE",
    1073: "ERROR_SERVICE_EXISTS",
    1077: "ERROR_SERVICE_NEVER_STARTED",
}

# NT Status Codes (for ntdll functions)
NT_STATUS_CODES = {
    0x00000000: "STATUS_SUCCESS",
    0xC0000001: "STATUS_UNSUCCESSFUL",
    0xC0000005: "STATUS_ACCESS_VIOLATION",
    0xC0000008: "STATUS_INVALID_HANDLE",
    0xC000000D: "STATUS_INVALID_PARAMETER",
    0xC0000017: "STATUS_NO_MEMORY",
    0xC0000022: "STATUS_ACCESS_DENIED",
    0xC0000033: "STATUS_OBJECT_NAME_INVALID",
    0xC0000034: "STATUS_OBJECT_NAME_NOT_FOUND",
    0xC0000035: "STATUS_OBJECT_NAME_COLLISION",
    0xC000003A: "STATUS_OBJECT_PATH_NOT_FOUND",
    0xC00000BA: "STATUS_NOT_A_DIRECTORY",
    0xC0000103: "STATUS_NOT_A_REPARSE_POINT",
    0xC000010B: "STATUS_DIRECTORY_NOT_EMPTY",
    0xC0000120: "STATUS_CANCELLED",
    0xC0000121: "STATUS_CANNOT_DELETE",
    0xC0000201: "STATUS_BUFFER_OVERFLOW",
    0xC0000225: "STATUS_NOT_FOUND",
    0xC0000234: "STATUS_DELETE_PENDING",
}
```

### 11.2 Error Handler

```python
class ErrorHandler:
    """Centralized error handling"""
    
    def __init__(self):
        self._handlers: Dict[int, List[Callable]] = {}
        self._fallback_handler: Optional[Callable] = None
        
    def register_handler(
        self,
        error_code: int,
        handler: Callable
    ) -> None:
        """Register handler for specific error code"""
        if error_code not in self._handlers:
            self._handlers[error_code] = []
        self._handlers[error_code].append(handler)
        
    def handle_error(self, error: WindowsError) -> bool:
        """
        Handle error with registered handlers
        
        Args:
            error: Error to handle
            
        Returns:
            True if handled
        """
        handlers = self._handlers.get(error.error_code, [])
        for handler in handlers:
            try:
                if handler(error):
                    return True
            except Exception as e:
                logging.error(f"Error handler failed: {e}")
                
        if self._fallback_handler:
            return self._fallback_handler(error)
            
        return False
        
    def get_error_message(self, error_code: int) -> str:
        """Get formatted error message"""
        pass
        
    def log_error(self, error: WindowsError, level: int = logging.ERROR) -> None:
        """Log error with context"""
        pass
```

---

## 12. Implementation Examples

### 12.1 Complete File Operations Example

```python
# Example: Agent Configuration File Management

from pathlib import Path
import json

class AgentConfigManager:
    """
    Manages agent configuration files
    Demonstrates full file system operations
    """
    
    def __init__(self, agent_context: AgentContext):
        self.context = agent_context
        self.file_manager = NTFSFileManager(agent_context)
        self.path_manager = WindowsPathManager()
        self.permission_manager = PermissionManager()
        self.temp_manager = TempFileManager()
        
        # Resolve agent paths
        self.config_dir = self.path_manager.resolve_path(
            "{localappdata}\\OpenClawAgent\\config"
        )
        self.ensure_config_directory()
        
    def ensure_config_directory(self) -> None:
        """Create and secure config directory"""
        # Create directory if needed
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir, exist_ok=True)
            
        # Set secure permissions
        self.permission_manager.add_permission(
            self.config_dir,
            "Administrators",
            FileAccessMask.FILE_ALL_ACCESS
        )
        self.permission_manager.add_permission(
            self.config_dir,
            "Users",
            FileAccessMask.FILE_GENERIC_READ | FileAccessMask.FILE_GENERIC_WRITE
        )
        
    def read_config(self, config_name: str) -> Dict[str, Any]:
        """Read configuration file"""
        config_path = os.path.join(self.config_dir, f"{config_name}.json")
        
        try:
            content = self.file_manager.read_file(config_path)
            return json.loads(content)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as e:
            self.context.logger.error(f"Invalid JSON in {config_name}: {e}")
            return {}
            
    def write_config(
        self,
        config_name: str,
        data: Dict[str, Any],
        backup: bool = True
    ) -> bool:
        """Write configuration file atomically"""
        config_path = os.path.join(self.config_dir, f"{config_name}.json")
        
        # Create backup if file exists
        if backup and os.path.exists(config_path):
            backup_path = f"{config_path}.backup"
            self.file_manager.copy_file(
                config_path,
                backup_path,
                overwrite=True,
                preserve_acl=True
            )
            
        # Write atomically using temp file
        content = json.dumps(data, indent=2)
        return self.file_manager.write_file(
            config_path,
            content,
            atomic=True,
            create_dirs=True
        )
        
    def delete_config(self, config_name: str, move_to_recycle: bool = True) -> bool:
        """Delete configuration file"""
        config_path = os.path.join(self.config_dir, f"{config_name}.json")
        
        return self.file_manager.delete_file(
            config_path,
            move_to_recycle=move_to_recycle
        )
        
    def list_configs(self) -> List[str]:
        """List all configuration files"""
        configs = []
        for file_info in self.file_manager.list_directory(
            self.config_dir,
            pattern="*.json",
            include_hidden=False
        ):
            if not file_info.is_directory:
                name = os.path.splitext(file_info.name)[0]
                configs.append(name)
        return configs
```

### 12.2 Registry Configuration Example

```python
# Example: Agent Registry Configuration

class AgentRegistryConfig:
    """
    Manages agent configuration in Windows Registry
    """
    
    REGISTRY_ROOT = "HKLM\\SOFTWARE\\OpenClawAgent"
    USER_ROOT = "HKCU\\Software\\OpenClawAgent"
    
    def __init__(self, agent_context: AgentContext):
        self.context = agent_context
        self.registry = RegistryManager()
        self.logger = logging.getLogger("AgentRegistryConfig")
        self._ensure_structure()
        
    def _ensure_structure(self) -> None:
        """Ensure registry structure exists"""
        # Create main keys
        for key_path in [
            f"{self.REGISTRY_ROOT}\\Config",
            f"{self.REGISTRY_ROOT}\\Services",
            f"{self.REGISTRY_ROOT}\\Loops",
            f"{self.USER_ROOT}\\Preferences",
            f"{self.USER_ROOT}\\History",
        ]:
            self.registry.open_key(
                "HKLM" if key_path.startswith("HKLM") else "HKCU",
                key_path[5:],  # Remove HKLM\ or HKCU\
                create=True
            )
            
    def get_machine_config(self, key: str, default: Any = None) -> Any:
        """Get machine-wide configuration value"""
        return self.registry.read_value(
            "HKLM",
            f"SOFTWARE\\OpenClawAgent\\Config",
            key,
            default
        )[0]
        
    def set_machine_config(self, key: str, value: Any) -> bool:
        """Set machine-wide configuration value"""
        return self.registry.write_value(
            "HKLM",
            f"SOFTWARE\\OpenClawAgent\\Config",
            key,
            value
        )
        
    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """Get user-specific preference"""
        return self.registry.read_value(
            "HKCU",
            f"Software\\OpenClawAgent\\Preferences",
            key,
            default
        )[0]
        
    def set_user_preference(self, key: str, value: Any) -> bool:
        """Set user-specific preference"""
        return self.registry.write_value(
            "HKCU",
            f"Software\\OpenClawAgent\\Preferences",
            key,
            value
        )
        
    def get_loop_config(self, loop_id: int) -> Dict[str, Any]:
        """Get loop configuration"""
        loop_key = f"SOFTWARE\\OpenClawAgent\\Loops\\Loop{loop_id:02d}"
        config = {}
        
        for value_name, value, value_type in self.registry.enum_values(
            "HKLM", loop_key
        ):
            config[value_name] = value
            
        return config
        
    def set_loop_config(self, loop_id: int, config: Dict[str, Any]) -> bool:
        """Set loop configuration"""
        loop_key = f"SOFTWARE\\OpenClawAgent\\Loops\\Loop{loop_id:02d}"
        
        # Ensure key exists
        self.registry.open_key("HKLM", loop_key, create=True)
        
        # Write all config values
        for key, value in config.items():
            if not self.registry.write_value("HKLM", loop_key, key, value):
                return False
                
        return True
        
    def watch_config_changes(self, callback: Callable) -> WatchHandle:
        """Watch for configuration changes"""
        return self.registry.watch_key(
            "HKLM",
            "SOFTWARE\\OpenClawAgent\\Config",
            callback,
            filter=REG_NOTIFY_CHANGE_LAST_SET
        )
```

### 12.3 File Watching Example

```python
# Example: Agent File System Monitoring

class AgentFileMonitor:
    """
    Monitors agent-related directories for changes
    """
    
    def __init__(self, agent_context: AgentContext):
        self.context = agent_context
        self.watcher_manager = FileWatcherManager()
        self.path_manager = WindowsPathManager()
        self.logger = logging.getLogger("AgentFileMonitor")
        
        self._watches: Dict[str, WatchHandle] = {}
        
    def start_monitoring(self) -> None:
        """Start monitoring agent directories"""
        # Monitor config directory
        config_dir = self.path_manager.resolve_path(
            "{localappdata}\\OpenClawAgent\\config"
        )
        self._watches["config"] = self.watcher_manager.add_watch(
            config_dir,
            self._on_config_change,
            recursive=True,
            filter=(
                FileWatcherManager.FILE_NOTIFY_CHANGE_FILE_NAME |
                FileWatcherManager.FILE_NOTIFY_CHANGE_LAST_WRITE
            )
        )
        
        # Monitor data directory
        data_dir = self.path_manager.resolve_path(
            "{localappdata}\\OpenClawAgent\\data"
        )
        self._watches["data"] = self.watcher_manager.add_watch(
            data_dir,
            self._on_data_change,
            recursive=True
        )
        
        self.logger.info(f"Started monitoring {len(self._watches)} directories")
        
    def stop_monitoring(self) -> None:
        """Stop all monitoring"""
        for name, handle in self._watches.items():
            self.watcher_manager.remove_watch(handle)
            self.logger.debug(f"Stopped watching {name}")
            
        self._watches.clear()
        
    def _on_config_change(self, event: FileChangeEvent) -> None:
        """Handle configuration file changes"""
        self.logger.info(
            f"Config change: {event.action.value} - {event.path}"
        )
        
        # Notify agent core
        self.context.event_bus.publish(
            "config_changed",
            {
                "action": event.action.value,
                "path": event.path,
                "timestamp": event.timestamp.isoformat()
            }
        )
        
    def _on_data_change(self, event: FileChangeEvent) -> None:
        """Handle data directory changes"""
        self.logger.debug(
            f"Data change: {event.action.value} - {event.path}"
        )
        
        # Update agent state if needed
        if event.action == FileAction.CREATED:
            self.context.event_bus.publish("data_created", {"path": event.path})
        elif event.action == FileAction.MODIFIED:
            self.context.event_bus.publish("data_modified", {"path": event.path})
```

---

## Appendix A: Windows API Function Reference

### File Operations (kernel32.dll)

| Function | Purpose |
|----------|---------|
| CreateFileW | Open/create file |
| ReadFile | Read from file |
| WriteFile | Write to file |
| CloseHandle | Close file handle |
| DeleteFileW | Delete file |
| MoveFileExW | Move/rename file |
| CopyFileExW | Copy file |
| GetFileAttributesExW | Get file attributes |
| SetFileAttributesW | Set file attributes |
| GetFileInformationByHandle | Get detailed info |
| SetFileInformationByHandle | Set file info |
| GetFileSizeEx | Get file size |
| SetEndOfFile | Truncate/extend file |
| CreateDirectoryW | Create directory |
| RemoveDirectoryW | Remove directory |
| FindFirstFileExW | Begin file search |
| FindNextFileW | Continue file search |
| ReadDirectoryChangesW | Monitor changes |
| GetFullPathNameW | Get full path |
| GetLongPathNameW | Get long path |
| GetShortPathNameW | Get 8.3 path |
| GetTempPathW | Get temp directory |
| GetTempFileNameW | Generate temp name |

### Registry Operations (advapi32.dll)

| Function | Purpose |
|----------|---------|
| RegOpenKeyExW | Open registry key |
| RegCreateKeyExW | Create registry key |
| RegCloseKey | Close key handle |
| RegQueryValueExW | Read value |
| RegSetValueExW | Write value |
| RegDeleteValueW | Delete value |
| RegDeleteKeyExW | Delete key |
| RegEnumKeyExW | Enumerate subkeys |
| RegEnumValueW | Enumerate values |
| RegQueryInfoKeyW | Get key info |
| RegNotifyChangeKeyValue | Watch for changes |
| RegSaveKeyExW | Save key to file |
| RegLoadKeyW | Load key from file |
| RegConnectRegistryW | Connect to remote |

### Security Operations (advapi32.dll)

| Function | Purpose |
|----------|---------|
| GetFileSecurityW | Get security descriptor |
| SetFileSecurityW | Set security descriptor |
| GetNamedSecurityInfoW | Get object security |
| SetNamedSecurityInfoW | Set object security |
| LookupAccountNameW | Name to SID |
| LookupAccountSidW | SID to name |
| OpenProcessToken | Get process token |
| OpenThreadToken | Get thread token |
| GetTokenInformation | Query token info |
| AdjustTokenPrivileges | Enable/disable privileges |
| ImpersonateLoggedOnUser | Impersonate user |
| RevertToSelf | End impersonation |

---

## Appendix B: Data Structures

### FILETIME Structure
```c
typedef struct _FILETIME {
    DWORD dwLowDateTime;
    DWORD dwHighDateTime;
} FILETIME;
```

### WIN32_FIND_DATA Structure
```c
typedef struct _WIN32_FIND_DATAW {
    DWORD dwFileAttributes;
    FILETIME ftCreationTime;
    FILETIME ftLastAccessTime;
    FILETIME ftLastWriteTime;
    DWORD nFileSizeHigh;
    DWORD nFileSizeLow;
    DWORD dwReserved0;
    DWORD dwReserved1;
    WCHAR cFileName[MAX_PATH];
    WCHAR cAlternateFileName[14];
} WIN32_FIND_DATAW;
```

### SECURITY_DESCRIPTOR Structure
```c
typedef struct _SECURITY_DESCRIPTOR {
    BYTE Revision;
    BYTE Sbz1;
    WORD Control;
    PSID Owner;
    PSID Group;
    PACL Sacl;
    PACL Dacl;
} SECURITY_DESCRIPTOR;
```

---

## Document Information

- **Version:** 1.0.0
- **Last Updated:** 2025-01-20
- **Author:** Windows Systems Integration Team
- **Status:** Technical Specification
- **Target Platform:** Windows 10/11 (x64)
- **Framework:** OpenClaw AI Agent System

---

*End of Technical Specification*
