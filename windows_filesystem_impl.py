"""
Windows 10 File System & Registry Implementation
OpenClaw-Inspired AI Agent Framework

This module provides concrete implementations for:
- NTFS file operations
- Windows path handling
- Registry access
- File watching
- ACL management
- Temporary file management
- Recycle bin operations

Requires: pywin32, ctypes
"""

import os
import sys
import json
import time
import shutil
import logging
import tempfile
import threading
import atexit
from pathlib import Path
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Tuple, Optional, Union, 
    Iterator, Callable, Set, BinaryIO
)
from enum import Enum, IntFlag, auto
from datetime import datetime
from collections import namedtuple
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import ctypes
from ctypes import wintypes

# Windows API imports (requires pywin32)
try:
    import win32file
    import win32api
    import win32security
    import win32con
    import win32event
    import win32process
    import win32job
    import pywintypes
    import winreg
    from win32com.shell import shell, shellcon
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False
    logging.warning("pywin32 not available - Windows-specific features disabled")


# =============================================================================
# CONSTANTS
# =============================================================================

class FileAccessRights:
    """Windows file access rights"""
    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    GENERIC_EXECUTE = 0x20000000
    GENERIC_ALL = 0x10000000
    
    DELETE = 0x00010000
    READ_CONTROL = 0x00020000
    WRITE_DAC = 0x00040000
    WRITE_OWNER = 0x00080000
    SYNCHRONIZE = 0x00100000
    
    FILE_READ_DATA = 0x0001
    FILE_WRITE_DATA = 0x0002
    FILE_APPEND_DATA = 0x0004
    FILE_READ_EA = 0x0008
    FILE_WRITE_EA = 0x0010
    FILE_EXECUTE = 0x0020
    FILE_DELETE_CHILD = 0x0040
    FILE_READ_ATTRIBUTES = 0x0080
    FILE_WRITE_ATTRIBUTES = 0x0100

class FileShareMode:
    """Windows file share modes"""
    READ = 0x00000001
    WRITE = 0x00000002
    DELETE = 0x00000004

class FileCreationDisposition:
    """Windows file creation dispositions"""
    CREATE_NEW = 1
    CREATE_ALWAYS = 2
    OPEN_EXISTING = 3
    OPEN_ALWAYS = 4
    TRUNCATE_EXISTING = 5

class FileAttributes:
    """Windows file attributes"""
    READONLY = 0x00000001
    HIDDEN = 0x00000002
    SYSTEM = 0x00000004
    DIRECTORY = 0x00000010
    ARCHIVE = 0x00000020
    DEVICE = 0x00000040
    NORMAL = 0x00000080
    TEMPORARY = 0x00000100
    SPARSE_FILE = 0x00000200
    REPARSE_POINT = 0x00000400
    COMPRESSED = 0x00000800
    OFFLINE = 0x00001000
    NOT_CONTENT_INDEXED = 0x00002000
    ENCRYPTED = 0x00004000
    
    DELETE_ON_CLOSE = 0x04000000
    SEQUENTIAL_SCAN = 0x08000000
    RANDOM_ACCESS = 0x10000000

class RegistryValueType(Enum):
    """Windows registry value types"""
    REG_NONE = 0
    REG_SZ = 1
    REG_EXPAND_SZ = 2
    REG_BINARY = 3
    REG_DWORD = 4
    REG_DWORD_BIG_ENDIAN = 5
    REG_LINK = 6
    REG_MULTI_SZ = 7
    REG_RESOURCE_LIST = 8
    REG_FULL_RESOURCE_DESCRIPTOR = 9
    REG_RESOURCE_REQUIREMENTS_LIST = 10
    REG_QWORD = 11

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FileInfo:
    """File information structure"""
    name: str
    full_path: str
    size: int = 0
    creation_time: datetime = field(default_factory=datetime.now)
    last_access_time: datetime = field(default_factory=datetime.now)
    last_write_time: datetime = field(default_factory=datetime.now)
    attributes: int = 0
    is_directory: bool = False
    is_reparse_point: bool = False
    is_compressed: bool = False
    is_encrypted: bool = False

@dataclass
class SecurityDescriptor:
    """Security descriptor structure"""
    owner: str = ""
    group: str = ""
    dacl: Optional[Any] = None
    sacl: Optional[Any] = None

@dataclass
class FileChangeEvent:
    """File change event structure"""
    action: str
    path: str
    old_path: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

# =============================================================================
# NTFS FILE MANAGER
# =============================================================================

class NTFSFileManager:
    """
    NTFS File Operations Manager
    Provides atomic, transactional file operations
    """
    
    def __init__(self):
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
        Read file contents
        
        Args:
            path: Full file path
            encoding: Text encoding
            binary: Read as binary
            offset: Starting byte offset
            size: Bytes to read (-1 for all)
            
        Returns:
            File contents
        """
        path = os.path.expandvars(path)
        
        mode = 'rb' if binary else 'r'
        
        with open(path, mode, encoding=None if binary else encoding) as f:
            if offset > 0:
                f.seek(offset)
            if size > 0:
                content = f.read(size)
            else:
                content = f.read()
                
        return content
        
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
        Write content to file
        
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
        path = os.path.expandvars(path)
        
        # Create directories if needed
        if create_dirs:
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
        
        # Determine mode
        if isinstance(content, bytes):
            mode = 'ab' if append else 'wb'
            encoding = None
        else:
            mode = 'a' if append else 'w'
        
        # Atomic write using temp file
        if atomic and not append:
            temp_path = path + '.tmp'
            try:
                with open(temp_path, mode, encoding=encoding) as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                
                # Atomic rename
                if HAS_WIN32:
                    win32file.MoveFileEx(
                        temp_path, 
                        path,
                        win32file.MOVEFILE_REPLACE_EXISTING | 
                        win32file.MOVEFILE_WRITE_THROUGH
                    )
                else:
                    os.replace(temp_path, path)
                    
                return True
            except OSError as e:
                self.logger.error(f"Atomic write failed: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise
        else:
            with open(path, mode, encoding=encoding) as f:
                f.write(content)
                f.flush()
            return True
            
    def delete_file(
        self,
        path: str,
        secure: bool = False,
        move_to_recycle: bool = True
    ) -> bool:
        """
        Delete file
        
        Args:
            path: File to delete
            secure: Overwrite before deletion
            move_to_recycle: Move to recycle bin
            
        Returns:
            True if successful
        """
        path = os.path.expandvars(path)
        
        if not os.path.exists(path):
            return False
            
        if move_to_recycle and HAS_WIN32:
            return self._move_to_recycle_bin(path)
        
        if secure:
            self._secure_delete(path)
        else:
            os.remove(path)
            
        return True
        
    def _move_to_recycle_bin(self, path: str) -> bool:
        """Move file to recycle bin"""
        try:
            from win32com.shell import shell, shellcon
            shell.SHFileOperation((
                0,
                shellcon.FO_DELETE,
                path,
                None,
                shellcon.FOF_ALLOWUNDO | shellcon.FOF_NOCONFIRMATION | 
                shellcon.FOF_SILENT,
                None,
                None
            ))
            return True
        except (OSError, ImportError) as e:
            self.logger.error(f"Recycle bin operation failed: {e}")
            return False

    def _secure_delete(self, path: str, passes: int = 3) -> None:
        """Securely delete file by overwriting"""
        file_size = os.path.getsize(path)
        
        with open(path, 'r+b') as f:
            for _ in range(passes):
                f.seek(0)
                f.write(os.urandom(file_size))
                f.flush()
                os.fsync(f.fileno())
                
        os.remove(path)
        
    def copy_file(
        self,
        source: str,
        destination: str,
        overwrite: bool = False,
        preserve_acl: bool = True,
        preserve_timestamps: bool = True
    ) -> bool:
        """
        Copy file
        
        Args:
            source: Source file path
            destination: Destination path
            overwrite: Allow overwriting
            preserve_acl: Copy ACL
            preserve_timestamps: Copy timestamps
            
        Returns:
            True if successful
        """
        source = os.path.expandvars(source)
        destination = os.path.expandvars(destination)
        
        if not os.path.exists(source):
            raise FileNotFoundError(f"Source not found: {source}")
            
        if os.path.exists(destination) and not overwrite:
            raise FileExistsError(f"Destination exists: {destination}")
        
        # Create destination directory
        dest_dir = os.path.dirname(destination)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)
        
        if HAS_WIN32 and preserve_acl:
            # Use Windows API for ACL preservation
            win32api.CopyFile(source, destination, not overwrite)
        else:
            shutil.copy2(source, destination)
            
        return True
        
    def move_file(
        self,
        source: str,
        destination: str,
        overwrite: bool = False,
        allow_cross_volume: bool = True
    ) -> bool:
        """
        Move file
        
        Args:
            source: Source file path
            destination: Destination path
            overwrite: Allow overwriting
            allow_cross_volume: Allow cross-volume moves
            
        Returns:
            True if successful
        """
        source = os.path.expandvars(source)
        destination = os.path.expandvars(destination)
        
        if not os.path.exists(source):
            raise FileNotFoundError(f"Source not found: {source}")
            
        if os.path.exists(destination) and not overwrite:
            raise FileExistsError(f"Destination exists: {destination}")
        
        # Create destination directory
        dest_dir = os.path.dirname(destination)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)
        
        # Check if same volume
        if not allow_cross_volume:
            src_drive = os.path.splitdrive(source)[0]
            dst_drive = os.path.splitdrive(destination)[0]
            if src_drive.lower() != dst_drive.lower():
                raise ValueError("Cross-volume move not allowed")
        
        if HAS_WIN32:
            flags = win32file.MOVEFILE_WRITE_THROUGH
            if overwrite:
                flags |= win32file.MOVEFILE_REPLACE_EXISTING
            if allow_cross_volume:
                flags |= win32file.MOVEFILE_COPY_ALLOWED
                
            win32file.MoveFileEx(source, destination, flags)
        else:
            shutil.move(source, destination)
            
        return True
        
    def get_file_info(self, path: str) -> FileInfo:
        """Get comprehensive file information"""
        path = os.path.expandvars(path)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
        
        stat = os.stat(path)
        
        # Get Windows-specific attributes
        attributes = 0
        if HAS_WIN32:
            attributes = win32file.GetFileAttributes(path)
        
        return FileInfo(
            name=os.path.basename(path),
            full_path=os.path.abspath(path),
            size=stat.st_size,
            creation_time=datetime.fromtimestamp(stat.st_ctime),
            last_access_time=datetime.fromtimestamp(stat.st_atime),
            last_write_time=datetime.fromtimestamp(stat.st_mtime),
            attributes=attributes,
            is_directory=os.path.isdir(path),
            is_reparse_point=bool(attributes & FileAttributes.REPARSE_POINT),
            is_compressed=bool(attributes & FileAttributes.COMPRESSED),
            is_encrypted=bool(attributes & FileAttributes.ENCRYPTED)
        )
        
    def list_directory(
        self,
        path: str,
        pattern: str = "*",
        include_hidden: bool = False,
        include_system: bool = False,
        recursive: bool = False
    ) -> Iterator[FileInfo]:
        """List directory contents"""
        path = os.path.expandvars(path)
        
        if not os.path.isdir(path):
            raise NotADirectoryError(f"Not a directory: {path}")
        
        import fnmatch
        
        for root, dirs, files in os.walk(path):
            # Filter directories if not recursive
            if not recursive and root != path:
                break
                
            for name in files + dirs:
                if not fnmatch.fnmatch(name, pattern):
                    continue
                    
                full_path = os.path.join(root, name)
                
                try:
                    info = self.get_file_info(full_path)
                    
                    # Filter hidden/system
                    if not include_hidden and (info.attributes & FileAttributes.HIDDEN):
                        continue
                    if not include_system and (info.attributes & FileAttributes.SYSTEM):
                        continue
                        
                    yield info
                except (OSError, PermissionError) as e:
                    self.logger.warning(f"Could not get info for {full_path}: {e}")

# =============================================================================
# WINDOWS PATH MANAGER
# =============================================================================

class WindowsPathManager:
    """Windows path resolution and management"""
    
    # Known folder GUIDs
    KNOWN_FOLDERS = {
        "Desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
        "Documents": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
        "Downloads": "374DE290-123F-4565-9164-39C4925E467B",
        "Pictures": "33E28130-4E1E-4676-835A-98395C3BC3BB",
        "Music": "4BD8D571-6D19-48D3-BE97-422220080E43",
        "Videos": "18989B1D-99B5-455B-841C-AB7C74E4DDFC",
    }
    
    # Reserved Windows filenames
    RESERVED_NAMES = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5",
        "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5",
        "LPT6", "LPT7", "LPT8", "LPT9"
    }
    
    def __init__(self):
        self.logger = logging.getLogger("WindowsPathManager")
        self._folder_cache: Dict[str, str] = {}
        self._load_special_folders()
        
    def _load_special_folders(self):
        """Load special folder paths"""
        # Environment variable based paths
        self._folder_cache.update({
            "userprofile": os.environ.get("USERPROFILE", ""),
            "appdata": os.environ.get("APPDATA", ""),
            "localappdata": os.environ.get("LOCALAPPDATA", ""),
            "temp": os.environ.get("TEMP", ""),
            "systemroot": os.environ.get("SystemRoot", r"C:\Windows"),
            "programfiles": os.environ.get("ProgramFiles", r"C:\Program Files"),
            "programfiles(x86)": os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            "programdata": os.environ.get("ProgramData", r"C:\ProgramData"),
            "public": os.environ.get("PUBLIC", r"C:\Users\Public"),
        })
        
        # Known folders via Windows API
        if HAS_WIN32:
            try:
                from win32com.shell import shell, shellcon
                for name, guid in self.KNOWN_FOLDERS.items():
                    try:
                        path = shell.SHGetKnownFolderPath(guid)
                        self._folder_cache[name.lower()] = path
                    except OSError:
                        pass
            except (OSError, ImportError) as e:
                self.logger.warning(f"Could not load known folders: {e}")
                
    def resolve_path(self, path: str, expand_env: bool = True) -> str:
        """
        Resolve path with environment variables and special folders
        
        Args:
            path: Path string (may contain env vars and special folder tokens)
            expand_env: Expand environment variables
            
        Returns:
            Absolute, normalized path
        """
        result = path
        
        # Expand environment variables
        if expand_env:
            result = os.path.expandvars(result)
        
        # Replace special folder tokens {foldername}
        for token, folder_path in self._folder_cache.items():
            result = result.replace(f"{{{token}}}", folder_path)
            result = result.replace(f"%{token.upper()}%", folder_path)
        
        # Normalize path
        result = os.path.normpath(result)
        
        # Convert to absolute path
        if not os.path.isabs(result):
            result = os.path.abspath(result)
            
        return result
        
    def get_special_folder(self, folder_id: str) -> str:
        """Get special folder path"""
        folder_id = folder_id.lower().strip('{}')
        return self._folder_cache.get(folder_id, "")
        
    def normalize_path(self, path: str) -> str:
        """Normalize path for Windows"""
        path = os.path.expandvars(path)
        path = os.path.normpath(path)
        path = os.path.abspath(path)
        return path
        
    def to_long_path(self, path: str) -> str:
        """Convert to extended-length path"""
        path = self.normalize_path(path)
        if not path.startswith("\\\\?\\"):
            if path.startswith("\\\\"):
                path = "\\\\?\\UNC\\" + path[2:]
            else:
                path = "\\\\?\\" + path
        return path
        
    def from_long_path(self, path: str) -> str:
        """Remove extended-length path prefix"""
        if path.startswith("\\\\?\\UNC\\"):
            return "\\\\" + path[8:]
        elif path.startswith("\\\\?\\"):
            return path[4:]
        return path
        
    def is_valid_filename(self, name: str) -> bool:
        """Check if filename is valid for Windows"""
        # Check empty
        if not name or not name.strip():
            return False
            
        # Check reserved names
        base_name = name.split('.')[0].upper()
        if base_name in self.RESERVED_NAMES:
            return False
        
        # Check invalid characters
        invalid_chars = '<>:"/\\|?*'
        if any(c in name for c in invalid_chars):
            return False
            
        # Check length
        if len(name) > 255:
            return False
            
        # Check ending
        if name.endswith(' ') or name.endswith('.'):
            return False
            
        return True
        
    def sanitize_filename(self, name: str, replacement: str = "_") -> str:
        """Sanitize filename for Windows"""
        # Replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, replacement)
        
        # Remove control characters
        name = ''.join(c for c in name if ord(c) >= 32)
        
        # Check reserved name
        base_name = name.split('.')[0].upper()
        if base_name in self.RESERVED_NAMES:
            name = replacement + name
        
        # Trim spaces and dots from ends
        name = name.strip(' .')
        
        # Truncate if too long
        if len(name) > 255:
            name = name[:255]
            
        return name or replacement

# =============================================================================
# REGISTRY MANAGER
# =============================================================================

class RegistryManager:
    """Windows Registry access manager"""
    
    # Root key mapping
    ROOT_KEYS = {
        "HKCR": winreg.HKEY_CLASSES_ROOT,
        "HKCU": winreg.HKEY_CURRENT_USER,
        "HKLM": winreg.HKEY_LOCAL_MACHINE,
        "HKU": winreg.HKEY_USERS,
        "HKCC": winreg.HKEY_CURRENT_CONFIG,
    }
    
    def __init__(self):
        self.logger = logging.getLogger("RegistryManager")
        
    def _parse_path(self, full_path: str) -> Tuple[int, str]:
        """Parse full registry path into root and subkey"""
        parts = full_path.split('\\', 1)
        root_name = parts[0].upper()
        subkey = parts[1] if len(parts) > 1 else ""
        
        if root_name not in self.ROOT_KEYS:
            raise ValueError(f"Invalid root key: {root_name}")
            
        return self.ROOT_KEYS[root_name], subkey
        
    def open_key(
        self,
        root: Union[int, str],
        subkey: str,
        access: int = winreg.KEY_READ,
        create: bool = False
    ) -> int:
        """Open or create registry key"""
        if isinstance(root, str):
            root = self.ROOT_KEYS.get(root.upper(), root)
            
        if create:
            return winreg.CreateKeyEx(root, subkey, 0, access)
        else:
            return winreg.OpenKeyEx(root, subkey, 0, access)
            
    def close_key(self, key_handle: int) -> None:
        """Close registry key handle"""
        winreg.CloseKey(key_handle)
        
    def read_value(
        self,
        root: Union[int, str],
        subkey: str,
        value_name: str,
        default: Any = None
    ) -> Tuple[Any, RegistryValueType]:
        """Read registry value"""
        try:
            if isinstance(root, str):
                root, subkey = self._parse_key_path(f"{root}\\{subkey}")
                
            with winreg.OpenKey(root, subkey, 0, winreg.KEY_READ) as key:
                value, value_type = winreg.QueryValueEx(key, value_name)
                return value, RegistryValueType(value_type)
        except FileNotFoundError:
            return default, RegistryValueType.REG_NONE
        except WindowsError:
            return default, RegistryValueType.REG_NONE
            
    def write_value(
        self,
        root: Union[int, str],
        subkey: str,
        value_name: str,
        value: Any,
        value_type: RegistryValueType = RegistryValueType.REG_SZ
    ) -> bool:
        """Write registry value"""
        try:
            if isinstance(root, str):
                root, subkey = self._parse_key_path(f"{root}\\{subkey}")
                
            with winreg.CreateKey(root, subkey) as key:
                winreg.SetValueEx(key, value_name, 0, value_type.value, value)
            return True
        except WindowsError as e:
            self.logger.error(f"Failed to write registry value: {e}")
            return False
            
    def delete_value(
        self,
        root: Union[int, str],
        subkey: str,
        value_name: str
    ) -> bool:
        """Delete registry value"""
        try:
            if isinstance(root, str):
                root, subkey = self._parse_key_path(f"{root}\\{subkey}")
                
            with winreg.OpenKey(root, subkey, 0, winreg.KEY_WRITE) as key:
                winreg.DeleteValue(key, value_name)
            return True
        except WindowsError as e:
            self.logger.error(f"Failed to delete registry value: {e}")
            return False
            
    def delete_key(
        self,
        root: Union[int, str],
        subkey: str,
        recursive: bool = False
    ) -> bool:
        """Delete registry key"""
        try:
            if isinstance(root, str):
                root, subkey = self._parse_key_path(f"{root}\\{subkey}")
                
            if recursive:
                # Delete subkeys first
                with winreg.OpenKey(root, subkey, 0, winreg.KEY_READ) as key:
                    i = 0
                    while True:
                        try:
                            subsubkey = winreg.EnumKey(key, i)
                            self.delete_key(root, f"{subkey}\\{subsubkey}", True)
                        except OSError:
                            break
                        i += 1
                        
            winreg.DeleteKey(root, subkey)
            return True
        except WindowsError as e:
            self.logger.error(f"Failed to delete registry key: {e}")
            return False
            
    def enum_keys(
        self,
        root: Union[int, str],
        subkey: str = ""
    ) -> Iterator[str]:
        """Enumerate subkey names"""
        try:
            if isinstance(root, str):
                root, subkey = self._parse_key_path(f"{root}\\{subkey}")
                
            with winreg.OpenKey(root, subkey, 0, winreg.KEY_READ) as key:
                i = 0
                while True:
                    try:
                        yield winreg.EnumKey(key, i)
                    except OSError:
                        break
                    i += 1
        except WindowsError as e:
            self.logger.error(f"Failed to enumerate keys: {e}")
            
    def enum_values(
        self,
        root: Union[int, str],
        subkey: str = ""
    ) -> Iterator[Tuple[str, Any, RegistryValueType]]:
        """Enumerate value names and data"""
        try:
            if isinstance(root, str):
                root, subkey = self._parse_key_path(f"{root}\\{subkey}")
                
            with winreg.OpenKey(root, subkey, 0, winreg.KEY_READ) as key:
                i = 0
                while True:
                    try:
                        name, value, value_type = winreg.EnumValue(key, i)
                        yield name, value, RegistryValueType(value_type)
                    except OSError:
                        break
                    i += 1
        except WindowsError as e:
            self.logger.error(f"Failed to enumerate values: {e}")
            
    def key_exists(self, root: Union[int, str], subkey: str) -> bool:
        """Check if registry key exists"""
        try:
            if isinstance(root, str):
                root, subkey = self._parse_key_path(f"{root}\\{subkey}")
                
            with winreg.OpenKey(root, subkey, 0, winreg.KEY_READ):
                return True
        except WindowsError:
            return False
            
    def value_exists(
        self,
        root: Union[int, str],
        subkey: str,
        value_name: str
    ) -> bool:
        """Check if registry value exists"""
        try:
            if isinstance(root, str):
                root, subkey = self._parse_key_path(f"{root}\\{subkey}")
                
            with winreg.OpenKey(root, subkey, 0, winreg.KEY_READ) as key:
                winreg.QueryValueEx(key, value_name)
                return True
        except WindowsError:
            return False

# =============================================================================
# FILE WATCHER
# =============================================================================

class FileWatcher:
    """Windows file system change watcher"""
    
    # Notification filters
    NOTIFY_CHANGE_FILE_NAME = 0x00000001
    NOTIFY_CHANGE_DIR_NAME = 0x00000002
    NOTIFY_CHANGE_ATTRIBUTES = 0x00000004
    NOTIFY_CHANGE_SIZE = 0x00000008
    NOTIFY_CHANGE_LAST_WRITE = 0x00000010
    NOTIFY_CHANGE_LAST_ACCESS = 0x00000020
    NOTIFY_CHANGE_CREATION = 0x00000040
    NOTIFY_CHANGE_SECURITY = 0x00000100
    
    # Action types
    ACTION_ADDED = 0x00000001
    ACTION_REMOVED = 0x00000002
    ACTION_MODIFIED = 0x00000003
    ACTION_RENAMED_OLD = 0x00000004
    ACTION_RENAMED_NEW = 0x00000005
    
    def __init__(self):
        self.logger = logging.getLogger("FileWatcher")
        self._watches: Dict[str, threading.Thread] = {}
        self._stop_events: Dict[str, threading.Event] = {}
        
    def add_watch(
        self,
        path: str,
        callback: Callable[[FileChangeEvent], None],
        recursive: bool = True,
        filter_mask: int = NOTIFY_CHANGE_LAST_WRITE | NOTIFY_CHANGE_FILE_NAME
    ) -> str:
        """
        Add file system watch
        
        Args:
            path: Directory to watch
            callback: Function to call on changes
            recursive: Watch subdirectories
            filter_mask: Change notification filter
            
        Returns:
            Watch ID
        """
        watch_id = f"{path}_{id(callback)}"
        stop_event = threading.Event()
        self._stop_events[watch_id] = stop_event
        
        thread = threading.Thread(
            target=self._watch_loop,
            args=(path, callback, recursive, filter_mask, stop_event),
            daemon=True
        )
        thread.start()
        self._watches[watch_id] = thread
        
        return watch_id
        
    def remove_watch(self, watch_id: str) -> bool:
        """Remove file system watch"""
        if watch_id in self._stop_events:
            self._stop_events[watch_id].set()
            self._watches[watch_id].join(timeout=5)
            del self._stop_events[watch_id]
            del self._watches[watch_id]
            return True
        return False
        
    def _watch_loop(
        self,
        path: str,
        callback: Callable,
        recursive: bool,
        filter_mask: int,
        stop_event: threading.Event
    ) -> None:
        """Main watch loop"""
        if not HAS_WIN32:
            self.logger.error("File watching requires pywin32")
            return
            
        try:
            # Open directory handle
            handle = win32file.CreateFile(
                path,
                win32con.FILE_LIST_DIRECTORY,
                win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE,
                None,
                win32con.OPEN_EXISTING,
                win32con.FILE_FLAG_BACKUP_SEMANTICS | win32con.FILE_FLAG_OVERLAPPED,
                None
            )
            
            overlapped = pywintypes.OVERLAPPED()
            overlapped.hEvent = win32event.CreateEvent(None, 0, 0, None)
            
            buffer = win32file.AllocateReadBuffer(8192)
            
            while not stop_event.is_set():
                try:
                    # Start async read
                    win32file.ReadDirectoryChangesW(
                        handle,
                        buffer,
                        recursive,
                        filter_mask,
                        overlapped
                    )
                    
                    # Wait for event or stop
                    wait_result = win32event.WaitForSingleObject(
                        overlapped.hEvent,
                        1000  # 1 second timeout
                    )
                    
                    if wait_result == win32event.WAIT_OBJECT_0:
                        # Get results
                        bytes_returned = win32file.GetOverlappedResult(
                            handle, overlapped, False
                        )
                        
                        # Process notifications
                        self._process_notifications(
                            buffer, bytes_returned, path, callback
                        )
                        
                except OSError as e:
                    if not stop_event.is_set():
                        self.logger.error(f"Watch error: {e}")

        except OSError as e:
            self.logger.error(f"Failed to start watch: {e}")
        finally:
            if 'handle' in locals():
                win32file.CloseHandle(handle)
                
    def _process_notifications(
        self,
        buffer,
        bytes_returned: int,
        base_path: str,
        callback: Callable
    ) -> None:
        """Process notification buffer"""
        offset = 0
        
        while offset < bytes_returned:
            # Parse FILE_NOTIFY_INFORMATION structure
            next_entry_offset = int.from_bytes(buffer[offset:offset+4], 'little')
            action = int.from_bytes(buffer[offset+4:offset+8], 'little')
            filename_length = int.from_bytes(buffer[offset+8:offset+12], 'little')
            
            filename = buffer[offset+12:offset+12+filename_length].decode('utf-16-le')
            full_path = os.path.join(base_path, filename)
            
            # Map action to event
            action_map = {
                self.ACTION_ADDED: "created",
                self.ACTION_REMOVED: "deleted",
                self.ACTION_MODIFIED: "modified",
                self.ACTION_RENAMED_OLD: "renamed_from",
                self.ACTION_RENAMED_NEW: "renamed_to",
            }
            
            event = FileChangeEvent(
                action=action_map.get(action, "unknown"),
                path=full_path,
                timestamp=datetime.now()
            )
            
            try:
                callback(event)
            except (OSError, ValueError, RuntimeError) as e:
                self.logger.error(f"Callback error: {e}")
            
            if next_entry_offset == 0:
                break
            offset += next_entry_offset

# =============================================================================
# TEMP FILE MANAGER
# =============================================================================

class TempFileManager:
    """Temporary file management"""
    
    def __init__(
        self,
        base_path: Optional[str] = None,
        prefix: str = "ocl_",
        suffix: str = ".tmp",
        cleanup_on_exit: bool = True
    ):
        self.base_path = base_path or tempfile.gettempdir()
        self.prefix = prefix
        self.suffix = suffix
        self.cleanup_on_exit = cleanup_on_exit
        self._tracked_files: Set[str] = set()
        self._tracked_dirs: Set[str] = set()
        self._lock = threading.Lock()
        self.logger = logging.getLogger("TempFileManager")
        
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
        """Create temporary file"""
        prefix = prefix or self.prefix
        suffix = suffix or self.suffix
        dir = dir or self.base_path
        
        fd, path = tempfile.mkstemp(
            suffix=suffix,
            prefix=prefix,
            dir=dir,
            text=text
        )
        
        if delete_on_close:
            if HAS_WIN32:
                # Set FILE_FLAG_DELETE_ON_CLOSE
                pass
        else:
            with self._lock:
                self._tracked_files.add(path)
                
        return fd, path
        
    def create_temp_directory(
        self,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        dir: Optional[str] = None
    ) -> str:
        """Create temporary directory"""
        prefix = prefix or self.prefix
        suffix = suffix or ""
        dir = dir or self.base_path
        
        path = tempfile.mkdtemp(
            suffix=suffix,
            prefix=prefix,
            dir=dir
        )
        
        with self._lock:
            self._tracked_dirs.add(path)
            
        return path
        
    def delete_temp_file(self, path: str) -> bool:
        """Delete tracked temp file"""
        try:
            if os.path.exists(path):
                os.remove(path)
            with self._lock:
                self._tracked_files.discard(path)
            return True
        except OSError as e:
            self.logger.error(f"Failed to delete temp file: {e}")
            return False
            
    def delete_temp_directory(self, path: str, recursive: bool = True) -> bool:
        """Delete tracked temp directory"""
        try:
            if os.path.exists(path):
                if recursive:
                    shutil.rmtree(path)
                else:
                    os.rmdir(path)
            with self._lock:
                self._tracked_dirs.discard(path)
            return True
        except OSError as e:
            self.logger.error(f"Failed to delete temp directory: {e}")
            return False
            
    def cleanup_all(self) -> Tuple[int, int]:
        """Clean up all tracked temp files and directories"""
        files_deleted = 0
        dirs_deleted = 0
        
        with self._lock:
            # Delete files
            for path in list(self._tracked_files):
                if self.delete_temp_file(path):
                    files_deleted += 1
                    
            # Delete directories
            for path in list(self._tracked_dirs):
                if self.delete_temp_directory(path):
                    dirs_deleted += 1
                    
        return files_deleted, dirs_deleted
        
    def _cleanup_all(self) -> None:
        """Internal cleanup handler"""
        files, dirs = self.cleanup_all()
        if files > 0 or dirs > 0:
            self.logger.info(f"Cleaned up {files} temp files, {dirs} temp directories")

# =============================================================================
# RECYCLE BIN MANAGER
# =============================================================================

class RecycleBinManager:
    """Windows Recycle Bin operations"""
    
    def __init__(self):
        self.logger = logging.getLogger("RecycleBinManager")
        
    def delete_to_recycle_bin(
        self,
        path: str,
        silent: bool = True,
        no_confirmation: bool = True
    ) -> bool:
        """Move file/directory to recycle bin"""
        if not HAS_WIN32:
            self.logger.error("Recycle bin operations require pywin32")
            return False
            
        try:
            from win32com.shell import shell, shellcon
            
            flags = shellcon.FOF_ALLOWUNDO
            if silent:
                flags |= shellcon.FOF_SILENT
            if no_confirmation:
                flags |= shellcon.FOF_NOCONFIRMATION
                
            result, aborted = shell.SHFileOperation((
                0,
                shellcon.FO_DELETE,
                path,
                None,
                flags,
                None,
                None
            ))
            
            return result == 0 and not aborted
        except (OSError, ImportError) as e:
            self.logger.error(f"Recycle bin operation failed: {e}")
            return False
            
    def empty_recycle_bin(
        self,
        drive: Optional[str] = None,
        no_confirmation: bool = True,
        no_progress: bool = True,
        no_sound: bool = True
    ) -> bool:
        """Empty recycle bin"""
        if not HAS_WIN32:
            self.logger.error("Recycle bin operations require pywin32")
            return False
            
        try:
            from win32com.shell import shell, shellcon
            
            flags = 0
            if no_confirmation:
                flags |= shellcon.SHERB_NOCONFIRMATION
            if no_progress:
                flags |= shellcon.SHERB_NOPROGRESSUI
            if no_sound:
                flags |= shellcon.SHERB_NOSOUND
                
            shell.SHEmptyRecycleBin(None, drive, flags)
            return True
        except (OSError, ImportError) as e:
            self.logger.error(f"Empty recycle bin failed: {e}")
            return False

# =============================================================================
# AGENT CONFIGURATION EXAMPLE
# =============================================================================

class AgentConfigManager:
    """
    Example: Agent Configuration File Management
    Demonstrates integrated file system operations
    """
    
    def __init__(self):
        self.file_manager = NTFSFileManager()
        self.path_manager = WindowsPathManager()
        self.registry = RegistryManager()
        self.temp_manager = TempFileManager()
        
        # Resolve agent paths
        self.config_dir = self.path_manager.resolve_path(
            "{localappdata}\\OpenClawAgent\\config"
        )
        self.data_dir = self.path_manager.resolve_path(
            "{localappdata}\\OpenClawAgent\\data"
        )
        self.log_dir = self.path_manager.resolve_path(
            "{localappdata}\\OpenClawAgent\\logs"
        )
        
        # Ensure directories exist
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
    def read_config(self, config_name: str) -> Dict[str, Any]:
        """Read configuration file"""
        config_path = os.path.join(self.config_dir, f"{config_name}.json")
        
        try:
            content = self.file_manager.read_file(config_path)
            return json.loads(content)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
            
    def write_config(
        self,
        config_name: str,
        data: Dict[str, Any],
        backup: bool = True
    ) -> bool:
        """Write configuration file atomically"""
        config_path = os.path.join(self.config_dir, f"{config_name}.json")
        
        # Create backup
        if backup and os.path.exists(config_path):
            backup_path = f"{config_path}.backup"
            self.file_manager.copy_file(
                config_path,
                backup_path,
                overwrite=True
            )
            
        # Write
        content = json.dumps(data, indent=2)
        return self.file_manager.write_file(
            config_path,
            content,
            atomic=True
        )
        
    def get_registry_config(self, key: str, default: Any = None) -> Any:
        """Get configuration from registry"""
        return self.registry.read_value(
            "HKLM",
            "SOFTWARE\\OpenClawAgent\\Config",
            key,
            default
        )[0]
        
    def set_registry_config(self, key: str, value: Any) -> bool:
        """Set configuration in registry"""
        return self.registry.write_value(
            "HKLM",
            "SOFTWARE\\OpenClawAgent\\Config",
            key,
            value,
            RegistryValueType.REG_SZ if isinstance(value, str) else RegistryValueType.REG_DWORD
        )

# =============================================================================
# MAIN / TEST
# =============================================================================

if __name__ == "__main__":
    # Test the implementations
    print("Windows File System & Registry Implementation Test")
    print("=" * 50)
    
    # Path Manager Test
    print("\n1. Testing WindowsPathManager...")
    pm = WindowsPathManager()
    test_path = "{localappdata}\\OpenClawAgent\\test.txt"
    resolved = pm.resolve_path(test_path)
    print(f"   Input:  {test_path}")
    print(f"   Output: {resolved}")
    
    # File Manager Test
    print("\n2. Testing NTFSFileManager...")
    fm = NTFSFileManager()
    temp_dir = tempfile.mkdtemp()
    test_file = os.path.join(temp_dir, "test.txt")
    
    # Write test
    fm.write_file(test_file, "Hello, World!")
    print(f"   Written: {test_file}")
    
    # Read test
    content = fm.read_file(test_file)
    print(f"   Read: {content}")
    
    # File info test
    info = fm.get_file_info(test_file)
    print(f"   Size: {info.size} bytes")
    print(f"   Created: {info.creation_time}")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    print(f"   Cleaned up temp directory")
    
    # Registry Test
    print("\n3. Testing RegistryManager...")
    rm = RegistryManager()
    
    # Write test
    test_key = "HKCU\\Software\\OpenClawAgent\\Test"
    success = rm.write_value("HKCU", "Software\\OpenClawAgent\\Test", "TestValue", "TestData")
    print(f"   Write registry value: {'Success' if success else 'Failed'}")
    
    # Read test
    value, vtype = rm.read_value("HKCU", "Software\\OpenClawAgent\\Test", "TestValue")
    print(f"   Read registry value: {value} (type: {vtype})")
    
    # Cleanup
    rm.delete_key("HKCU", "Software\\OpenClawAgent", recursive=True)
    print(f"   Cleaned up registry key")
    
    # Temp File Manager Test
    print("\n4. Testing TempFileManager...")
    tm = TempFileManager()
    fd, temp_path = tm.create_temp_file(suffix=".txt")
    os.write(fd, b"Test content")
    os.close(fd)
    print(f"   Created temp file: {temp_path}")
    
    files, dirs = tm.cleanup_all()
    print(f"   Cleaned up: {files} files, {dirs} directories")
    
    print("\n" + "=" * 50)
    print("All tests completed!")
