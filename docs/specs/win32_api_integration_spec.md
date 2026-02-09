# Win32 API Integration Specification
## Windows 10 OpenClaw-Inspired AI Agent System

**Version:** 1.0  
**Platform:** Windows 10 (Build 19041+)  
**Architecture:** x64 / x86 / ARM64  
**Date:** 2024

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Core DLLs and API Categories](#2-core-dlls-and-api-categories)
3. [Node.js/Electron FFI Integration](#3-nodejselectron-ffi-integration)
4. [Process Management APIs](#4-process-management-apis)
5. [Window Management APIs](#5-window-management-apis)
6. [System Information APIs](#6-system-information-apis)
7. [File Operations APIs](#7-file-operations-apis)
8. [Registry Access APIs](#8-registry-access-apis)
9. [Security and Privilege APIs](#9-security-and-privilege-apis)

---

## 1. Architecture Overview

### 1.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI Agent Application                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  GPT-5.2    │  │  Agent Core │  │   Agentic Loop Manager │ │
│  │  Engine     │  │  Controller │  │   (15 Hardcoded Loops)  │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    Win32 API Integration Layer                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │ kernel32 │ │ user32   │ │ shell32  │ │ advapi32         │  │
│  │ (System) │ │ (UI)     │ │ (Shell)  │ │ (Security/Reg)   │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │ psapi    │ │ pdh      │ │ netapi32 │ │ wtsapi32         │  │
│  │ (Proc)   │ │ (Perf)   │ │ (Network)│ │ (Terminal)       │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    FFI Bridge (Node-FFI / Koffi)                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  C++ Native Addon / N-API / node-ffi-napi / Koffi       │   │
│  └─────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                         Windows 10 Kernel                       │
│              (NTDLL → Executive → Kernel Drivers)               │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 API Layer Stack

| Layer | Purpose | Key Components |
|-------|---------|----------------|
| Application | AI Agent Logic | GPT-5.2, Loops, Identity |
| Integration | FFI Bridge | Koffi, node-ffi-napi, C++ Addons |
| Win32 APIs | System Interface | kernel32, user32, shell32, advapi32 |
| Native APIs | Kernel Interface | ntdll.dll (Nt/Zw functions) |
| Kernel | Hardware Abstraction | Executive, Kernel, Drivers |

---

## 2. Core DLLs and API Categories

### 2.1 Primary System DLLs

```cpp
// Core DLLs for Agent System
#pragma comment(lib, "kernel32.lib")    // Base OS functions
#pragma comment(lib, "user32.lib")      // Window management
#pragma comment(lib, "shell32.lib")     // Shell operations
#pragma comment(lib, "advapi32.lib")    // Registry & Security
#pragma comment(lib, "psapi.lib")       // Process information
#pragma comment(lib, "pdh.lib")         // Performance counters
#pragma comment(lib, "netapi32.lib")    // Network functions
#pragma comment(lib, "wtsapi32.lib")    // Terminal services
#pragma comment(lib, "iphlpapi.lib")    // IP helper functions
#pragma comment(lib, "ws2_32.lib")      // Winsock
#pragma comment(lib, "comctl32.lib")    // Common controls
#pragma comment(lib, "ole32.lib")       // COM/OLE
#pragma comment(lib, "oleaut32.lib")    // OLE Automation
#pragma comment(lib, "version.lib")     // Version information
#pragma comment(lib, "shlwapi.lib")     // Shell utilities
#pragma comment(lib, "crypt32.lib")     // Cryptography
#pragma comment(lib, "wintrust.lib")    // Trust verification
```

### 2.2 API Category Mapping

| Function Category | Primary DLL | Secondary DLLs | Use Case |
|-------------------|-------------|----------------|----------|
| Process/Thread | kernel32 | ntdll, psapi | Agent process control |
| Memory Management | kernel32 | ntdll | Memory monitoring |
| File Operations | kernel32 | shell32 | File system access |
| Window Management | user32 | kernel32 | UI automation |
| Shell Operations | shell32 | shlwapi | Explorer integration |
| Registry Access | advapi32 | kernel32 | Configuration storage |
| Security/ACL | advapi32 | kernel32 | Permission management |
| Performance Data | pdh | psapi | System monitoring |
| Network Info | iphlpapi | ws2_32 | Network discovery |
| Terminal Services | wtsapi32 | kernel32 | Session management |

---

## 3. Node.js/Electron FFI Integration

### 3.1 FFI Library Options

```javascript
// Option 1: Koffi (Recommended - Modern, Fast)
const koffi = require('koffi');

// Option 2: node-ffi-napi (Legacy compatibility)
const ffi = require('ffi-napi');
const ref = require('ref-napi');

// Option 3: C++ Native Addon (Maximum Performance)
// Using N-API or node-addon-api
```

### 3.2 Koffi Integration Pattern

```javascript
// ============================================
// Koffi FFI Setup for Win32 APIs
// ============================================
const koffi = require('koffi');

// Load Windows DLLs
const kernel32 = koffi.load('kernel32.dll');
const user32 = koffi.load('user32.dll');
const shell32 = koffi.load('shell32.dll');
const advapi32 = koffi.load('advapi32.dll');
const psapi = koffi.load('psapi.dll');
const pdh = koffi.load('pdh.dll');

// Define Windows Types
const WinTypes = {
    // Basic Types
    BOOL: 'int',
    BOOLEAN: 'uchar',
    BYTE: 'uchar',
    CHAR: 'char',
    DWORD: 'uint32',
    WORD: 'uint16',
    INT: 'int',
    UINT: 'uint',
    LONG: 'long',
    ULONG: 'ulong',
    SHORT: 'short',
    USHORT: 'ushort',
    
    // Handle Types
    HANDLE: 'void*',
    HWND: 'void*',
    HINSTANCE: 'void*',
    HMODULE: 'void*',
    HPVOID: 'void*',
    
    // Pointer Types
    LPVOID: 'void*',
    LPCVOID: 'const void*',
    LPSTR: 'char*',
    LPCSTR: 'const char*',
    LPWSTR: 'int16*',  // wchar_t* on Windows
    LPCWSTR: 'const int16*',
    LPTSTR: 'int16*',  // Unicode by default
    LPCTSTR: 'const int16*',
    
    // Special Types
    DWORD_PTR: 'size_t',
    ULONG_PTR: 'size_t',
    SIZE_T: 'size_t',
    SSIZE_T: 'ssize_t',
    LPARAM: 'ssize_t',
    WPARAM: 'size_t',
    LRESULT: 'ssize_t',
    
    // Structure Placeholders
    LPSECURITY_ATTRIBUTES: 'void*',
    LPSTARTUPINFO: 'void*',
    LPPROCESS_INFORMATION: 'void*',
    LPFILETIME: 'void*',
    LPSYSTEM_INFO: 'void*',
    LPMEMORYSTATUSEX: 'void*',
    LPOSVERSIONINFOEX: 'void*',
};

module.exports = { kernel32, user32, shell32, advapi32, psapi, pdh, WinTypes };
```

### 3.3 Calling Convention Mapping

```javascript
// ============================================
// Windows Calling Conventions
// ============================================

// stdcall (most Win32 APIs)
// Function parameters pushed right-to-left, callee cleans stack
// Used by: kernel32, user32, shell32, advapi32, etc.

// cdecl (C runtime, varargs)
// Caller cleans stack
// Used by: CRT functions, printf-style functions

// Koffi handles calling conventions automatically for Windows DLLs
// Default is stdcall for Win32 APIs

// Example: Declaring stdcall function
const Beep = kernel32.func('BOOL __stdcall Beep(DWORD dwFreq, DWORD dwDuration)');

// Example: Declaring cdecl function
const printf = koffi.load('msvcrt.dll').func('int __cdecl printf(const char* fmt, ...)');
```

### 3.4 Structure Definitions

```javascript
// ============================================
// Windows Structure Definitions for Koffi
// ============================================

// FILETIME structure
const FILETIME = koffi.struct('FILETIME', {
    dwLowDateTime: 'uint32',
    dwHighDateTime: 'uint32'
});

// SYSTEMTIME structure
const SYSTEMTIME = koffi.struct('SYSTEMTIME', {
    wYear: 'uint16',
    wMonth: 'uint16',
    wDayOfWeek: 'uint16',
    wDay: 'uint16',
    wHour: 'uint16',
    wMinute: 'uint16',
    wSecond: 'uint16',
    wMilliseconds: 'uint16'
});

// PROCESS_INFORMATION structure
const PROCESS_INFORMATION = koffi.struct('PROCESS_INFORMATION', {
    hProcess: 'void*',
    hThread: 'void*',
    dwProcessId: 'uint32',
    dwThreadId: 'uint32'
});

// STARTUPINFO structure
const STARTUPINFO = koffi.struct('STARTUPINFO', {
    cb: 'uint32',
    lpReserved: 'int16*',
    lpDesktop: 'int16*',
    lpTitle: 'int16*',
    dwX: 'uint32',
    dwY: 'uint32',
    dwXSize: 'uint32',
    dwYSize: 'uint32',
    dwXCountChars: 'uint32',
    dwYCountChars: 'uint32',
    dwFillAttribute: 'uint32',
    dwFlags: 'uint32',
    wShowWindow: 'uint16',
    cbReserved2: 'uint16',
    lpReserved2: 'uint8*',
    hStdInput: 'void*',
    hStdOutput: 'void*',
    hStdError: 'void*'
});

// SECURITY_ATTRIBUTES structure
const SECURITY_ATTRIBUTES = koffi.struct('SECURITY_ATTRIBUTES', {
    nLength: 'uint32',
    lpSecurityDescriptor: 'void*',
    bInheritHandle: 'int'
});

// MEMORYSTATUSEX structure
const MEMORYSTATUSEX = koffi.struct('MEMORYSTATUSEX', {
    dwLength: 'uint32',
    dwMemoryLoad: 'uint32',
    ullTotalPhys: 'uint64',
    ullAvailPhys: 'uint64',
    ullTotalPageFile: 'uint64',
    ullAvailPageFile: 'uint64',
    ullTotalVirtual: 'uint64',
    ullAvailVirtual: 'uint64',
    ullAvailExtendedVirtual: 'uint64'
});

// SYSTEM_INFO structure
const SYSTEM_INFO = koffi.struct('SYSTEM_INFO', {
    wProcessorArchitecture: 'uint16',
    wReserved: 'uint16',
    dwPageSize: 'uint32',
    lpMinimumApplicationAddress: 'void*',
    lpMaximumApplicationAddress: 'void*',
    dwActiveProcessorMask: 'size_t',
    dwNumberOfProcessors: 'uint32',
    dwProcessorType: 'uint32',
    dwAllocationGranularity: 'uint32',
    wProcessorLevel: 'uint16',
    wProcessorRevision: 'uint16'
});

// RECT structure
const RECT = koffi.struct('RECT', {
    left: 'int32',
    top: 'int32',
    right: 'int32',
    bottom: 'int32'
});

// POINT structure
const POINT = koffi.struct('POINT', {
    x: 'int32',
    y: 'int32'
});

module.exports.structures = {
    FILETIME, SYSTEMTIME, PROCESS_INFORMATION, STARTUPINFO,
    SECURITY_ATTRIBUTES, MEMORYSTATUSEX, SYSTEM_INFO, RECT, POINT
};
```

---

## 4. Process Management APIs

### 4.1 Process Creation and Control

```javascript
// ============================================
// Process Management API Integration
// ============================================

class ProcessManager {
    constructor() {
        this.kernel32 = koffi.load('kernel32.dll');
        this.psapi = koffi.load('psapi.dll');
        this._defineFunctions();
    }
    
    _defineFunctions() {
        // Process Creation
        this.CreateProcess = this.kernel32.func(`
            BOOL CreateProcess(
                LPCWSTR lpApplicationName,
                LPWSTR lpCommandLine,
                LPSECURITY_ATTRIBUTES lpProcessAttributes,
                LPSECURITY_ATTRIBUTES lpThreadAttributes,
                BOOL bInheritHandles,
                DWORD dwCreationFlags,
                LPVOID lpEnvironment,
                LPCWSTR lpCurrentDirectory,
                LPSTARTUPINFO lpStartupInfo,
                LPPROCESS_INFORMATION lpProcessInformation
            )
        `);
        
        // Process Termination
        this.TerminateProcess = this.kernel32.func(`
            BOOL TerminateProcess(HANDLE hProcess, UINT uExitCode)
        `);
        
        // Process Handle Operations
        this.OpenProcess = this.kernel32.func(`
            HANDLE OpenProcess(DWORD dwDesiredAccess, BOOL bInheritHandle, DWORD dwProcessId)
        `);
        
        this.CloseHandle = this.kernel32.func(`
            BOOL CloseHandle(HANDLE hObject)
        `);
        
        this.GetCurrentProcess = this.kernel32.func(`
            HANDLE GetCurrentProcess()
        `);
        
        this.GetCurrentProcessId = this.kernel32.func(`
            DWORD GetCurrentProcessId()
        `);
        
        // Process Information
        this.GetExitCodeProcess = this.kernel32.func(`
            BOOL GetExitCodeProcess(HANDLE hProcess, LPDWORD lpExitCode)
        `);
        
        this.GetProcessTimes = this.kernel32.func(`
            BOOL GetProcessTimes(
                HANDLE hProcess,
                LPFILETIME lpCreationTime,
                LPFILETIME lpExitTime,
                LPFILETIME lpKernelTime,
                LPFILETIME lpUserTime
            )
        `);
        
        // Process Enumeration (PSAPI)
        this.EnumProcesses = this.psapi.func(`
            BOOL EnumProcesses(DWORD* lpidProcess, DWORD cb, DWORD* lpcbNeeded)
        `);
        
        this.EnumProcessModules = this.psapi.func(`
            BOOL EnumProcessModules(
                HANDLE hProcess,
                HMODULE* lphModule,
                DWORD cb,
                LPDWORD lpcbNeeded
            )
        `);
        
        this.GetModuleBaseName = this.psapi.func(`
            DWORD GetModuleBaseNameW(
                HANDLE hProcess,
                HMODULE hModule,
                LPWSTR lpBaseName,
                DWORD nSize
            )
        `);
        
        this.GetModuleFileNameEx = this.psapi.func(`
            DWORD GetModuleFileNameExW(
                HANDLE hProcess,
                HMODULE hModule,
                LPWSTR lpFilename,
                DWORD nSize
            )
        `);
        
        // Process Priority
        this.GetPriorityClass = this.kernel32.func(`
            DWORD GetPriorityClass(HANDLE hProcess)
        `);
        
        this.SetPriorityClass = this.kernel32.func(`
            BOOL SetPriorityClass(HANDLE hProcess, DWORD dwPriorityClass)
        `);
    }
}

// Process Priority Classes
const PriorityClass = {
    IDLE_PRIORITY_CLASS: 0x00000040,
    BELOW_NORMAL_PRIORITY_CLASS: 0x00004000,
    NORMAL_PRIORITY_CLASS: 0x00000020,
    ABOVE_NORMAL_PRIORITY_CLASS: 0x00008000,
    HIGH_PRIORITY_CLASS: 0x00000080,
    REALTIME_PRIORITY_CLASS: 0x00000100
};

// Process Creation Flags
const CreationFlags = {
    CREATE_NEW_CONSOLE: 0x00000010,
    CREATE_NEW_PROCESS_GROUP: 0x00000200,
    CREATE_NO_WINDOW: 0x08000000,
    CREATE_SUSPENDED: 0x00000004,
    CREATE_UNICODE_ENVIRONMENT: 0x00000400,
    DEBUG_PROCESS: 0x00000001,
    DETACHED_PROCESS: 0x00000008,
    INHERIT_PARENT_AFFINITY: 0x00010000
};

// Process Access Rights
const PROCESS_ACCESS = {
    TERMINATE: 0x0001,
    CREATE_THREAD: 0x0002,
    SET_SESSIONID: 0x0004,
    VM_OPERATION: 0x0008,
    VM_READ: 0x0010,
    VM_WRITE: 0x0020,
    DUP_HANDLE: 0x0040,
    CREATE_PROCESS: 0x0080,
    SET_QUOTA: 0x0100,
    SET_INFORMATION: 0x0200,
    QUERY_INFORMATION: 0x0400,
    SUSPEND_RESUME: 0x0800,
    QUERY_LIMITED_INFORMATION: 0x1000,
    SET_LIMITED_INFORMATION: 0x2000,
    ALL_ACCESS: 0x1F0FFF
};

module.exports = { ProcessManager, PriorityClass, CreationFlags, PROCESS_ACCESS };
```

---

## 5. Window Management APIs

### 5.1 Window Enumeration and Control

```javascript
// ============================================
// Window Management API Integration
// ============================================

class WindowManager {
    constructor() {
        this.user32 = koffi.load('user32.dll');
        this.kernel32 = koffi.load('kernel32.dll');
        this._defineFunctions();
        this._enumCallbacks = new Map();
    }
    
    _defineFunctions() {
        // Window Finding
        this.FindWindow = this.user32.func(`
            HWND FindWindowW(LPCWSTR lpClassName, LPCWSTR lpWindowName)
        `);
        
        this.FindWindowEx = this.user32.func(`
            HWND FindWindowExW(
                HWND hwndParent,
                HWND hwndChildAfter,
                LPCWSTR lpszClass,
                LPCWSTR lpszWindow
            )
        `);
        
        this.GetWindow = this.user32.func(`
            HWND GetWindow(HWND hWnd, UINT uCmd)
        `);
        
        this.GetDesktopWindow = this.user32.func(`
            HWND GetDesktopWindow()
        `);
        
        this.GetForegroundWindow = this.user32.func(`
            HWND GetForegroundWindow()
        `);
        
        this.GetWindowThreadProcessId = this.user32.func(`
            DWORD GetWindowThreadProcessId(HWND hWnd, LPDWORD lpdwProcessId)
        `);
        
        // Window Information
        this.GetWindowText = this.user32.func(`
            int GetWindowTextW(HWND hWnd, LPWSTR lpString, int nMaxCount)
        `);
        
        this.GetWindowTextLength = this.user32.func(`
            int GetWindowTextLengthW(HWND hWnd)
        `);
        
        this.GetClassName = this.user32.func(`
            int GetClassNameW(HWND hWnd, LPWSTR lpClassName, int nMaxCount)
        `);
        
        this.IsWindow = this.user32.func(`BOOL IsWindow(HWND hWnd)`);
        this.IsWindowVisible = this.user32.func(`BOOL IsWindowVisible(HWND hWnd)`);
        this.IsWindowEnabled = this.user32.func(`BOOL IsWindowEnabled(HWND hWnd)`);
        this.IsIconic = this.user32.func(`BOOL IsIconic(HWND hWnd)`);
        this.IsZoomed = this.user32.func(`BOOL IsZoomed(HWND hWnd)`);
        
        // Window Rectangle
        this.GetWindowRect = this.user32.func(`BOOL GetWindowRect(HWND hWnd, LPRECT lpRect)`);
        this.GetClientRect = this.user32.func(`BOOL GetClientRect(HWND hWnd, LPRECT lpRect)`);
        
        // Window Position and Size
        this.SetWindowPos = this.user32.func(`
            BOOL SetWindowPos(
                HWND hWnd,
                HWND hWndInsertAfter,
                int X,
                int Y,
                int cx,
                int cy,
                UINT uFlags
            )
        `);
        
        this.MoveWindow = this.user32.func(`
            BOOL MoveWindow(HWND hWnd, int X, int Y, int nWidth, int nHeight, BOOL bRepaint)
        `);
        
        // Window State
        this.ShowWindow = this.user32.func(`BOOL ShowWindow(HWND hWnd, int nCmdShow)`);
        this.ShowWindowAsync = this.user32.func(`BOOL ShowWindowAsync(HWND hWnd, int nCmdShow)`);
        this.SetForegroundWindow = this.user32.func(`BOOL SetForegroundWindow(HWND hWnd)`);
        this.SetActiveWindow = this.user32.func(`HWND SetActiveWindow(HWND hWnd)`);
        this.EnableWindow = this.user32.func(`BOOL EnableWindow(HWND hWnd, BOOL bEnable)`);
        this.CloseWindow = this.user32.func(`BOOL CloseWindow(HWND hWnd)`);
        this.DestroyWindow = this.user32.func(`BOOL DestroyWindow(HWND hWnd)`);
        
        // Window Messages
        this.SendMessage = this.user32.func(`
            LRESULT SendMessageW(HWND hWnd, UINT Msg, WPARAM wParam, LPARAM lParam)
        `);
        
        this.PostMessage = this.user32.func(`
            BOOL PostMessageW(HWND hWnd, UINT Msg, WPARAM wParam, LPARAM lParam)
        `);
        
        // Window Enumeration
        this.EnumWindows = this.user32.func(`BOOL EnumWindows(ENUMWINDOWPROC lpEnumFunc, LPARAM lParam)`);
        this.EnumChildWindows = this.user32.func(`
            BOOL EnumChildWindows(HWND hWndParent, ENUMWINDOWPROC lpEnumFunc, LPARAM lParam)
        `);
        
        // Input Simulation
        this.SetCursorPos = this.user32.func(`BOOL SetCursorPos(int X, int Y)`);
        this.GetCursorPos = this.user32.func(`BOOL GetCursorPos(LPPOINT lpPoint)`);
        this.mouse_event = this.user32.func(`
            void mouse_event(DWORD dwFlags, DWORD dx, DWORD dy, DWORD dwData, ULONG_PTR dwExtraInfo)
        `);
        this.keybd_event = this.user32.func(`
            void keybd_event(BYTE bVk, BYTE bScan, DWORD dwFlags, ULONG_PTR dwExtraInfo)
        `);
        this.MapVirtualKey = this.user32.func(`UINT MapVirtualKeyW(UINT uCode, UINT uMapType)`);
        this.VkKeyScan = this.user32.func(`SHORT VkKeyScanW(WCHAR ch)`);
    }
}

// ShowWindow Commands
const SW = {
    HIDE: 0,
    SHOWNORMAL: 1,
    SHOWMINIMIZED: 2,
    SHOWMAXIMIZED: 3,
    SHOWNOACTIVATE: 4,
    SHOW: 5,
    MINIMIZE: 6,
    SHOWMINNOACTIVE: 7,
    SHOWNA: 8,
    RESTORE: 9,
    SHOWDEFAULT: 10,
    FORCEMINIMIZE: 11
};

// SetWindowPos Flags
const SWP = {
    NOSIZE: 0x0001,
    NOMOVE: 0x0002,
    NOZORDER: 0x0004,
    NOREDRAW: 0x0008,
    NOACTIVATE: 0x0010,
    FRAMECHANGED: 0x0020,
    SHOWWINDOW: 0x0040,
    HIDEWINDOW: 0x0080,
    NOCOPYBITS: 0x0100,
    NOOWNERZORDER: 0x0200,
    NOSENDCHANGING: 0x0400,
    DEFERERASE: 0x2000,
    ASYNCWINDOWPOS: 0x4000
};

// Window Messages (Common)
const WM = {
    NULL: 0x0000,
    CREATE: 0x0001,
    DESTROY: 0x0002,
    MOVE: 0x0003,
    SIZE: 0x0005,
    ACTIVATE: 0x0006,
    SETFOCUS: 0x0007,
    KILLFOCUS: 0x0008,
    ENABLE: 0x000A,
    SETREDRAW: 0x000B,
    SETTEXT: 0x000C,
    GETTEXT: 0x000D,
    GETTEXTLENGTH: 0x000E,
    PAINT: 0x000F,
    CLOSE: 0x0010,
    QUIT: 0x0012,
    SHOWWINDOW: 0x0018,
    KEYDOWN: 0x0100,
    KEYUP: 0x0101,
    CHAR: 0x0102,
    SYSKEYDOWN: 0x0104,
    SYSKEYUP: 0x0105,
    MOUSEMOVE: 0x0200,
    LBUTTONDOWN: 0x0201,
    LBUTTONUP: 0x0202,
    LBUTTONDBLCLK: 0x0203,
    RBUTTONDOWN: 0x0204,
    RBUTTONUP: 0x0205,
    RBUTTONDBLCLK: 0x0206,
    MBUTTONDOWN: 0x0207,
    MBUTTONUP: 0x0208,
    MBUTTONDBLCLK: 0x0209,
    MOUSEWHEEL: 0x020A,
    CUT: 0x0300,
    COPY: 0x0301,
    PASTE: 0x0302,
    CLEAR: 0x0303,
    UNDO: 0x0304,
    USER: 0x0400
};

// Virtual Key Codes
const VK = {
    LBUTTON: 0x01,
    RBUTTON: 0x02,
    CANCEL: 0x03,
    MBUTTON: 0x04,
    XBUTTON1: 0x05,
    XBUTTON2: 0x06,
    BACK: 0x08,
    TAB: 0x09,
    CLEAR: 0x0C,
    RETURN: 0x0D,
    SHIFT: 0x10,
    CONTROL: 0x11,
    MENU: 0x12,
    PAUSE: 0x13,
    CAPITAL: 0x14,
    ESCAPE: 0x1B,
    SPACE: 0x20,
    PRIOR: 0x21,
    NEXT: 0x22,
    END: 0x23,
    HOME: 0x24,
    LEFT: 0x25,
    UP: 0x26,
    RIGHT: 0x27,
    DOWN: 0x28,
    SELECT: 0x29,
    PRINT: 0x2A,
    EXECUTE: 0x2B,
    SNAPSHOT: 0x2C,
    INSERT: 0x2D,
    DELETE: 0x2E,
    HELP: 0x2F,
    NUM0: 0x30, NUM1: 0x31, NUM2: 0x32, NUM3: 0x33, NUM4: 0x34,
    NUM5: 0x35, NUM6: 0x36, NUM7: 0x37, NUM8: 0x38, NUM9: 0x39,
    A: 0x41, B: 0x42, C: 0x43, D: 0x44, E: 0x45, F: 0x46, G: 0x47,
    H: 0x48, I: 0x49, J: 0x4A, K: 0x4B, L: 0x4C, M: 0x4D, N: 0x4E,
    O: 0x4F, P: 0x50, Q: 0x51, R: 0x52, S: 0x53, T: 0x54, U: 0x55,
    V: 0x56, W: 0x57, X: 0x58, Y: 0x59, Z: 0x5A,
    LWIN: 0x5B,
    RWIN: 0x5C,
    APPS: 0x5D,
    SLEEP: 0x5F,
    NUMPAD0: 0x60, NUMPAD1: 0x61, NUMPAD2: 0x62, NUMPAD3: 0x63,
    NUMPAD4: 0x64, NUMPAD5: 0x65, NUMPAD6: 0x66, NUMPAD7: 0x67,
    NUMPAD8: 0x68, NUMPAD9: 0x69,
    MULTIPLY: 0x6A,
    ADD: 0x6B,
    SEPARATOR: 0x6C,
    SUBTRACT: 0x6D,
    DECIMAL: 0x6E,
    DIVIDE: 0x6F,
    F1: 0x70, F2: 0x71, F3: 0x72, F4: 0x73, F5: 0x74, F6: 0x75,
    F7: 0x76, F8: 0x77, F9: 0x78, F10: 0x79, F11: 0x7A, F12: 0x7B,
    NUMLOCK: 0x90,
    SCROLL: 0x91,
    LSHIFT: 0xA0,
    RSHIFT: 0xA1,
    LCONTROL: 0xA2,
    RCONTROL: 0xA3,
    LMENU: 0xA4,
    RMENU: 0xA5
};

module.exports = { WindowManager, SW, SWP, WM, VK };
```

---

## 6. System Information APIs

### 6.1 System and Hardware Information

```javascript
// ============================================
// System Information API Integration
// ============================================

class SystemInfoManager {
    constructor() {
        this.kernel32 = koffi.load('kernel32.dll');
        this.pdh = koffi.load('pdh.dll');
        this.psapi = koffi.load('psapi.dll');
        this._defineFunctions();
    }
    
    _defineFunctions() {
        // System Information
        this.GetSystemInfo = this.kernel32.func(`void GetSystemInfo(LPSYSTEM_INFO lpSystemInfo)`);
        this.GetNativeSystemInfo = this.kernel32.func(`void GetNativeSystemInfo(LPSYSTEM_INFO lpSystemInfo)`);
        this.GetVersionEx = this.kernel32.func(`BOOL GetVersionExW(LPOSVERSIONINFOEX lpVersionInformation)`);
        this.GetTickCount = this.kernel32.func(`DWORD GetTickCount()`);
        this.GetTickCount64 = this.kernel32.func(`ULONGLONG GetTickCount64()`);
        
        // Memory Information
        this.GlobalMemoryStatusEx = this.kernel32.func(`BOOL GlobalMemoryStatusEx(LPMEMORYSTATUSEX lpBuffer)`);
        
        // Performance Information
        this.GetPerformanceInfo = this.psapi.func(`BOOL GetPerformanceInfo(PPERFORMANCE_INFORMATION pPerformanceInformation, DWORD cb)`);
        
        // Power Information
        this.GetSystemPowerStatus = this.kernel32.func(`BOOL GetSystemPowerStatus(LPSYSTEM_POWER_STATUS lpSystemPowerStatus)`);
        
        // Time Information
        this.GetSystemTime = this.kernel32.func(`void GetSystemTime(LPSYSTEMTIME lpSystemTime)`);
        this.GetLocalTime = this.kernel32.func(`void GetLocalTime(LPSYSTEMTIME lpSystemTime)`);
        this.GetSystemTimeAsFileTime = this.kernel32.func(`void GetSystemTimeAsFileTime(LPFILETIME lpSystemTimeAsFileTime)`);
        this.FileTimeToSystemTime = this.kernel32.func(`BOOL FileTimeToSystemTime(const FILETIME* lpFileTime, LPSYSTEMTIME lpSystemTime)`);
        this.SystemTimeToFileTime = this.kernel32.func(`BOOL SystemTimeToFileTime(const SYSTEMTIME* lpSystemTime, LPFILETIME lpFileTime)`);
        
        // Computer Name
        this.GetComputerName = this.kernel32.func(`BOOL GetComputerNameW(LPWSTR lpBuffer, LPDWORD nSize)`);
        this.GetComputerNameEx = this.kernel32.func(`BOOL GetComputerNameExW(COMPUTER_NAME_FORMAT NameType, LPWSTR lpBuffer, LPDWORD nSize)`);
        
        // User Name
        this.GetUserName = koffi.load('advapi32.dll').func(`BOOL GetUserNameW(LPWSTR lpBuffer, LPDWORD pcbBuffer)`);
        
        // Performance Counters (PDH)
        this.PdhOpenQuery = this.pdh.func(`PDH_STATUS PdhOpenQuery(LPCWSTR szDataSource, DWORD_PTR dwUserData, PDH_HQUERY* phQuery)`);
        this.PdhAddCounter = this.pdh.func(`PDH_STATUS PdhAddCounterW(PDH_HQUERY hQuery, LPCWSTR szFullCounterPath, DWORD_PTR dwUserData, PDH_HCOUNTER* phCounter)`);
        this.PdhCollectQueryData = this.pdh.func(`PDH_STATUS PdhCollectQueryData(PDH_HQUERY hQuery)`);
        this.PdhGetFormattedCounterValue = this.pdh.func(`PDH_STATUS PdhGetFormattedCounterValue(PDH_HCOUNTER hCounter, DWORD dwFormat, LPDWORD lpdwType, PPDH_FMT_COUNTERVALUE pValue)`);
        this.PdhCloseQuery = this.pdh.func(`PDH_STATUS PdhCloseQuery(PDH_HQUERY hQuery)`);
        
        // Environment
        this.GetEnvironmentVariable = this.kernel32.func(`DWORD GetEnvironmentVariableW(LPCWSTR lpName, LPWSTR lpBuffer, DWORD nSize)`);
        this.SetEnvironmentVariable = this.kernel32.func(`BOOL SetEnvironmentVariableW(LPCWSTR lpName, LPCWSTR lpValue)`);
        this.GetEnvironmentStrings = this.kernel32.func(`LPWCH GetEnvironmentStringsW()`);
        this.FreeEnvironmentStrings = this.kernel32.func(`BOOL FreeEnvironmentStringsW(LPWCH lpszEnvironmentBlock)`);
    }
}

// Processor Architectures
const PROCESSOR_ARCHITECTURE = {
    INTEL: 0,
    MIPS: 1,
    ALPHA: 2,
    PPC: 3,
    SHX: 4,
    ARM: 5,
    IA64: 6,
    ALPHA64: 7,
    MSIL: 8,
    AMD64: 9,
    IA32_ON_WIN64: 10,
    NEUTRAL: 11,
    ARM64: 12,
    ARM32_ON_WIN64: 13,
    IA32_ON_ARM64: 14,
    UNKNOWN: 0xFFFF
};

// Computer Name Formats
const COMPUTER_NAME_FORMAT = {
    NetBIOS: 0,
    DnsHostname: 1,
    DnsDomain: 2,
    DnsFullyQualified: 3,
    PhysicalNetBIOS: 4,
    PhysicalDnsHostname: 5,
    PhysicalDnsDomain: 6,
    PhysicalDnsFullyQualified: 7,
    Max: 8
};

module.exports = { SystemInfoManager, PROCESSOR_ARCHITECTURE, COMPUTER_NAME_FORMAT };
```

---

## 7. File Operations APIs

### 7.1 Advanced File Operations

```javascript
// ============================================
// File Operations API Integration
// ============================================

class FileOperationsManager {
    constructor() {
        this.kernel32 = koffi.load('kernel32.dll');
        this.shell32 = koffi.load('shell32.dll');
        this.shlwapi = koffi.load('shlwapi.dll');
        this._defineFunctions();
    }
    
    _defineFunctions() {
        // File Creation and Opening
        this.CreateFile = this.kernel32.func(`
            HANDLE CreateFileW(
                LPCWSTR lpFileName,
                DWORD dwDesiredAccess,
                DWORD dwShareMode,
                LPSECURITY_ATTRIBUTES lpSecurityAttributes,
                DWORD dwCreationDisposition,
                DWORD dwFlagsAndAttributes,
                HANDLE hTemplateFile
            )
        `);
        
        this.CloseHandle = this.kernel32.func(`BOOL CloseHandle(HANDLE hObject)`);
        
        // File Reading/Writing
        this.ReadFile = this.kernel32.func(`
            BOOL ReadFile(
                HANDLE hFile,
                LPVOID lpBuffer,
                DWORD nNumberOfBytesToRead,
                LPDWORD lpNumberOfBytesRead,
                LPOVERLAPPED lpOverlapped
            )
        `);
        
        this.WriteFile = this.kernel32.func(`
            BOOL WriteFile(
                HANDLE hFile,
                LPCVOID lpBuffer,
                DWORD nNumberOfBytesToWrite,
                LPDWORD lpNumberOfBytesWritten,
                LPOVERLAPPED lpOverlapped
            )
        `);
        
        // File Positioning
        this.SetFilePointer = this.kernel32.func(`
            DWORD SetFilePointer(HANDLE hFile, LONG lDistanceToMove, PLONG lpDistanceToMoveHigh, DWORD dwMoveMethod)
        `);
        
        this.SetFilePointerEx = this.kernel32.func(`
            BOOL SetFilePointerEx(HANDLE hFile, LARGE_INTEGER liDistanceToMove, PLARGE_INTEGER lpNewFilePointer, DWORD dwMoveMethod)
        `);
        
        // File Information
        this.GetFileSize = this.kernel32.func(`DWORD GetFileSize(HANDLE hFile, LPDWORD lpFileSizeHigh)`);
        this.GetFileSizeEx = this.kernel32.func(`BOOL GetFileSizeEx(HANDLE hFile, PLARGE_INTEGER lpFileSize)`);
        this.GetFileTime = this.kernel32.func(`BOOL GetFileTime(HANDLE hFile, LPFILETIME lpCreationTime, LPFILETIME lpLastAccessTime, LPFILETIME lpLastWriteTime)`);
        this.SetFileTime = this.kernel32.func(`BOOL SetFileTime(HANDLE hFile, const FILETIME* lpCreationTime, const FILETIME* lpLastAccessTime, const FILETIME* lpLastWriteTime)`);
        this.GetFileAttributes = this.kernel32.func(`DWORD GetFileAttributesW(LPCWSTR lpFileName)`);
        this.SetFileAttributes = this.kernel32.func(`BOOL SetFileAttributesW(LPCWSTR lpFileName, DWORD dwFileAttributes)`);
        
        // File Operations
        this.CopyFile = this.kernel32.func(`BOOL CopyFileW(LPCWSTR lpExistingFileName, LPCWSTR lpNewFileName, BOOL bFailIfExists)`);
        this.CopyFileEx = this.kernel32.func(`BOOL CopyFileExW(LPCWSTR lpExistingFileName, LPCWSTR lpNewFileName, LPPROGRESS_ROUTINE lpProgressRoutine, LPVOID lpData, LPBOOL pbCancel, DWORD dwCopyFlags)`);
        this.MoveFile = this.kernel32.func(`BOOL MoveFileW(LPCWSTR lpExistingFileName, LPCWSTR lpNewFileName)`);
        this.MoveFileEx = this.kernel32.func(`BOOL MoveFileExW(LPCWSTR lpExistingFileName, LPCWSTR lpNewFileName, DWORD dwFlags)`);
        this.DeleteFile = this.kernel32.func(`BOOL DeleteFileW(LPCWSTR lpFileName)`);
        this.ReplaceFile = this.kernel32.func(`BOOL ReplaceFileW(LPCWSTR lpReplacedFileName, LPCWSTR lpReplacementFileName, LPCWSTR lpBackupFileName, DWORD dwReplaceFlags, LPVOID lpExclude, LPVOID lpReserved)`);
        
        // Directory Operations
        this.CreateDirectory = this.kernel32.func(`BOOL CreateDirectoryW(LPCWSTR lpPathName, LPSECURITY_ATTRIBUTES lpSecurityAttributes)`);
        this.RemoveDirectory = this.kernel32.func(`BOOL RemoveDirectoryW(LPCWSTR lpPathName)`);
        this.SetCurrentDirectory = this.kernel32.func(`BOOL SetCurrentDirectoryW(LPCWSTR lpPathName)`);
        this.GetCurrentDirectory = this.kernel32.func(`DWORD GetCurrentDirectoryW(DWORD nBufferLength, LPWSTR lpBuffer)`);
        
        // File Enumeration
        this.FindFirstFile = this.kernel32.func(`HANDLE FindFirstFileW(LPCWSTR lpFileName, LPWIN32_FIND_DATA lpFindFileData)`);
        this.FindFirstFileEx = this.kernel32.func(`HANDLE FindFirstFileExW(LPCWSTR lpFileName, FINDEX_INFO_LEVELS fInfoLevelId, LPVOID lpFindFileData, FINDEX_SEARCH_OPS fSearchOp, LPVOID lpSearchFilter, DWORD dwAdditionalFlags)`);
        this.FindNextFile = this.kernel32.func(`BOOL FindNextFileW(HANDLE hFindFile, LPWIN32_FIND_DATA lpFindFileData)`);
        this.FindClose = this.kernel32.func(`BOOL FindClose(HANDLE hFindFile)`);
        
        // File Locking
        this.LockFile = this.kernel32.func(`BOOL LockFile(HANDLE hFile, DWORD dwFileOffsetLow, DWORD dwFileOffsetHigh, DWORD nNumberOfBytesToLockLow, DWORD nNumberOfBytesToLockHigh)`);
        this.UnlockFile = this.kernel32.func(`BOOL UnlockFile(HANDLE hFile, DWORD dwFileOffsetLow, DWORD dwFileOffsetHigh, DWORD nNumberOfBytesToUnlockLow, DWORD nNumberOfBytesToUnlockHigh)`);
        
        // File Mapping
        this.CreateFileMapping = this.kernel32.func(`HANDLE CreateFileMappingW(HANDLE hFile, LPSECURITY_ATTRIBUTES lpFileMappingAttributes, DWORD flProtect, DWORD dwMaximumSizeHigh, DWORD dwMaximumSizeLow, LPCWSTR lpName)`);
        this.OpenFileMapping = this.kernel32.func(`HANDLE OpenFileMappingW(DWORD dwDesiredAccess, BOOL bInheritHandle, LPCWSTR lpName)`);
        this.MapViewOfFile = this.kernel32.func(`LPVOID MapViewOfFile(HANDLE hFileMappingObject, DWORD dwDesiredAccess, DWORD dwFileOffsetHigh, DWORD dwFileOffsetLow, SIZE_T dwNumberOfBytesToMap)`);
        this.UnmapViewOfFile = this.kernel32.func(`BOOL UnmapViewOfFile(LPCVOID lpBaseAddress)`);
        this.FlushViewOfFile = this.kernel32.func(`BOOL FlushViewOfFile(LPCVOID lpBaseAddress, SIZE_T dwNumberOfBytesToFlush)`);
        
        // Path Operations (shlwapi)
        this.PathFileExists = this.shlwapi.func(`BOOL PathFileExistsW(LPCWSTR pszPath)`);
        this.PathIsDirectory = this.shlwapi.func(`BOOL PathIsDirectoryW(LPCWSTR pszPath)`);
        this.PathIsFileSpec = this.shlwapi.func(`BOOL PathIsFileSpecW(LPCWSTR pszPath)`);
        
        // Shell Operations
        this.SHCreateDirectoryEx = this.shell32.func(`int SHCreateDirectoryExW(HWND hwnd, LPCWSTR pszPath, const SECURITY_ATTRIBUTES* psa)`);
        this.SHFileOperation = this.shell32.func(`int SHFileOperationW(LPSHFILEOPSTRUCT lpFileOp)`);
    }
}

// File Access Rights
const FILE_ACCESS = {
    GENERIC_READ: 0x80000000,
    GENERIC_WRITE: 0x40000000,
    GENERIC_EXECUTE: 0x20000000,
    GENERIC_ALL: 0x10000000
};

// File Share Modes
const FILE_SHARE = {
    NONE: 0,
    READ: 0x1,
    WRITE: 0x2,
    DELETE: 0x4
};

// File Creation Dispositions
const FILE_CREATION = {
    CREATE_NEW: 1,
    CREATE_ALWAYS: 2,
    OPEN_EXISTING: 3,
    OPEN_ALWAYS: 4,
    TRUNCATE_EXISTING: 5
};

// File Attributes
const FILE_ATTRIBUTE = {
    READONLY: 0x1,
    HIDDEN: 0x2,
    SYSTEM: 0x4,
    DIRECTORY: 0x10,
    ARCHIVE: 0x20,
    DEVICE: 0x40,
    NORMAL: 0x80,
    TEMPORARY: 0x100,
    SPARSE_FILE: 0x200,
    REPARSE_POINT: 0x400,
    COMPRESSED: 0x800,
    OFFLINE: 0x1000,
    NOT_CONTENT_INDEXED: 0x2000,
    ENCRYPTED: 0x4000
};

module.exports = { FileOperationsManager, FILE_ACCESS, FILE_SHARE, FILE_CREATION, FILE_ATTRIBUTE };
```

---

## 8. Registry Access APIs

### 8.1 Registry Operations

```javascript
// ============================================
// Registry Access API Integration
// ============================================

class RegistryManager {
    constructor() {
        this.advapi32 = koffi.load('advapi32.dll');
        this._defineFunctions();
    }
    
    _defineFunctions() {
        // Registry Key Operations
        this.RegOpenKeyEx = this.advapi32.func(`
            LSTATUS RegOpenKeyExW(HKEY hKey, LPCWSTR lpSubKey, DWORD ulOptions, REGSAM samDesired, PHKEY phkResult)
        `);
        
        this.RegCreateKeyEx = this.advapi32.func(`
            LSTATUS RegCreateKeyExW(HKEY hKey, LPCWSTR lpSubKey, DWORD Reserved, LPWSTR lpClass, DWORD dwOptions, REGSAM samDesired, const LPSECURITY_ATTRIBUTES lpSecurityAttributes, PHKEY phkResult, LPDWORD lpdwDisposition)
        `);
        
        this.RegCloseKey = this.advapi32.func(`LSTATUS RegCloseKey(HKEY hKey)`);
        this.RegDeleteKey = this.advapi32.func(`LSTATUS RegDeleteKeyW(HKEY hKey, LPCWSTR lpSubKey)`);
        this.RegDeleteKeyEx = this.advapi32.func(`LSTATUS RegDeleteKeyExW(HKEY hKey, LPCWSTR lpSubKey, REGSAM samDesired, DWORD Reserved)`);
        this.RegDeleteTree = this.advapi32.func(`LSTATUS RegDeleteTreeW(HKEY hKey, LPCWSTR lpSubKey)`);
        
        // Registry Value Operations
        this.RegQueryValueEx = this.advapi32.func(`
            LSTATUS RegQueryValueExW(HKEY hKey, LPCWSTR lpValueName, LPDWORD lpReserved, LPDWORD lpType, LPBYTE lpData, LPDWORD lpcbData)
        `);
        
        this.RegSetValueEx = this.advapi32.func(`
            LSTATUS RegSetValueExW(HKEY hKey, LPCWSTR lpValueName, DWORD Reserved, DWORD dwType, const BYTE* lpData, DWORD cbData)
        `);
        
        this.RegDeleteValue = this.advapi32.func(`LSTATUS RegDeleteValueW(HKEY hKey, LPCWSTR lpValueName)`);
        
        // Registry Enumeration
        this.RegEnumKeyEx = this.advapi32.func(`
            LSTATUS RegEnumKeyExW(HKEY hKey, DWORD dwIndex, LPWSTR lpName, LPDWORD lpcchName, LPDWORD lpReserved, LPWSTR lpClass, LPDWORD lpcchClass, PFILETIME lpftLastWriteTime)
        `);
        
        this.RegEnumValue = this.advapi32.func(`
            LSTATUS RegEnumValueW(HKEY hKey, DWORD dwIndex, LPWSTR lpValueName, LPDWORD lpcchValueName, LPDWORD lpReserved, LPDWORD lpType, LPBYTE lpData, LPDWORD lpcbData)
        `);
        
        this.RegQueryInfoKey = this.advapi32.func(`
            LSTATUS RegQueryInfoKeyW(HKEY hKey, LPWSTR lpClass, LPDWORD lpcchClass, LPDWORD lpReserved, LPDWORD lpcSubKeys, LPDWORD lpcbMaxSubKeyLen, LPDWORD lpcbMaxClassLen, LPDWORD lpcValues, LPDWORD lpcbMaxValueNameLen, LPDWORD lpcbMaxValueLen, LPDWORD lpcbSecurityDescriptor, PFILETIME lpftLastWriteTime)
        `);
        
        // Registry Security
        this.RegGetKeySecurity = this.advapi32.func(`
            LSTATUS RegGetKeySecurity(HKEY hKey, SECURITY_INFORMATION SecurityInformation, PSECURITY_DESCRIPTOR pSecurityDescriptor, LPDWORD lpcbSecurityDescriptor)
        `);
        
        this.RegSetKeySecurity = this.advapi32.func(`
            LSTATUS RegSetKeySecurity(HKEY hKey, SECURITY_INFORMATION SecurityInformation, PSECURITY_DESCRIPTOR pSecurityDescriptor)
        `);
        
        // Registry Notifications
        this.RegNotifyChangeKeyValue = this.advapi32.func(`
            LSTATUS RegNotifyChangeKeyValue(HKEY hKey, BOOL bWatchSubtree, DWORD dwNotifyFilter, HANDLE hEvent, BOOL fAsynchronous)
        `);
        
        // Registry Save/Load
        this.RegSaveKeyEx = this.advapi32.func(`
            LSTATUS RegSaveKeyExW(HKEY hKey, LPCWSTR lpFile, const LPSECURITY_ATTRIBUTES lpSecurityAttributes, DWORD Flags)
        `);
        
        this.RegLoadKey = this.advapi32.func(`LSTATUS RegLoadKeyW(HKEY hKey, LPCWSTR lpSubKey, LPCWSTR lpFile)`);
        this.RegUnLoadKey = this.advapi32.func(`LSTATUS RegUnLoadKeyW(HKEY hKey, LPCWSTR lpSubKey)`);
        
        // Registry Connection
        this.RegConnectRegistry = this.advapi32.func(`LSTATUS RegConnectRegistryW(LPCWSTR lpMachineName, HKEY hKey, PHKEY phkResult)`);
    }
}

// Predefined Registry Keys
const HKEY = {
    CLASSES_ROOT: 0x80000000n,
    CURRENT_USER: 0x80000001n,
    LOCAL_MACHINE: 0x80000002n,
    USERS: 0x80000003n,
    PERFORMANCE_DATA: 0x80000004n,
    CURRENT_CONFIG: 0x80000005n,
    DYN_DATA: 0x80000006n
};

// Registry Access Rights
const KEY = {
    QUERY_VALUE: 0x1,
    SET_VALUE: 0x2,
    CREATE_SUB_KEY: 0x4,
    ENUMERATE_SUB_KEYS: 0x8,
    NOTIFY: 0x10,
    CREATE_LINK: 0x20,
    WOW64_32KEY: 0x200,
    WOW64_64KEY: 0x100,
    WOW64_RES: 0x300,
    READ: 0x20019,
    WRITE: 0x20006,
    EXECUTE: 0x20019,
    ALL_ACCESS: 0xF003F
};

// Registry Value Types
const REG_TYPE = {
    NONE: 0,
    SZ: 1,
    EXPAND_SZ: 2,
    BINARY: 3,
    DWORD: 4,
    DWORD_BIG_ENDIAN: 5,
    LINK: 6,
    MULTI_SZ: 7,
    RESOURCE_LIST: 8,
    FULL_RESOURCE_DESCRIPTOR: 9,
    RESOURCE_REQUIREMENTS_LIST: 10,
    QWORD: 11
};

module.exports = { RegistryManager, HKEY, KEY, REG_TYPE };
```

---

## 9. Security and Privilege APIs

### 9.1 Security Operations

```javascript
// ============================================
// Security and Privilege API Integration
// ============================================

class SecurityManager {
    constructor() {
        this.advapi32 = koffi.load('advapi32.dll');
        this.kernel32 = koffi.load('kernel32.dll');
        this._defineFunctions();
    }
    
    _defineFunctions() {
        // Token Operations
        this.OpenProcessToken = this.advapi32.func(`BOOL OpenProcessToken(HANDLE ProcessHandle, DWORD DesiredAccess, PHANDLE TokenHandle)`);
        this.OpenThreadToken = this.advapi32.func(`BOOL OpenThreadToken(HANDLE ThreadHandle, DWORD DesiredAccess, BOOL OpenAsSelf, PHANDLE TokenHandle)`);
        this.GetTokenInformation = this.advapi32.func(`BOOL GetTokenInformation(HANDLE TokenHandle, TOKEN_INFORMATION_CLASS TokenInformationClass, LPVOID TokenInformation, DWORD TokenInformationLength, PDWORD ReturnLength)`);
        this.SetTokenInformation = this.advapi32.func(`BOOL SetTokenInformation(HANDLE TokenHandle, TOKEN_INFORMATION_CLASS TokenInformationClass, LPVOID TokenInformation, DWORD TokenInformationLength)`);
        this.DuplicateToken = this.advapi32.func(`BOOL DuplicateToken(HANDLE ExistingTokenHandle, SECURITY_IMPERSONATION_LEVEL ImpersonationLevel, PHANDLE DuplicateTokenHandle)`);
        this.DuplicateTokenEx = this.advapi32.func(`BOOL DuplicateTokenEx(HANDLE hExistingToken, DWORD dwDesiredAccess, LPSECURITY_ATTRIBUTES lpTokenAttributes, SECURITY_IMPERSONATION_LEVEL ImpersonationLevel, TOKEN_TYPE TokenType, PHANDLE phNewToken)`);
        this.CloseHandle = this.kernel32.func(`BOOL CloseHandle(HANDLE hObject)`);
        
        // Privilege Operations
        this.LookupPrivilegeValue = this.advapi32.func(`BOOL LookupPrivilegeValueW(LPCWSTR lpSystemName, LPCWSTR lpName, PLUID lpLuid)`);
        this.LookupPrivilegeName = this.advapi32.func(`BOOL LookupPrivilegeNameW(LPCWSTR lpSystemName, PLUID lpLuid, LPWSTR lpName, LPDWORD cchName)`);
        this.AdjustTokenPrivileges = this.advapi32.func(`BOOL AdjustTokenPrivileges(HANDLE TokenHandle, BOOL DisableAllPrivileges, PTOKEN_PRIVILEGES NewState, DWORD BufferLength, PTOKEN_PRIVILEGES PreviousState, PDWORD ReturnLength)`);
        
        // Security Descriptor Operations
        this.InitializeSecurityDescriptor = this.advapi32.func(`BOOL InitializeSecurityDescriptor(PSECURITY_DESCRIPTOR pSecurityDescriptor, DWORD dwRevision)`);
        this.GetSecurityDescriptorDacl = this.advapi32.func(`BOOL GetSecurityDescriptorDacl(PSECURITY_DESCRIPTOR pSecurityDescriptor, LPBOOL lpbDaclPresent, PACL* pDacl, LPBOOL lpbDaclDefaulted)`);
        this.SetSecurityDescriptorDacl = this.advapi32.func(`BOOL SetSecurityDescriptorDacl(PSECURITY_DESCRIPTOR pSecurityDescriptor, BOOL bDaclPresent, PACL pDacl, BOOL bDaclDefaulted)`);
        
        // ACL Operations
        this.InitializeAcl = this.advapi32.func(`BOOL InitializeAcl(PACL pAcl, DWORD nAclLength, DWORD dwAclRevision)`);
        this.AddAccessAllowedAce = this.advapi32.func(`BOOL AddAccessAllowedAce(PACL pAcl, DWORD dwAceRevision, DWORD AccessMask, PSID pSid)`);
        this.AddAccessDeniedAce = this.advapi32.func(`BOOL AddAccessDeniedAce(PACL pAcl, DWORD dwAceRevision, DWORD AccessMask, PSID pSid)`);
        
        // SID Operations
        this.AllocateAndInitializeSid = this.advapi32.func(`BOOL AllocateAndInitializeSid(PSID_IDENTIFIER_AUTHORITY pIdentifierAuthority, BYTE nSubAuthorityCount, DWORD nSubAuthority0, DWORD nSubAuthority1, DWORD nSubAuthority2, DWORD nSubAuthority3, DWORD nSubAuthority4, DWORD nSubAuthority5, DWORD nSubAuthority6, DWORD nSubAuthority7, PSID* pSid)`);
        this.FreeSid = this.advapi32.func(`PVOID FreeSid(PSID pSid)`);
        this.EqualSid = this.advapi32.func(`BOOL EqualSid(PSID pSid1, PSID pSid2)`);
        this.GetLengthSid = this.advapi32.func(`DWORD GetLengthSid(PSID pSid)`);
        this.CopySid = this.advapi32.func(`BOOL CopySid(DWORD nDestinationSidLength, PSID pDestinationSid, PSID pSourceSid)`);
        
        // User/Group Lookup
        this.LookupAccountName = this.advapi32.func(`BOOL LookupAccountNameW(LPCWSTR lpSystemName, LPCWSTR lpAccountName, PSID Sid, LPDWORD cbSid, LPWSTR ReferencedDomainName, LPDWORD cchReferencedDomainName, PSID_NAME_USE peUse)`);
        this.LookupAccountSid = this.advapi32.func(`BOOL LookupAccountSidW(LPCWSTR lpSystemName, PSID Sid, LPWSTR Name, LPDWORD cchName, LPWSTR ReferencedDomainName, LPDWORD cchReferencedDomainName, PSID_NAME_USE peUse)`);
    }
}

// Token Access Rights
const TOKEN_ACCESS = {
    ASSIGN_PRIMARY: 0x1,
    DUPLICATE: 0x2,
    IMPERSONATE: 0x4,
    QUERY: 0x8,
    QUERY_SOURCE: 0x10,
    ADJUST_PRIVILEGES: 0x20,
    ADJUST_GROUPS: 0x40,
    ADJUST_DEFAULT: 0x80,
    ADJUST_SESSIONID: 0x100,
    READ: 0x20008,
    WRITE: 0x200E0,
    EXECUTE: 0x20008,
    ALL_ACCESS: 0xF01FF
};

// Privilege Constants
const SE_PRIVILEGE = {
    CREATE_TOKEN: 'SeCreateTokenPrivilege',
    ASSIGNPRIMARYTOKEN: 'SeAssignPrimaryTokenPrivilege',
    LOCK_MEMORY: 'SeLockMemoryPrivilege',
    INCREASE_QUOTA: 'SeIncreaseQuotaPrivilege',
    MACHINE_ACCOUNT: 'SeMachineAccountPrivilege',
    TCB: 'SeTcbPrivilege',
    SECURITY: 'SeSecurityPrivilege',
    TAKE_OWNERSHIP: 'SeTakeOwnershipPrivilege',
    LOAD_DRIVER: 'SeLoadDriverPrivilege',
    SYSTEM_PROFILE: 'SeSystemProfilePrivilege',
    SYSTEMTIME: 'SeSystemtimePrivilege',
    PROFILE_SINGLE_PROCESS: 'SeProfileSingleProcessPrivilege',
    INCREASE_BASE_PRIORITY: 'SeIncreaseBasePriorityPrivilege',
    CREATE_PAGEFILE: 'SeCreatePagefilePrivilege',
    CREATE_PERMANENT: 'SeCreatePermanentPrivilege',
    BACKUP: 'SeBackupPrivilege',
    RESTORE: 'SeRestorePrivilege',
    SHUTDOWN: 'SeShutdownPrivilege',
    DEBUG: 'SeDebugPrivilege',
    AUDIT: 'SeAuditPrivilege',
    SYSTEM_ENVIRONMENT: 'SeSystemEnvironmentPrivilege',
    CHANGE_NOTIFY: 'SeChangeNotifyPrivilege',
    REMOTE_SHUTDOWN: 'SeRemoteShutdownPrivilege',
    UNDOCK: 'SeUndockPrivilege',
    SYNC_AGENT: 'SeSyncAgentPrivilege',
    ENABLE_DELEGATION: 'SeEnableDelegationPrivilege',
    MANAGE_VOLUME: 'SeManageVolumePrivilege',
    IMPERSONATE: 'SeImpersonatePrivilege',
    CREATE_GLOBAL: 'SeCreateGlobalPrivilege',
    TRUSTED_CREDMAN_ACCESS: 'SeTrustedCredManAccessPrivilege',
    RELABEL: 'SeRelabelPrivilege',
    INCREASE_WORKING_SET: 'SeIncreaseWorkingSetPrivilege',
    TIME_ZONE: 'SeTimeZonePrivilege',
    CREATE_SYMBOLIC_LINK: 'SeCreateSymbolicLinkPrivilege',
    DELEGATE_SESSION_USER_IMPERSONATE: 'SeDelegateSessionUserImpersonatePrivilege'
};

// Security Information Flags
const SECURITY_INFORMATION = {
    OWNER: 0x1,
    GROUP: 0x2,
    DACL: 0x4,
    SACL: 0x8,
    LABEL: 0x10,
    ATTRIBUTE: 0x20,
    SCOPE: 0x40,
    PROCESS_TRUST_LABEL: 0x80,
    ACCESS_FILTER: 0x100,
    BACKUP: 0x10000,
    PROTECTED_DACL: 0x80000000,
    PROTECTED_SACL: 0x40000000,
    UNPROTECTED_DACL: 0x20000000,
    UNPROTECTED_SACL: 0x10000000
};

module.exports = { SecurityManager, TOKEN_ACCESS, SE_PRIVILEGE, SECURITY_INFORMATION };
```

---

## Summary

This specification provides a comprehensive Win32 API integration layer for a Windows 10 AI Agent system. The key components include:

### Core DLLs Required:
- **kernel32.dll** - Base OS functions, process/thread, memory, file I/O
- **user32.dll** - Window management, input simulation, messages
- **shell32.dll** - Shell operations, file operations
- **advapi32.dll** - Registry, security, privileges
- **psapi.dll** - Process information, memory counters
- **pdh.dll** - Performance counters
- **shlwapi.dll** - Path utilities

### FFI Integration:
- **Koffi** (recommended) - Modern, fast FFI library
- Alternative: node-ffi-napi for legacy compatibility
- C++ Native Addons for maximum performance

### Key API Categories:
1. **Process Management** - CreateProcess, OpenProcess, EnumProcesses, TerminateProcess
2. **Window Management** - FindWindow, SendMessage, SetWindowPos, Input Simulation
3. **System Information** - GetSystemInfo, GlobalMemoryStatusEx, Performance Counters
4. **File Operations** - CreateFile, ReadFile, WriteFile, Directory Operations
5. **Registry Access** - RegOpenKeyEx, RegQueryValueEx, RegSetValueEx
6. **Security APIs** - Token Operations, Privileges, ACLs, SIDs

### Generated Files:
- `/mnt/okcomputer/output/win32_api_integration_spec.md`
