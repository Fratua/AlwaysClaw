# COM Objects and Windows Runtime (WinRT) Integration Specification
## OpenClaw Windows 10 AI Agent System

---

## Executive Summary

This document provides comprehensive technical specifications for integrating COM objects and Windows Runtime (WinRT) APIs into the Windows 10 version of the OpenClaw AI agent system. The architecture enables full system automation, TTS/STT capabilities, UWP integration, and 24/7 background operation through a hybrid COM/WinRT approach.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [COM Interop from Node.js](#2-com-interop-from-nodejs)
3. [Common COM Objects for System Automation](#3-common-com-objects-for-system-automation)
4. [Windows Runtime (WinRT) Activation](#4-windows-runtime-winrt-activation)
5. [UWP API Access from Desktop Agent](#5-uwp-api-access-from-desktop-agent)
6. [Windows.Media.Speech for TTS/STT](#6-windowsmediaspeech-for-ttsstt)
7. [Windows.System and Windows.Storage APIs](#7-windowssystem-and-windowsstorage-apis)
8. [COM Security and Marshaling](#8-com-security-and-marshaling)
9. [Async Pattern Handling for WinRT](#9-async-pattern-handling-for-winrt)
10. [Implementation Examples](#10-implementation-examples)

---

## 1. Architecture Overview

### 1.1 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OpenClaw AI Agent (Node.js)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   Agent Core    │  │  Agentic Loops  │  │   Soul/Identity │             │
│  │   (GPT-5.2)     │  │   (15 Hardcoded)│  │    System       │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
│           │                    │                    │                       │
│  ┌────────▼────────────────────▼────────────────────▼────────┐             │
│  │              Windows Integration Layer                      │             │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │             │
│  │  │  COM Interop │  │  WinRT Bridge│  │  Node-API    │     │             │
│  │  │   (edge.js)  │  │   (NodeRT)   │  │   Addons     │     │             │
│  │  └──────────────┘  └──────────────┘  └──────────────┘     │             │
│  └────────────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
┌───────▼────────┐          ┌─────────▼──────────┐      ┌───────────▼────────┐
│   COM Objects  │          │   WinRT APIs       │      │  Windows APIs      │
│                │          │                    │      │                    │
│ • Shell.Application    │  │ • Windows.Media.Speech    │  │ • Win32 APIs       │
│ • WScript.Shell        │  │ • Windows.System          │  │ • File Operations  │
│ • FileSystemObject     │  │ • Windows.Storage         │  │ • Registry         │
│ • WMI Objects          │  │ • Windows.Networking      │  │ • Services         │
│ • Speech API (SAPI)    │  │ • Windows.Data            │  │ • Processes        │
└────────────────┘          └────────────────────┘      └────────────────────┘
```

### 1.2 Integration Patterns

| Pattern | Technology | Use Case | Performance |
|---------|------------|----------|-------------|
| Direct COM | node-win32com, ffi-napi | Legacy automation, WMI | Medium |
| .NET Bridge | edge.js | Complex COM interop | High |
| WinRT Native | NodeRT | Modern Windows APIs | High |
| Native Addon | N-API | Performance-critical | Highest |
| PowerShell | child_process | Administrative tasks | Medium |

---

## 2. COM Interop from Node.js

### 2.1 Recommended Libraries

#### 2.1.1 Primary: edge.js (Recommended)

```javascript
// edge.js - .NET/Node.js interop
const edge = require('edge-js');

// Create COM object through .NET bridge
const createComObject = edge.func({
    source: `
        using System;
        using System.Threading.Tasks;
        using System.Runtime.InteropServices;
        
        public class ComBridge {
            public async Task<object> Invoke(dynamic input) {
                Type comType = Type.GetTypeFromProgID(input.progId);
                dynamic comObject = Activator.CreateInstance(comType);
                return comObject;
            }
        }
    `,
    references: ['System.Runtime.InteropServices.dll']
});
```

**Installation:**
```bash
npm install edge-js
```

**Requirements:**
- .NET Framework 4.5+ or .NET Core 3.1+
- Windows 10 SDK
- Visual Studio Build Tools

#### 2.1.2 Alternative: node-win32com

```javascript
const win32com = require('node-win32com');

// Create COM object
const shell = win32com.createObject('Shell.Application');

// Invoke methods
shell.Explore('C:\\Windows');
```

#### 2.1.3 Alternative: ffi-napi + ref-napi

```javascript
const ffi = require('ffi-napi');
const ref = require('ref-napi');

// Load ole32.dll for COM functions
const ole32 = ffi.Library('ole32', {
    'CoInitializeEx': ['long', ['pointer', 'uint32']],
    'CoCreateInstance': ['long', [ref.refType(ref.types.void), 'pointer', 'uint32', ref.refType(ref.types.void), 'pointer']],
    'CoUninitialize': ['void', []]
});

// COM initialization flags
const COINIT_APARTMENTTHREADED = 0x2;
const COINIT_MULTITHREADED = 0x0;
const CLSCTX_ALL = 23;
```

### 2.2 COM Interface Definitions

```typescript
// TypeScript definitions for common COM interfaces

interface IShellApplication {
    Explore(folderPath: string): void;
    Open(folderPath: string): void;
    NameSpace(folder: string | number): Folder;
    BrowseForFolder(hwnd: number, title: string, options: number, rootFolder: string): Folder;
    ShellExecute(file: string, arguments?: string, directory?: string, operation?: string, show?: number): void;
    MinimizeAll(): void;
    UndoMinimizeAll(): void;
    CascadeWindows(): void;
    TileHorizontally(): void;
    TileVertically(): void;
    ShutdownWindows(): void;
    Suspend(): void;
    EjectPC(): void;
    ServiceStart(serviceName: string, persistent: boolean): void;
    ServiceStop(serviceName: string, persistent: boolean): void;
    IsServiceRunning(serviceName: string): boolean;
    CanStartStopService(serviceName: string): boolean;
    GetSystemInformation(name: string): any;
}

interface IWScriptShell {
    CreateObject(progId: string, prefix?: string): any;
    GetObject(pathname: string, progId?: string, prefix?: string): any;
    Run(command: string, windowStyle?: number, waitOnReturn?: boolean): number;
    Exec(command: string): WshExec;
    ExpandEnvironmentStrings(src: string): string;
    Popup(text: string, secondsToWait?: number, title?: string, type?: number): number;
    RegRead(name: string): any;
    RegWrite(name: string, value: any, type?: string): void;
    RegDelete(name: string): void;
    SendKeys(keys: string, wait?: boolean): void;
    AppActivate(title: string | number): boolean;
}

interface IFileSystemObject {
    CreateFolder(folderName: string): Folder;
    DeleteFolder(folderName: string, force?: boolean): void;
    MoveFolder(source: string, destination: string): void;
    CopyFolder(source: string, destination: string, overwrite?: boolean): void;
    FolderExists(folderName: string): boolean;
    
    CreateTextFile(filename: string, overwrite?: boolean, unicode?: boolean): TextStream;
    OpenTextFile(filename: string, iomode?: number, create?: boolean, format?: number): TextStream;
    GetFile(filespec: string): File;
    DeleteFile(filespec: string, force?: boolean): void;
    MoveFile(source: string, destination: string): void;
    CopyFile(source: string, destination: string, overwrite?: boolean): void;
    FileExists(filespec: string): boolean;
    
    GetAbsolutePathName(path: string): string;
    GetBaseName(path: string): string;
    GetExtensionName(path: string): string;
    GetFileName(pathspec: string): string;
    GetParentFolderName(path: string): string;
    GetSpecialFolder(specialFolderType: number): Folder;
    GetTempName(): string;
    
    Drives: Drives;
}
```

---

## 3. Common COM Objects for System Automation

### 3.1 Shell.Application (CLSID: {13709620-C279-11CE-A49E-444553540000})

```javascript
// OpenClaw Shell Automation Module
class ShellAutomation {
    constructor() {
        this.shell = edge.func({
            source: `
                using System;
                using System.Dynamic;
                using Shell32;
                
                public class ShellWrapper {
                    private dynamic shell;
                    
                    public ShellWrapper() {
                        shell = Activator.CreateInstance(Type.GetTypeFromProgID("Shell.Application"));
                    }
                    
                    public object Explore(string path) {
                        shell.Explore(path);
                        return null;
                    }
                    
                    public object ShellExecute(string file, string args = "", 
                                               string dir = "", string op = "open", 
                                               int show = 1) {
                        return shell.ShellExecute(file, args, dir, op, show);
                    }
                    
                    public object MinimizeAll() {
                        shell.MinimizeAll();
                        return null;
                    }
                    
                    public object GetSystemInformation(string name) {
                        return shell.GetSystemInformation(name);
                    }
                }
            `
        });
    }
    
    async explore(folderPath) {
        return await this.shell({ method: 'Explore', path: folderPath });
    }
    
    async shellExecute(file, args = '', dir = '', op = 'open', show = 1) {
        return await this.shell({ 
            method: 'ShellExecute', 
            file, args, dir, op, show 
        });
    }
}
```

**Key Methods:**

| Method | Description | Parameters |
|--------|-------------|------------|
| `Explore(path)` | Open folder in Explorer | `path`: Folder path |
| `ShellExecute(file, args, dir, op, show)` | Execute file/operation | `file`: Target file |
| `NameSpace(folder)` | Get folder object | `folder`: Path or CSIDL |
| `BrowseForFolder(hwnd, title, options, root)` | Folder picker dialog | Dialog parameters |
| `MinimizeAll()` | Minimize all windows | None |
| `ServiceStart(name, persistent)` | Start Windows service | Service parameters |
| `GetSystemInformation(name)` | Query system info | Info type name |

### 3.2 WScript.Shell (CLSID: {72C24DD5-D70A-438B-8A42-98424B88AFB8})

```javascript
class WScriptAutomation {
    constructor() {
        this.wsh = null;
        this.initialize();
    }
    
    async initialize() {
        const createWSH = edge.func({
            source: `
                using System;
                using IWshRuntimeLibrary;
                
                public class WshWrapper {
                    private WshShell shell;
                    
                    public WshWrapper() {
                        shell = new WshShell();
                    }
                    
                    public string ExpandEnvironment(string src) {
                        return shell.ExpandEnvironmentStrings(src);
                    }
                    
                    public int Run(string command, int windowStyle = 0, bool wait = false) {
                        return shell.Run(command, windowStyle, wait);
                    }
                    
                    public object RegRead(string name) {
                        return shell.RegRead(name);
                    }
                    
                    public void RegWrite(string name, object value, string type = "REG_SZ") {
                        shell.RegWrite(name, value, type);
                    }
                    
                    public void SendKeys(string keys) {
                        shell.SendKeys(keys);
                    }
                    
                    public bool AppActivate(string title) {
                        return shell.AppActivate(title);
                    }
                }
            `,
            references: ['IWshRuntimeLibrary.dll']
        });
        
        this.wsh = await createWSH(null);
    }
    
    async run(command, windowStyle = 0, wait = false) {
        return await this.wsh.Run(command, windowStyle, wait);
    }
    
    async sendKeys(keys) {
        await this.wsh.SendKeys(keys);
    }
}
```

### 3.3 FileSystemObject (CLSID: {0D43FE01-F093-11CF-8940-00A0C9054228})

```javascript
class FileSystemAutomation {
    constructor() {
        this.fso = edge.func({
            source: `
                using System;
                using Scripting;
                
                public class FsoWrapper {
                    private FileSystemObject fso;
                    
                    public FsoWrapper() {
                        fso = new FileSystemObject();
                    }
                    
                    public bool FileExists(string path) {
                        return fso.FileExists(path);
                    }
                    
                    public bool FolderExists(string path) {
                        return fso.FolderExists(path);
                    }
                    
                    public void CreateFolder(string path) {
                        fso.CreateFolder(path);
                    }
                    
                    public void DeleteFile(string path, bool force = false) {
                        fso.DeleteFile(path, force);
                    }
                    
                    public void CopyFile(string source, string dest, bool overwrite = false) {
                        fso.CopyFile(source, dest, overwrite);
                    }
                    
                    public string GetAbsolutePathName(string path) {
                        return fso.GetAbsolutePathName(path);
                    }
                    
                    public string GetSpecialFolder(int type) {
                        return fso.GetSpecialFolder(type).Path;
                    }
                    
                    public dynamic GetFile(string path) {
                        var file = fso.GetFile(path);
                        return new {
                            Name = file.Name,
                            Path = file.Path,
                            Size = file.Size,
                            DateCreated = file.DateCreated,
                            DateLastModified = file.DateLastModified
                        };
                    }
                }
            `,
            references: ['Scrrun.dll']
        });
    }
}
```

### 3.4 WMI (Windows Management Instrumentation)

```javascript
class WmiAutomation {
    constructor() {
        this.wmi = edge.func({
            source: `
                using System;
                using System.Management;
                using System.Collections.Generic;
                
                public class WmiWrapper {
                    public List<Dictionary<string, object>> Query(string wql) {
                        var results = new List<Dictionary<string, object>>();
                        
                        using (var searcher = new ManagementObjectSearcher(wql)) {
                            foreach (ManagementObject obj in searcher.Get()) {
                                var dict = new Dictionary<string, object>();
                                foreach (PropertyData prop in obj.Properties) {
                                    dict[prop.Name] = prop.Value;
                                }
                                results.Add(dict);
                            }
                        }
                        
                        return results;
                    }
                    
                    public object GetProperty(string wql, string propertyName) {
                        using (var searcher = new ManagementObjectSearcher(wql)) {
                            foreach (ManagementObject obj in searcher.Get()) {
                                return obj[propertyName];
                            }
                        }
                        return null;
                    }
                    
                    public void InvokeMethod(string path, string methodName, object[] args) {
                        using (var obj = new ManagementObject(path)) {
                            obj.InvokeMethod(methodName, args);
                        }
                    }
                }
            `,
            references: ['System.Management.dll']
        });
    }
    
    async query(wql) {
        return await this.wmi({ method: 'Query', wql });
    }
    
    async getSystemInfo() {
        const queries = {
            os: 'SELECT * FROM Win32_OperatingSystem',
            processor: 'SELECT * FROM Win32_Processor',
            memory: 'SELECT * FROM Win32_PhysicalMemory',
            disk: 'SELECT * FROM Win32_LogicalDisk',
            network: 'SELECT * FROM Win32_NetworkAdapter WHERE NetEnabled = TRUE',
            processes: 'SELECT * FROM Win32_Process',
            services: 'SELECT * FROM Win32_Service'
        };
        
        const results = {};
        for (const [key, wql] of Object.entries(queries)) {
            results[key] = await this.query(wql);
        }
        return results;
    }
}
```

### 3.5 Special Folder Constants (CSIDL)

```javascript
const CSIDL = {
    DESKTOP: 0x0000,                    // Desktop
    INTERNET: 0x0001,                   // Internet Explorer
    PROGRAMS: 0x0002,                   // Start Menu/Programs
    CONTROLS: 0x0003,                   // Control Panel
    PRINTERS: 0x0004,                   // Printers
    PERSONAL: 0x0005,                   // My Documents
    FAVORITES: 0x0006,                  // Favorites
    STARTUP: 0x0007,                    // Startup
    RECENT: 0x0008,                     // Recent
    SENDTO: 0x0009,                     // SendTo
    BITBUCKET: 0x000a,                  // Recycle Bin
    STARTMENU: 0x000b,                  // Start Menu
    MYDOCUMENTS: 0x000c,                // My Documents (personal)
    MYMUSIC: 0x000d,                    // My Music
    MYVIDEO: 0x000e,                    // My Video
    DESKTOPDIRECTORY: 0x0010,           // Desktop Directory
    DRIVES: 0x0011,                     // My Computer
    NETWORK: 0x0012,                    // Network Neighborhood
    NETHOOD: 0x0013,                    // NetHood
    FONTS: 0x0014,                      // Fonts
    TEMPLATES: 0x0015,                  // Templates
    COMMON_STARTMENU: 0x0016,           // Common Start Menu
    COMMON_PROGRAMS: 0x0017,            // Common Programs
    COMMON_STARTUP: 0x0018,             // Common Startup
    COMMON_DESKTOPDIRECTORY: 0x0019,    // Common Desktop
    APPDATA: 0x001a,                    // Application Data
    PRINTHOOD: 0x001b,                  // PrintHood
    LOCAL_APPDATA: 0x001c,              // Local Application Data
    ALTSTARTUP: 0x001d,                 // Alternate Startup
    COMMON_ALTSTARTUP: 0x001e,          // Common Alternate Startup
    COMMON_FAVORITES: 0x001f,           // Common Favorites
    INTERNET_CACHE: 0x0020,             // Internet Cache
    COOKIES: 0x0021,                    // Cookies
    HISTORY: 0x0022,                    // History
    COMMON_APPDATA: 0x0023,             // Common Application Data
    WINDOWS: 0x0024,                    // Windows Directory
    SYSTEM: 0x0025,                     // System Directory
    PROGRAM_FILES: 0x0026,              // Program Files
    MYPICTURES: 0x0027,                 // My Pictures
    PROFILE: 0x0028,                    // User Profile
    SYSTEMX86: 0x0029,                  // System (x86)
    PROGRAM_FILESX86: 0x002a,           // Program Files (x86)
    COMMON_PROGRAM_FILES: 0x002b,       // Common Program Files
    COMMON_PROGRAM_FILESX86: 0x002c,    // Common Program Files (x86)
    COMMON_TEMPLATES: 0x002d,           // Common Templates
    COMMON_DOCUMENTS: 0x002e,           // Common Documents
    COMMON_ADMINTOOLS: 0x002f,          // Common Admin Tools
    ADMINTOOLS: 0x0030,                 // Admin Tools
    CONNECTIONS: 0x0031,                // Network Connections
    COMMON_MUSIC: 0x0035,               // Common Music
    COMMON_PICTURES: 0x0036,            // Common Pictures
    COMMON_VIDEOS: 0x0037,              // Common Videos
    RESOURCES: 0x0038,                  // Resources
    RESOURCES_LOCALIZED: 0x0039,        // Localized Resources
    COMMON_OEM_LINKS: 0x003a,           // Common OEM Links
    CDBURN_AREA: 0x003b,                // CD Burning
    COMPUTERSNEARME: 0x003d             // Computers Near Me
};
```

---

## 4. Windows Runtime (WinRT) Activation

### 4.1 WinRT Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    WinRT Activation Flow                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐   │
│  │   Node.js   │────▶│   NodeRT    │────▶│  WinRT Runtime  │   │
│  │   Agent     │     │   Binding   │     │   (COM-based)   │   │
│  └─────────────┘     └─────────────┘     └─────────────────┘   │
│         │                   │                    │              │
│         │            ┌──────▼──────┐             │              │
│         │            │  Windows.   │             │              │
│         │            │  Runtime.   │             │              │
│         │            │  dll        │             │              │
│         │            └─────────────┘             │              │
│         │                                        │              │
│         └────────────────────────────────────────┘              │
│                         Activation APIs                          │
│              (RoActivateInstance, GetActivationFactory)          │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Activation Patterns

#### 4.2.1 Direct Activation (In-Process)

```javascript
// Using NodeRT for direct WinRT activation
const { 
    SpeechSynthesizer 
} = require('@nodert-win10-rs4/windows.media.speechsynthesis');

// Direct activation - creates instance in same process
const synthesizer = new SpeechSynthesizer();
```

#### 4.2.2 Factory-Based Activation

```javascript
// Factory pattern for complex initialization
const { 
    Geolocator,
    GeolocationAccessStatus
} = require('@nodert-win10-rs4/windows.devices.geolocation');

// Access status check before activation
async function createGeolocator() {
    const { status } = await Geolocator.requestAccessAsync();
    
    if (status === GeolocationAccessStatus.allowed) {
        return new Geolocator();
    }
    throw new Error('Geolocation access denied');
}
```

#### 4.2.3 Out-of-Process Activation (Desktop Bridge)

```xml
<!-- Package.appxmanifest for OOP server -->
<Package ...>
  <Applications>
    <Application Id="OpenClawAgent" 
                 Executable="OpenClaw.exe"
                 EntryPoint="Windows.FullTrustApplication">
      <Extensions>
        <!-- In-Process Server -->
        <Extension Category="windows.activatableClass.inProcessServer">
          <InProcessServer>
            <Path>OpenClawWinRT.dll</Path>
            <ActivatableClass ActivatableClassId="OpenClaw.AgentRuntime" 
                              ThreadingModel="both"/>
          </InProcessServer>
        </Extension>
        
        <!-- Out-of-Process Server -->
        <Extension Category="windows.activatableClass.outOfProcessServer">
          <OutOfProcessServer ServerName="OpenClaw.AgentServer"
                              uap5:IdentityType="activateAsPackage"
                              uap5:RunFullTrust="true">
            <Path>OpenClawAgent.exe</Path>
            <Instancing>singleInstance</Instancing>
            <ActivatableClass ActivatableClassId="OpenClaw.AgentService"/>
          </OutOfProcessServer>
        </Extension>
      </Extensions>
    </Application>
  </Applications>
</Package>
```

### 4.3 WinRT Component Implementation

```cpp
// OpenClawWinRT.idl - WinRT Component Definition
namespace OpenClaw
{
    [version(1.0)]
    interface IAgentRuntime : IInspectable
    {
        HRESULT Initialize([in] HSTRING configPath);
        HRESULT ExecuteCommand([in] HSTRING command, [out, retval] HSTRING* result);
        HRESULT GetStatus([out, retval] AgentStatus* status);
    };

    [version(1.0)]
    runtimeclass AgentRuntime
    {
        [default] interface IAgentRuntime;
        interface Windows.Foundation.IClosable;
    };
}
```

```cpp
// AgentRuntime.h - C++/WinRT Implementation
#pragma once
#include "OpenClawWinRT.g.h"

namespace winrt::OpenClaw::implementation
{
    struct AgentRuntime : AgentRuntimeT<AgentRuntime>
    {
        AgentRuntime() = default;
        
        void Initialize(hstring const& configPath);
        hstring ExecuteCommand(hstring const& command);
        AgentStatus GetStatus();
        void Close();
        
    private:
        std::wstring m_configPath;
        bool m_initialized = false;
    };
}
```

---

## 5. UWP API Access from Desktop Agent

### 5.1 Desktop Bridge Configuration

```xml
<!-- Package.appxmanifest -->
<?xml version="1.0" encoding="utf-8"?>
<Package xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"
         xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"
         xmlns:uap5="http://schemas.microsoft.com/appx/manifest/uap/windows10/5"
         xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities"
         IgnorableNamespaces="uap uap5 rescap">
  
  <Identity Name="OpenClaw.Agent"
            Publisher="CN=OpenClaw"
            Version="1.0.0.0"
            ProcessorArchitecture="x64"/>
  
  <Properties>
    <DisplayName>OpenClaw AI Agent</DisplayName>
    <PublisherDisplayName>OpenClaw Systems</PublisherDisplayName>
    <Logo>Assets\StoreLogo.png</Logo>
    <uap10:AllowExternalContent>
      <uap10:ExternalContent Path="agent-data\"/>
    </uap10:AllowExternalContent>
  </Properties>
  
  <Dependencies>
    <TargetDeviceFamily Name="Windows.Desktop" MinVersion="10.0.19041.0" MaxVersionTested="10.0.22000.0"/>
  </Dependencies>
  
  <Resources>
    <Resource Language="en-us"/>
  </Resources>
  
  <Capabilities>
    <!-- Standard Capabilities -->
    <uap:Capability Name="internetClient"/>
    <uap:Capability Name="internetClientServer"/>
    <uap:Capability Name="privateNetworkClientServer"/>
    <uap:Capability Name="musicLibrary"/>
    <uap:Capability Name="picturesLibrary"/>
    <uap:Capability Name="videosLibrary"/>
    <uap:Capability Name="documentsLibrary"/>
    <uap:Capability Name="removableStorage"/>
    
    <!-- Device Capabilities -->
    <DeviceCapability Name="microphone"/>
    <DeviceCapability Name="webcam"/>
    <DeviceCapability Name="location"/>
    <DeviceCapability Name="bluetooth"/>
    <DeviceCapability Name="radios"/>
    
    <!-- Restricted Capabilities -->
    <rescap:Capability Name="runFullTrust"/>
    <rescap:Capability Name="allowElevation"/>
    <rescap:Capability Name="packageManagement"/>
    <rescap:Capability Name="packageQuery"/>
  </Capabilities>
  
  <Applications>
    <Application Id="OpenClawAgent"
                 Executable="node.exe"
                 EntryPoint="Windows.FullTrustApplication">
      <uap:VisualElements DisplayName="OpenClaw AI Agent"
                          Description="24/7 AI Agent System"
                          BackgroundColor="transparent"
                          Square150x150Logo="Assets\Square150x150Logo.png"
                          Square44x44Logo="Assets\Square44x44Logo.png"/>
      
      <Extensions>
        <!-- Background Task -->
        <Extension Category="windows.backgroundTasks"
                   EntryPoint="OpenClaw.BackgroundTask">
          <BackgroundTasks>
            <Task Type="timer"/>
            <Task Type="systemEvent"/>
            <Task Type="pushNotification"/>
          </BackgroundTasks>
        </Extension>
        
        <!-- Protocol Activation -->
        <uap:Extension Category="windows.protocol">
          <uap:Protocol Name="openclaw">
            <uap:DisplayName>OpenClaw Protocol</uap:DisplayName>
          </uap:Protocol>
        </uap:Extension>
        
        <!-- File Type Association -->
        <uap:Extension Category="windows.fileTypeAssociation">
          <uap:FileTypeAssociation Name="openclaw-config">
            <uap:SupportedFileTypes>
              <uap:FileType>.claw</uap:FileType>
              <uap:FileType>.clawrc</uap:FileType>
            </uap:SupportedFileTypes>
          </uap:FileTypeAssociation>
        </uap:Extension>
        
        <!-- Toast Notifications -->
        <uap:Extension Category="windows.toastNotificationActivation"
                       ToastActivatorCLSID="YOUR-GUID-HERE"/>
      </Extensions>
    </Application>
  </Applications>
</Package>
```

### 5.2 API Availability Matrix

| WinRT Namespace | Desktop Support | Requires Package | Notes |
|-----------------|-----------------|------------------|-------|
| `Windows.Media.Speech` | Full | No | Direct from desktop |
| `Windows.System` | Partial | No | Launcher requires IInitializeWithWindow |
| `Windows.Storage` | Full | No | Pickers require IInitializeWithWindow |
| `Windows.Networking` | Full | No | Direct from desktop |
| `Windows.Data` | Full | No | Direct from desktop |
| `Windows.ApplicationModel` | Partial | Yes | Background tasks need package |
| `Windows.UI.Notifications` | Partial | Yes | Toasts need package |
| `Windows.Devices` | Full | No | Direct from desktop |

### 5.3 IInitializeWithWindow Pattern

```javascript
// For APIs requiring window handle (desktop apps)
const { 
    FileOpenPicker,
    FileSavePicker,
    FolderPicker
} = require('@nodert-win10-rs4/windows.storage.pickers');

const { 
    Launcher,
    LauncherOptions 
} = require('@nodert-win10-rs4/windows.system');

class DesktopBridgeHelper {
    constructor() {
        // Get console window handle (for Node.js console apps)
        this.hwnd = this.getConsoleWindow();
    }
    
    getConsoleWindow() {
        const kernel32 = ffi.Library('kernel32', {
            'GetConsoleWindow': ['pointer', []]
        });
        return kernel32.GetConsoleWindow();
    }
    
    async initializePicker(picker) {
        // Initialize picker with window handle
        const IInitializeWithWindow = require('@nodert-win10-rs4/windows.system.interop');
        const init = picker.as(IInitializeWithWindow);
        init.Initialize(this.hwnd);
        return picker;
    }
    
    async showFileOpenPicker() {
        const picker = new FileOpenPicker();
        picker.suggestedStartLocation = PickerLocationId.documentsLibrary;
        picker.fileTypeFilter.append('.txt');
        picker.fileTypeFilter.append('.json');
        
        await this.initializePicker(picker);
        return await picker.pickSingleFileAsync();
    }
    
    async launchFile(file) {
        const options = new LauncherOptions();
        options.displayApplicationPicker = true;
        
        // LauncherOptions also needs window handle
        await this.initializePicker(options);
        
        return await Launcher.launchFileAsync(file, options);
    }
}
```

---

## 6. Windows.Media.Speech for TTS/STT

### 6.1 Text-to-Speech (TTS) Implementation

#### 6.1.1 Using @echogarden/windows-media-tts (Recommended)

```javascript
// OpenClaw TTS Module
const { 
    getVoiceList, 
    synthesize 
} = require('@echogarden/windows-media-tts');
const { writeFile } = require('node:fs/promises');
const { playAudio } = require('./audio-player');

class WindowsTTS {
    constructor() {
        this.voices = [];
        this.defaultVoice = null;
        this.initialize();
    }
    
    initialize() {
        this.voices = getVoiceList();
        this.defaultVoice = this.voices.find(v => v.language === 'en-US') || this.voices[0];
        console.log(`TTS Initialized with voice: ${this.defaultVoice?.displayName}`);
    }
    
    getVoices(language = null) {
        if (language) {
            return this.voices.filter(v => v.language === language);
        }
        return this.voices;
    }
    
    async speak(text, options = {}) {
        const {
            voiceName = this.defaultVoice?.displayName,
            speakingRate = 1.0,
            audioPitch = 1.0,
            enableSsml = false,
            outputFile = null
        } = options;
        
        try {
            const result = synthesize(text, {
                voiceName,
                speakingRate,
                audioPitch,
                enableSsml
            });
            
            const { audioData, markers, timedMetadataTracks } = result;
            
            // Save to file if requested
            if (outputFile) {
                await writeFile(outputFile, audioData);
            }
            
            // Play audio
            await playAudio(audioData);
            
            return {
                success: true,
                duration: this.calculateDuration(audioData),
                markers,
                timedMetadataTracks
            };
        } catch (error) {
            console.error('TTS Error:', error);
            throw error;
        }
    }
    
    async speakSsml(ssml, options = {}) {
        return this.speak(ssml, { ...options, enableSsml: true });
    }
    
    calculateDuration(audioData) {
        // WAVE format: 44 bytes header, rest is PCM
        const dataSize = audioData.length - 44;
        // Assuming 16-bit stereo 44100Hz (standard)
        return dataSize / (2 * 2 * 44100);
    }
}

// SSML Builder for advanced TTS
class SsmlBuilder {
    constructor() {
        this.parts = [];
    }
    
    speak(language = 'en-US') {
        this.parts.push(`<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="${language}">`);
        return this;
    }
    
    voice(name) {
        this.parts.push(`<voice name="${name}">`);
        return this;
    }
    
    prosody({ rate, pitch, volume, text }) {
        const attrs = [];
        if (rate) attrs.push(`rate="${rate}"`);
        if (pitch) attrs.push(`pitch="${pitch}"`);
        if (volume) attrs.push(`volume="${volume}"`);
        this.parts.push(`<prosody ${attrs.join(' ')}>${text}</prosody>`);
        return this;
    }
    
    emphasis(level = 'moderate', text) {
        this.parts.push(`<emphasis level="${level}">${text}</emphasis>`);
        return this;
    }
    
    break(time) {
        this.parts.push(`<break time="${time}"/>`);
        return this;
    }
    
    mark(name) {
        this.parts.push(`<mark name="${name}"/>`);
        return this;
    }
    
    sayAs(interpretAs, text, format = null) {
        const formatAttr = format ? ` format="${format}"` : '';
        this.parts.push(`<say-as interpret-as="${interpretAs}"${formatAttr}>${text}</say-as>`);
        return this;
    }
    
    audio(src) {
        this.parts.push(`<audio src="${src}"/>`);
        return this;
    }
    
    text(content) {
        this.parts.push(content);
        return this;
    }
    
    build() {
        this.parts.push('</voice></speak>');
        return this.parts.join('');
    }
}

// Usage example
const tts = new WindowsTTS();

// Simple speech
await tts.speak('Hello, I am OpenClaw, your AI assistant.');

// SSML advanced speech
const ssml = new SsmlBuilder()
    .speak('en-US')
    .voice('Microsoft David')
    .prosody({ rate: 'slow', pitch: 'low', text: 'Welcome to OpenClaw.' })
    .break('500ms')
    .emphasis('strong', 'Your 24/7 AI agent is ready.')
    .sayAs('date', '2024-01-15', 'mdy')
    .build();

await tts.speakSsml(ssml);
```

#### 6.1.2 Using NodeRT Directly

```javascript
// Alternative: Direct WinRT SpeechSynthesis
const { 
    SpeechSynthesizer,
    VoiceInformation
} = require('@nodert-win10-rs4/windows.media.speechsynthesis');

const {
    RandomAccessStreamReference
} = require('@nodert-win10-rs4/windows.storage.streams');

class WinRTTTS {
    constructor() {
        this.synthesizer = new SpeechSynthesizer();
        this.initialize();
    }
    
    async initialize() {
        // Get all voices
        const voices = SpeechSynthesizer.allVoices;
        
        // Set default voice (English)
        for (let i = 0; i < voices.size; i++) {
            const voice = voices.getAt(i);
            if (voice.language === 'en-US' && voice.gender === 1) { // Male
                this.synthesizer.voice = voice;
                break;
            }
        }
    }
    
    async synthesizeText(text) {
        const stream = await this.synthesizer.synthesizeTextToStreamAsync(text);
        return await this.readStream(stream);
    }
    
    async synthesizeSsml(ssml) {
        const stream = await this.synthesizer.synthesizeSsmlToStreamAsync(ssml);
        return await this.readStream(stream);
    }
    
    async readStream(stream) {
        const { InputStream } = require('@nodert-win10-rs4/nodert-streams');
        const nodeStream = new InputStream(stream.getInputStreamAt(0));
        
        return new Promise((resolve, reject) => {
            const chunks = [];
            nodeStream.on('data', chunk => chunks.push(chunk));
            nodeStream.on('end', () => resolve(Buffer.concat(chunks)));
            nodeStream.on('error', reject);
        });
    }
}
```

### 6.2 Speech-to-Text (STT) Implementation

```javascript
// OpenClaw STT Module using Windows.Media.SpeechRecognition
const {
    SpeechRecognizer,
    SpeechRecognitionScenario,
    SpeechRecognitionResultStatus
} = require('@nodert-win10-rs4/windows.media.speechrecognition');

const {
    Language
} = require('@nodert-win10-rs4/windows.globalization');

class WindowsSTT {
    constructor(options = {}) {
        this.language = options.language || 'en-US';
        this.recognizer = null;
        this.isListening = false;
        this.continuousMode = options.continuous || false;
    }
    
    async initialize() {
        // Check speech recognition availability
        const constraint = SpeechRecognitionScenario.default;
        
        this.recognizer = new SpeechRecognizer(new Language(this.language));
        
        // Configure recognizer
        this.recognizer.timeout = {
            initialSilenceTimeout: 5000,
            endSilenceTimeout: 1500,
            babbleTimeout: 0
        };
        
        // Compile constraints
        await this.recognizer.compileConstraintsAsync();
    }
    
    async recognizeOnce() {
        if (!this.recognizer) {
            await this.initialize();
        }
        
        this.isListening = true;
        
        try {
            const result = await this.recognizer.recognizeAsync();
            
            this.isListening = false;
            
            return {
                text: result.text,
                confidence: this.mapConfidence(result.confidence),
                status: result.status,
                rawConfidence: result.confidence
            };
        } catch (error) {
            this.isListening = false;
            throw error;
        }
    }
    
    async startContinuousRecognition(callback) {
        if (!this.recognizer) {
            await this.initialize();
        }
        
        this.continuousMode = true;
        
        // Set up event handlers
        this.recognizer.addEventListener('resultgenerated', (sender, args) => {
            if (args.result.status === SpeechRecognitionResultStatus.success) {
                callback({
                    type: 'result',
                    text: args.result.text,
                    confidence: this.mapConfidence(args.result.confidence),
                    isFinal: true
                });
            }
        });
        
        this.recognizer.addEventListener('hypothesisgenerated', (sender, args) => {
            callback({
                type: 'hypothesis',
                text: args.hypothesis.text,
                isFinal: false
            });
        });
        
        // Start continuous recognition
        await this.recognizer.continuousRecognitionSession.startAsync();
        this.isListening = true;
    }
    
    async stopContinuousRecognition() {
        if (this.recognizer && this.isListening) {
            await this.recognizer.continuousRecognitionSession.stopAsync();
            this.isListening = false;
        }
    }
    
    mapConfidence(winrtConfidence) {
        // Map WinRT confidence (0-3) to 0-1 scale
        const confidenceMap = {
            0: 0.3,  // Low
            1: 0.6,  // Medium
            2: 0.85, // High
            3: 0.98  // Very High
        };
        return confidenceMap[winrtConfidence] || 0.5;
    }
    
    dispose() {
        if (this.recognizer) {
            this.recognizer.close();
            this.recognizer = null;
        }
    }
}

// Grammar-based recognition
class GrammarBasedSTT extends WindowsSTT {
    constructor(grammarList, options = {}) {
        super(options);
        this.grammarList = grammarList;
    }
    
    async initialize() {
        this.recognizer = new SpeechRecognizer(new Language(this.language));
        
        // Add grammar constraints
        const {
            SpeechRecognitionListConstraint
        } = require('@nodert-win10-rs4/windows.media.speechrecognition');
        
        const constraint = new SpeechRecognitionListConstraint(this.grammarList);
        this.recognizer.constraints.append(constraint);
        
        await this.recognizer.compileConstraintsAsync();
    }
}

// Usage
const stt = new WindowsSTT({ language: 'en-US' });

// Single recognition
const result = await stt.recognizeOnce();
console.log(`Recognized: ${result.text} (confidence: ${result.confidence})`);

// Continuous recognition
await stt.startContinuousRecognition((event) => {
    if (event.type === 'result') {
        console.log(`Final: ${event.text}`);
    } else {
        console.log(`Hypothesis: ${event.text}`);
    }
});

// Stop after 30 seconds
setTimeout(() => stt.stopContinuousRecognition(), 30000);
```

---

## 7. Windows.System and Windows.Storage APIs

### 7.1 Windows.System.Launcher

```javascript
const {
    Launcher,
    LauncherOptions,
    FolderLauncherOptions
} = require('@nodert-win10-rs4/windows.system');

const {
    StorageFile,
    StorageFolder
} = require('@nodert-win10-rs4/windows.storage');

class SystemLauncher {
    constructor() {
        this.hwnd = this.getConsoleWindow();
    }
    
    getConsoleWindow() {
        const ffi = require('ffi-napi');
        const kernel32 = ffi.Library('kernel32', {
            'GetConsoleWindow': ['pointer', []]
        });
        return kernel32.GetConsoleWindow();
    }
    
    async launchFile(filePath, options = {}) {
        const file = await StorageFile.getFileFromPathAsync(filePath);
        
        const launcherOptions = new LauncherOptions();
        launcherOptions.displayApplicationPicker = options.showPicker || false;
        launcherOptions.treatAsUntrusted = options.untrusted || false;
        
        // Initialize with window handle for desktop apps
        const { InitializeWithWindow } = require('@nodert-win10-rs4/windows.system.interop');
        const init = launcherOptions.as(InitializeWithWindow);
        init.Initialize(this.hwnd);
        
        return await Launcher.launchFileAsync(file, launcherOptions);
    }
    
    async launchFolder(folderPath, options = {}) {
        const folder = await StorageFolder.getFolderFromPathAsync(folderPath);
        
        const folderOptions = new FolderLauncherOptions();
        if (options.itemsToSelect) {
            for (const itemPath of options.itemsToSelect) {
                const item = await StorageFile.getFileFromPathAsync(itemPath);
                folderOptions.itemsToSelect.append(item);
            }
        }
        
        return await Launcher.launchFolderAsync(folder, folderOptions);
    }
    
    async launchUri(uri, options = {}) {
        const launcherOptions = new LauncherOptions();
        launcherOptions.displayApplicationPicker = options.showPicker || false;
        
        const uriObj = new (require('@nodert-win10-rs4/windows.foundation')).Uri(uri);
        return await Launcher.launchUriAsync(uriObj, launcherOptions);
    }
    
    async queryFileSupport(filePath) {
        const file = await StorageFile.getFileFromPathAsync(filePath);
        const launchQuery = await Launcher.queryFileSupportAsync(file);
        return launchQuery;
    }
    
    async queryUriSupport(uri, launchQueryType = 'launch') {
        const { LaunchQuerySupportType, LaunchQuerySupportStatus } = require('@nodert-win10-rs4/windows.system');
        const uriObj = new (require('@nodert-win10-rs4/windows.foundation')).Uri(uri);
        const queryType = LaunchQuerySupportType[launchQueryType];
        return await Launcher.queryUriSupportAsync(uriObj, queryType);
    }
}
```

### 7.2 Windows.Storage API

```javascript
const {
    StorageFile,
    StorageFolder,
    KnownFolders,
    FileIO,
    PathIO,
    CachedFileManager,
    StorageItemTypes,
    FileAccessMode,
    CreationCollisionOption,
    NameCollisionOption
} = require('@nodert-win10-rs4/windows.storage');

const {
    FileOpenPicker,
    FileSavePicker,
    FolderPicker,
    PickerLocationId
} = require('@nodert-win10-rs4/windows.storage.pickers');

class StorageManager {
    constructor() {
        this.hwnd = this.getConsoleWindow();
    }
    
    getConsoleWindow() {
        const ffi = require('ffi-napi');
        const kernel32 = ffi.Library('kernel32', {
            'GetConsoleWindow': ['pointer', []]
        });
        return kernel32.GetConsoleWindow();
    }
    
    // Known Folders
    getKnownFolders() {
        return {
            documents: KnownFolders.documentsLibrary,
            pictures: KnownFolders.picturesLibrary,
            music: KnownFolders.musicLibrary,
            videos: KnownFolders.videosLibrary,
            homeGroup: KnownFolders.homeGroup,
            mediaServerDevices: KnownFolders.mediaServerDevices,
            savedPictures: KnownFolders.savedPictures,
            cameraRoll: KnownFolders.cameraRoll,
            appCaptures: KnownFolders.appCaptures,
            recordedCalls: KnownFolders.recordedCalls,
            objects3D: KnownFolders.objects3D
        };
    }
    
    async readTextFile(filePath) {
        const file = await StorageFile.getFileFromPathAsync(filePath);
        return await FileIO.readTextAsync(file);
    }
    
    async readLines(filePath) {
        const file = await StorageFile.getFileFromPathAsync(filePath);
        return await FileIO.readLinesAsync(file);
    }
    
    async writeTextFile(filePath, text, options = {}) {
        const folder = await StorageFolder.getFolderFromPathAsync(
            require('path').dirname(filePath)
        );
        const filename = require('path').basename(filePath);
        
        const collisionOption = options.replaceExisting 
            ? CreationCollisionOption.replaceExisting 
            : CreationCollisionOption.failIfExists;
        
        const file = await folder.createFileAsync(filename, collisionOption);
        await FileIO.writeTextAsync(file, text);
        return file;
    }
    
    async appendText(filePath, text) {
        const file = await StorageFile.getFileFromPathAsync(filePath);
        await FileIO.appendTextAsync(file, text);
    }
    
    async writeBytes(filePath, buffer) {
        const folder = await StorageFolder.getFolderFromPathAsync(
            require('path').dirname(filePath)
        );
        const filename = require('path').basename(filePath);
        const file = await folder.createFileAsync(filename, CreationCollisionOption.replaceExisting);
        
        // Convert Node Buffer to WinRT IBuffer
        const { CryptographicBuffer } = require('@nodert-win10-rs4/windows.security.cryptography');
        const winrtBuffer = CryptographicBuffer.createFromByteArray(Array.from(buffer));
        
        await FileIO.writeBufferAsync(file, winrtBuffer);
    }
    
    async copyFile(sourcePath, destPath, options = {}) {
        const sourceFile = await StorageFile.getFileFromPathAsync(sourcePath);
        const destFolder = await StorageFolder.getFolderFromPathAsync(
            require('path').dirname(destPath)
        );
        const destName = require('path').basename(destPath);
        
        const nameOption = options.replaceExisting 
            ? NameCollisionOption.replaceExisting 
            : NameCollisionOption.failIfExists;
        
        return await sourceFile.copyAsync(destFolder, destName, nameOption);
    }
    
    async moveFile(sourcePath, destPath, options = {}) {
        const sourceFile = await StorageFile.getFileFromPathAsync(sourcePath);
        const destFolder = await StorageFolder.getFolderFromPathAsync(
            require('path').dirname(destPath)
        );
        const destName = require('path').basename(destPath);
        
        const nameOption = options.replaceExisting 
            ? NameCollisionOption.replaceExisting 
            : NameCollisionOption.failIfExists;
        
        await sourceFile.moveAsync(destFolder, destName, nameOption);
    }
    
    async deleteFile(filePath, options = {}) {
        const file = await StorageFile.getFileFromPathAsync(filePath);
        const option = options.permanentlyDelete 
            ? require('@nodert-win10-rs4/windows.storage').StorageDeleteOption.permanentDelete
            : require('@nodert-win10-rs4/windows.storage').StorageDeleteOption.default;
        await file.deleteAsync(option);
    }
    
    async listFolder(folderPath, options = {}) {
        const folder = await StorageFolder.getFolderFromPathAsync(folderPath);
        const query = folder.createItemQuery();
        
        const items = await query.getItemsAsync(
            options.startIndex || 0, 
            options.maxItems || 100
        );
        
        const results = [];
        for (let i = 0; i < items.size; i++) {
            const item = items.getAt(i);
            results.push({
                name: item.name,
                path: item.path,
                type: item.isOfType(StorageItemTypes.file) ? 'file' : 'folder',
                dateCreated: item.dateCreated
            });
        }
        
        return results;
    }
    
    // File Pickers (require IInitializeWithWindow)
    async showOpenFilePicker(options = {}) {
        const picker = new FileOpenPicker();
        picker.suggestedStartLocation = options.startLocation || PickerLocationId.documentsLibrary;
        picker.viewMode = options.viewMode || require('@nodert-win10-rs4/windows.storage.pickers').PickerViewMode.list;
        
        // Add file type filters
        const filters = options.filters || ['*'];
        filters.forEach(f => picker.fileTypeFilter.append(f));
        
        // Initialize with window handle
        const { InitializeWithWindow } = require('@nodert-win10-rs4/windows.system.interop');
        const init = picker.as(InitializeWithWindow);
        init.Initialize(this.hwnd);
        
        if (options.multiple) {
            return await picker.pickMultipleFilesAsync();
        }
        return await picker.pickSingleFileAsync();
    }
    
    async showSaveFilePicker(options = {}) {
        const picker = new FileSavePicker();
        picker.suggestedStartLocation = options.startLocation || PickerLocationId.documentsLibrary;
        picker.suggestedFileName = options.suggestedName || 'Untitled';
        
        // Add file type choices
        if (options.fileTypes) {
            options.fileTypes.forEach(({ name, extensions }) => {
                const extList = new (require('@nodert-win10-rs4/windows.foundation.collections')).Vector();
                extensions.forEach(e => extList.append(e));
                picker.fileTypeChoices.insert(name, extList);
            });
        }
        
        // Initialize with window handle
        const { InitializeWithWindow } = require('@nodert-win10-rs4/windows.system.interop');
        const init = picker.as(InitializeWithWindow);
        init.Initialize(this.hwnd);
        
        return await picker.pickSaveFileAsync();
    }
    
    async showFolderPicker(options = {}) {
        const picker = new FolderPicker();
        picker.suggestedStartLocation = options.startLocation || PickerLocationId.documentsLibrary;
        picker.fileTypeFilter.append('*');
        
        // Initialize with window handle
        const { InitializeWithWindow } = require('@nodert-win10-rs4/windows.system.interop');
        const init = picker.as(InitializeWithWindow);
        init.Initialize(this.hwnd);
        
        return await picker.pickSingleFolderAsync();
    }
}
```

### 7.3 Application Data Storage

```javascript
const {
    ApplicationData,
    ApplicationDataContainer,
    ApplicationDataCreateDisposition,
    Locality
} = require('@nodert-win10-rs4/windows.storage');

class AppDataManager {
    constructor() {
        this.appData = ApplicationData.current;
        this.localSettings = this.appData.localSettings;
        this.roamingSettings = this.appData.roamingSettings;
        this.localFolder = this.appData.localFolder;
        this.roamingFolder = this.appData.roamingFolder;
        this.temporaryFolder = this.appData.temporaryFolder;
        
        // Sync roaming data
        this.setupRoamingSync();
    }
    
    setupRoamingSync() {
        this.appData.addEventListener('datachanged', (sender) => {
            console.log('Roaming data changed');
            // Reload roaming settings
        });
    }
    
    // Settings Operations
    setLocalSetting(key, value) {
        this.localSettings.values.insert(key, value);
    }
    
    getLocalSetting(key, defaultValue = null) {
        const value = this.localSettings.values.lookup(key);
        return value !== undefined ? value : defaultValue;
    }
    
    removeLocalSetting(key) {
        this.localSettings.values.remove(key);
    }
    
    setRoamingSetting(key, value) {
        this.roamingSettings.values.insert(key, value);
    }
    
    getRoamingSetting(key, defaultValue = null) {
        const value = this.roamingSettings.values.lookup(key);
        return value !== undefined ? value : defaultValue;
    }
    
    // Composite Settings
    createCompositeSetting(containerName) {
        return this.localSettings.createContainer(
            containerName,
            ApplicationDataCreateDisposition.always
        );
    }
    
    // File Operations
    async writeLocalFile(filename, content) {
        const file = await this.localFolder.createFileAsync(
            filename,
            CreationCollisionOption.replaceExisting
        );
        await FileIO.writeTextAsync(file, content);
        return file;
    }
    
    async readLocalFile(filename) {
        const file = await this.localFolder.getFileAsync(filename);
        return await FileIO.readTextAsync(file);
    }
    
    // Version Management
    get version() {
        return this.appData.version;
    }
    
    async setVersion(version, handler) {
        await this.appData.setVersionAsync(version, handler);
    }
    
    // Signal data change
    signalDataChanged() {
        this.appData.signalDataChanged();
    }
    
    // Clear data
    async clearAsync(locality) {
        await this.appData.clearAsync(locality);
    }
}
```

---

## 8. COM Security and Marshaling

### 8.1 COM Security Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COM Security Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        COM Security Context                          │   │
│  │                                                                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │   Process   │  │   Thread    │  │   Object    │  │   Call      │ │   │
│  │  │  Identity   │  │   Token     │  │   ACL       │  │   Context   │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Security Functions:                                                         │
│  • CoInitializeSecurity() - Set process-wide security                       │
│  • CoSetProxyBlanket() - Set authentication for proxy                       │
│  • CoQueryProxyBlanket() - Query proxy authentication                       │
│  • CoImpersonateClient() - Impersonate client context                       │
│  • CoRevertToSelf() - Revert to process token                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 COM Security Implementation

```javascript
// COM Security Configuration
const ffi = require('ffi-napi');
const ref = require('ref-napi');

// COM Security Constants
const RPC_C_AUTHN_LEVEL = {
    DEFAULT: 0,
    NONE: 1,
    CONNECT: 2,
    CALL: 3,
    PKT: 4,
    PKT_INTEGRITY: 5,
    PKT_PRIVACY: 6
};

const RPC_C_IMP_LEVEL = {
    DEFAULT: 0,
    ANONYMOUS: 1,
    IDENTIFY: 2,
    IMPERSONATE: 3,
    DELEGATE: 4
};

const EOAC = {
    NONE: 0,
    DEFAULT: 0x800,
    STATIC_CLOAKING: 0x20,
    DYNAMIC_CLOAKING: 0x40,
    ANY_AUTHORITY: 0x80,
    MAKE_FULLSIC: 0x100,
    SECURE_REFS: 0x2,
    ACCESS_CONTROL: 0x4,
    APPID: 0x8,
    DYNAMIC: 0x10,
    REQUIRE_FULLSIC: 0x200,
    AUTO_IMPERSONATE: 0x400,
    NO_CUSTOM_MARSHAL: 0x2000,
    DISABLE_AAA: 0x1000
};

class ComSecurity {
    constructor() {
        this.ole32 = ffi.Library('ole32', {
            'CoInitializeSecurity': ['long', [
                'pointer',      // PSECURITY_DESCRIPTOR
                'int',          // cAuthSvc
                'pointer',      // asAuthSvc
                'pointer',      // pReserved1
                'uint',         // dwAuthnLevel
                'uint',         // dwImpLevel
                'pointer',      // pAuthList
                'uint',         // dwCapabilities
                'pointer'       // pReserved3
            ]],
            'CoSetProxyBlanket': ['long', [
                'pointer',      // pProxy
                'uint',         // dwAuthnSvc
                'uint',         // dwAuthzSvc
                'pointer',      // pServerPrincName
                'uint',         // dwAuthnLevel
                'uint',         // dwImpLevel
                'pointer',      // pAuthInfo
                'uint'          // dwCapabilities
            ]],
            'CoQueryProxyBlanket': ['long', [
                'pointer',      // pProxy
                'pointer',      // pdwAuthnSvc
                'pointer',      // pdwAuthzSvc
                'pointer',      // pServerPrincName
                'pointer',      // pdwAuthnLevel
                'pointer',      // pdwImpLevel
                'pointer',      // pAuthInfo
                'pointer'       // pdwCapabilities
            ]]
        });
    }
    
    initializeSecurity(options = {}) {
        const authnLevel = options.authnLevel || RPC_C_AUTHN_LEVEL.PKT;
        const impLevel = options.impLevel || RPC_C_IMP_LEVEL.IMPERSONATE;
        const capabilities = options.capabilities || EOAC.DYNAMIC_CLOAKING;
        
        const result = this.ole32.CoInitializeSecurity(
            null,       // PSECURITY_DESCRIPTOR
            -1,         // cAuthSvc (default)
            null,       // asAuthSvc
            null,       // pReserved1
            authnLevel,
            impLevel,
            null,       // pAuthList
            capabilities,
            null        // pReserved3
        );
        
        if (result !== 0) {
            throw new Error(`CoInitializeSecurity failed: 0x${result.toString(16)}`);
        }
        
        return true;
    }
    
    setProxyBlanket(proxy, options = {}) {
        const authnSvc = options.authnSvc || 0; // RPC_C_AUTHN_WINNT
        const authzSvc = options.authzSvc || 0; // RPC_C_AUTHZ_NONE
        const authnLevel = options.authnLevel || RPC_C_AUTHN_LEVEL.PKT;
        const impLevel = options.impLevel || RPC_C_IMP_LEVEL.IMPERSONATE;
        const capabilities = options.capabilities || EOAC.NONE;
        
        const result = this.ole32.CoSetProxyBlanket(
            proxy,
            authnSvc,
            authzSvc,
            null,       // pServerPrincName
            authnLevel,
            impLevel,
            null,       // pAuthInfo
            capabilities
        );
        
        return result === 0;
    }
}

// Usage in OpenClaw
const comSecurity = new ComSecurity();
comSecurity.initializeSecurity({
    authnLevel: RPC_C_AUTHN_LEVEL.PKT_PRIVACY,
    impLevel: RPC_C_IMP_LEVEL.IMPERSONATE,
    capabilities: EOAC.DYNAMIC_CLOAKING | EOAC.STATIC_CLOAKING
});
```

### 8.3 COM Marshaling

```javascript
// COM Marshaling for Cross-Thread/Cross-Process
const ole32 = ffi.Library('ole32', {
    'CoMarshalInterThreadInterfaceInStream': ['long', [
        ref.refType(ref.types.void),  // riid
        'pointer',                      // pUnk
        'pointer'                       // ppStm
    ]],
    'CoGetInterfaceAndReleaseStream': ['long', [
        'pointer',                      // pStm
        ref.refType(ref.types.void),  // riid
        'pointer'                       // ppv
    ]],
    'CoMarshalInterface': ['long', [
        'pointer',                      // pStm
        ref.refType(ref.types.void),  // riid
        'pointer',                      // pUnk
        'uint',                         // dwDestContext
        'pointer',                      // pvDestContext
        'uint',                         // mshlflags
        'pointer'                       // pclsid
    ]],
    'CoUnmarshalInterface': ['long', [
        'pointer',                      // pStm
        ref.refType(ref.types.void),  // riid
        'pointer'                       // ppv
    ]]
});

// Marshaling flags
const MSHCTX = {
    INPROC: 0,
    LOCAL: 1,
    NOSHAREDMEM: 2,
    DIFFERENTMACHINE: 3,
    CROSSCTX: 4
};

const MSHLFLAGS = {
    NORMAL: 0,
    TABLESTRONG: 1,
    TABLEWEAK: 2,
    NOPING: 4
};

class ComMarshaling {
    marshalInterfaceToStream(object, interfaceId, destContext = MSHCTX.INPROC) {
        // Create memory stream
        const shlwapi = ffi.Library('shlwapi', {
            'SHCreateMemStream': ['pointer', ['pointer', 'uint']]
        });
        
        const stream = shlwapi.SHCreateMemStream(null, 0);
        
        const result = ole32.CoMarshalInterface(
            stream,
            interfaceId,
            object,
            destContext,
            null,
            MSHLFLAGS.NORMAL,
            null
        );
        
        if (result !== 0) {
            throw new Error(`CoMarshalInterface failed: 0x${result.toString(16)}`);
        }
        
        return stream;
    }
    
    unmarshalInterfaceFromStream(stream, interfaceId) {
        const ptr = ref.alloc('pointer');
        
        const result = ole32.CoUnmarshalInterface(
            stream,
            interfaceId,
            ptr
        );
        
        if (result !== 0) {
            throw new Error(`CoUnmarshalInterface failed: 0x${result.toString(16)}`);
        }
        
        return ptr.deref();
    }
}
```

### 8.4 Apartment Threading Model

```javascript
// COM Apartment Threading
const COINIT = {
    APARTMENTTHREADED: 0x2,
    MULTITHREADED: 0x0,
    DISABLE_OLE1DDE: 0x4,
    SPEED_OVER_MEMORY: 0x8
};

class ComApartment {
    constructor() {
        this.ole32 = ffi.Library('ole32', {
            'CoInitializeEx': ['long', ['pointer', 'uint32']],
            'CoUninitialize': ['void', []],
            'CoGetApartmentType': ['long', ['pointer', 'pointer']]
        });
        
        this.initialized = false;
    }
    
    initialize(apartmentType = COINIT.APARTMENTTHREADED) {
        const result = this.ole32.CoInitializeEx(null, apartmentType);
        
        if (result === 0 || result === 1) { // S_OK or S_FALSE (already initialized)
            this.initialized = true;
            return true;
        }
        
        throw new Error(`CoInitializeEx failed: 0x${result.toString(16)}`);
    }
    
    uninitialize() {
        if (this.initialized) {
            this.ole32.CoUninitialize();
            this.initialized = false;
        }
    }
    
    getApartmentType() {
        const aptType = ref.alloc('int');
        const aptQualifier = ref.alloc('int');
        
        const result = this.ole32.CoGetApartmentType(aptType, aptQualifier);
        
        if (result !== 0) {
            throw new Error(`CoGetApartmentType failed: 0x${result.toString(16)}`);
        }
        
        const types = ['STA', 'MTA', 'NTA', 'MAINSTA'];
        return {
            type: types[aptType.deref()] || 'UNKNOWN',
            qualifier: aptQualifier.deref()
        };
    }
}

// Thread-safe COM wrapper for Node.js
class ThreadSafeCom {
    constructor() {
        this.workQueue = [];
        this.isProcessing = false;
    }
    
    async executeOnSTA(comOperation) {
        // Use Node.js worker threads for STA
        const { Worker } = require('worker_threads');
        
        return new Promise((resolve, reject) => {
            const worker = new Worker(`
                const { parentPort } = require('worker_threads');
                const edge = require('edge-js');
                
                // Initialize COM on STA
                const initializeSTA = edge.func({
                    source: `
                        using System;
                        using System.Runtime.InteropServices;
                        
                        public class STAInitializer {
                            public object Invoke(object input) {
                                // COM automatically initializes as STA in single-threaded apartment
                                return null;
                            }
                        }
                    `
                });
                
                parentPort.on('message', async (task) => {
                    try {
                        const result = await task.operation();
                        parentPort.postMessage({ success: true, result });
                    } catch (error) {
                        parentPort.postMessage({ success: false, error: error.message });
                    }
                });
            `, { eval: true });
            
            worker.on('message', (message) => {
                worker.terminate();
                if (message.success) {
                    resolve(message.result);
                } else {
                    reject(new Error(message.error));
                }
            });
            
            worker.on('error', reject);
            worker.postMessage({ operation: comOperation.toString() });
        });
    }
}
```

---

## 9. Async Pattern Handling for WinRT

### 9.1 WinRT Async Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WinRT Async Pattern Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  WinRT Async Interfaces:                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │  IAsyncAction   │  │  IAsyncOperation│  │ IAsyncActionWith│              │
│  │                 │  │    <TResult>    │  │   Progress<T>   │              │
│  │  - No result    │  │  - With result  │  │  - With progress│              │
│  │  - No progress  │  │  - No progress  │  │  - No result    │              │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘              │
│           │                    │                    │                        │
│           └────────────────────┼────────────────────┘                        │
│                                │                                             │
│  ┌─────────────────────────────▼─────────────────────────────┐              │
│  │                      IAsyncInfo                            │              │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐│              │
│  │  │   Status    │  │   ErrorCode │  │   Cancel()          ││              │
│  │  │   Id        │  │   Completed │  │   Close()           ││              │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘│              │
│  └────────────────────────────────────────────────────────────┘              │
│                                                                              │
│  AsyncStatus Enum:                                                           │
│  • Started = 0, Completed = 1, Canceled = 2, Error = 3                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Async Handler Implementation

```javascript
// WinRT Async Pattern Wrapper for Node.js
class WinRTAsyncHandler {
    constructor() {
        this.AsyncStatus = {
            Started: 0,
            Completed: 1,
            Canceled: 2,
            Error: 3
        };
    }
    
    // Convert WinRT async to Promise
    async asPromise(asyncOperation) {
        return new Promise((resolve, reject) => {
            // Check if already completed
            if (asyncOperation.status === this.AsyncStatus.Completed) {
                resolve(asyncOperation.getResults());
                return;
            }
            
            if (asyncOperation.status === this.AsyncStatus.Error) {
                reject(new Error(asyncOperation.errorCode));
                return;
            }
            
            // Set up completion handler
            asyncOperation.completed = (operation, status) => {
                try {
                    if (status === this.AsyncStatus.Completed) {
                        const results = operation.getResults();
                        resolve(results);
                    } else if (status === this.AsyncStatus.Error) {
                        reject(new Error(operation.errorCode));
                    } else if (status === this.AsyncStatus.Canceled) {
                        reject(new Error('Operation canceled'));
                    }
                } catch (error) {
                    reject(error);
                }
            };
            
            // Set up error handler
            asyncOperation.error = (operation, error) => {
                reject(new Error(error));
            };
        });
    }
    
    // Async with progress tracking
    async asPromiseWithProgress(asyncOperation, onProgress) {
        return new Promise((resolve, reject) => {
            // Set up progress handler
            if (onProgress && asyncOperation.progress) {
                asyncOperation.progress = (operation, progress) => {
                    onProgress({
                        progress: progress,
                        operation: operation
                    });
                };
            }
            
            // Set up completion handler
            asyncOperation.completed = (operation, status) => {
                try {
                    if (status === this.AsyncStatus.Completed) {
                        resolve(operation.getResults());
                    } else if (status === this.AsyncStatus.Error) {
                        reject(new Error(operation.errorCode));
                    } else if (status === this.AsyncStatus.Canceled) {
                        reject(new Error('Operation canceled'));
                    }
                } catch (error) {
                    reject(error);
                }
            };
        });
    }
    
    // Cancel async operation
    cancel(asyncOperation) {
        if (asyncOperation && asyncOperation.cancel) {
            asyncOperation.cancel();
        }
    }
    
    // Close async operation (release resources)
    close(asyncOperation) {
        if (asyncOperation && asyncOperation.close) {
            asyncOperation.close();
        }
    }
    
    // Create cancellable promise
    createCancellable(asyncFactory) {
        let asyncOp = null;
        
        const promise = new Promise(async (resolve, reject) => {
            try {
                asyncOp = await asyncFactory();
                const result = await this.asPromise(asyncOp);
                resolve(result);
            } catch (error) {
                reject(error);
            }
        });
        
        promise.cancel = () => {
            if (asyncOp) {
                this.cancel(asyncOp);
            }
        };
        
        return promise;
    }
}

// Usage with NodeRT
const asyncHandler = new WinRTAsyncHandler();

// Example: File operation with progress
const { FileIO } = require('@nodert-win10-rs4/windows.storage');

async function readFileWithProgress(file) {
    const readOp = FileIO.readTextAsync(file);
    
    return await asyncHandler.asPromiseWithProgress(
        readOp,
        (progressInfo) => {
            console.log(`Progress: ${progressInfo.progress}`);
        }
    );
}

// Example: Cancellable operation
const cancellableOp = asyncHandler.createCancellable(async () => {
    const { Geolocator } = require('@nodert-win10-rs4/windows.devices.geolocation');
    const locator = new Geolocator();
    return locator.getGeopositionAsync();
});

// Cancel after 5 seconds
setTimeout(() => cancellableOp.cancel(), 5000);

try {
    const position = await cancellableOp;
    console.log(`Location: ${position.coordinate.latitude}, ${position.coordinate.longitude}`);
} catch (error) {
    console.error('Operation failed or was cancelled:', error.message);
}
```

### 9.3 Async Pattern Utilities

```javascript
// Async utilities for WinRT operations
class WinRTAsyncUtils {
    // Timeout wrapper for async operations
    static withTimeout(asyncOperation, timeoutMs) {
        return Promise.race([
            asyncOperation,
            new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Operation timed out')), timeoutMs)
            )
        ]);
    }
    
    // Retry wrapper for async operations
    static async withRetry(asyncFn, options = {}) {
        const maxRetries = options.maxRetries || 3;
        const delay = options.delay || 1000;
        const backoff = options.backoff || 2;
        
        let lastError;
        let currentDelay = delay;
        
        for (let attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                return await asyncFn();
            } catch (error) {
                lastError = error;
                
                if (attempt < maxRetries) {
                    await new Promise(resolve => setTimeout(resolve, currentDelay));
                    currentDelay *= backoff;
                }
            }
        }
        
        throw lastError;
    }
    
    // Sequential async operations
    static async sequence(asyncFns) {
        const results = [];
        for (const fn of asyncFns) {
            results.push(await fn());
        }
        return results;
    }
    
    // Parallel async operations with concurrency limit
    static async parallel(asyncFns, concurrency = 5) {
        const results = [];
        const executing = [];
        
        for (const [index, fn] of asyncFns.entries()) {
            const promise = fn().then(result => ({ index, result }));
            results.push(promise);
            
            if (asyncFns.length >= concurrency) {
                executing.push(promise);
                if (executing.length >= concurrency) {
                    await Promise.race(executing);
                }
            }
        }
        
        const settled = await Promise.all(results);
        return settled.sort((a, b) => a.index - b.index).map(r => r.result);
    }
    
    // Debounce for async operations
    static debounce(asyncFn, wait) {
        let timeout;
        let pendingPromise;
        
        return function(...args) {
            clearTimeout(timeout);
            
            if (!pendingPromise) {
                pendingPromise = new Promise((resolve, reject) => {
                    timeout = setTimeout(async () => {
                        try {
                            const result = await asyncFn.apply(this, args);
                            resolve(result);
                        } catch (error) {
                            reject(error);
                        } finally {
                            pendingPromise = null;
                        }
                    }, wait);
                });
            }
            
            return pendingPromise;
        };
    }
}
```

---

## 10. Implementation Examples

### 10.1 Complete OpenClaw Windows Integration Module

```javascript
// openclaw-windows.js - Main Integration Module
const edge = require('edge-js');
const path = require('path');

class OpenClawWindowsIntegration {
    constructor(config = {}) {
        this.config = {
            comSecurityLevel: config.comSecurityLevel || 'standard',
            enableWinRT: config.enableWinRT !== false,
            enableCOM: config.enableCOM !== false,
            apartmentThreading: config.apartmentThreading || 'STA',
            ...config
        };
        
        this.modules = {};
        this.initialized = false;
    }
    
    async initialize() {
        if (this.initialized) return;
        
        // Initialize COM security
        if (this.config.enableCOM) {
            await this.initializeComSecurity();
        }
        
        // Initialize WinRT
        if (this.config.enableWinRT) {
            await this.initializeWinRT();
        }
        
        // Load modules
        this.modules = {
            tts: new WindowsTTS(),
            stt: new WindowsSTT(),
            storage: new StorageManager(),
            launcher: new SystemLauncher(),
            shell: new ShellAutomation(),
            wsh: new WScriptAutomation(),
            wmi: new WmiAutomation(),
            appData: new AppDataManager()
        };
        
        // Initialize all modules
        for (const [name, module] of Object.entries(this.modules)) {
            if (module.initialize) {
                await module.initialize();
            }
        }
        
        this.initialized = true;
        console.log('OpenClaw Windows Integration initialized');
    }
    
    async initializeComSecurity() {
        const initSecurity = edge.func({
            source: `
                using System;
                using System.Runtime.InteropServices;
                
                public class ComSecurityInitializer {
                    [DllImport("ole32.dll")]
                    static extern int CoInitializeSecurity(
                        IntPtr pSecDesc, int cAuthSvc, IntPtr asAuthSvc,
                        IntPtr pReserved1, uint dwAuthnLevel, uint dwImpLevel,
                        IntPtr pAuthList, uint dwCapabilities, IntPtr pReserved3);
                    
                    public object Invoke(object input) {
                        // RPC_C_AUTHN_LEVEL_PKT = 4
                        // RPC_C_IMP_LEVEL_IMPERSONATE = 3
                        // EOAC_DYNAMIC_CLOAKING = 0x40
                        int result = CoInitializeSecurity(
                            IntPtr.Zero, -1, IntPtr.Zero, IntPtr.Zero,
                            4, 3, IntPtr.Zero, 0x40, IntPtr.Zero);
                        
                        return new { Result = result, Success = result == 0 };
                    }
                }
            `
        });
        
        const result = await initSecurity(null);
        if (!result.Success) {
            console.warn(`COM Security initialization returned: 0x${result.Result.toString(16)}`);
        }
    }
    
    async initializeWinRT() {
        // WinRT is initialized automatically by NodeRT
        // But we can verify it's working
        try {
            const { ApplicationData } = require('@nodert-win10-rs4/windows.storage');
            const appData = ApplicationData.current;
            console.log(`WinRT initialized. App version: ${appData.version}`);
        } catch (error) {
            console.error('WinRT initialization failed:', error);
            throw error;
        }
    }
    
    // Module accessors
    get tts() { return this.modules.tts; }
    get stt() { return this.modules.stt; }
    get storage() { return this.modules.storage; }
    get launcher() { return this.modules.launcher; }
    get shell() { return this.modules.shell; }
    get wsh() { return this.modules.wsh; }
    get wmi() { return this.modules.wmi; }
    get appData() { return this.modules.appData; }
    
    // High-level agent operations
    async speak(message, options = {}) {
        return await this.tts.speak(message, options);
    }
    
    async listen(options = {}) {
        return await this.stt.recognizeOnce(options);
    }
    
    async executeCommand(command, options = {}) {
        // Use WSH for command execution
        return await this.wsh.run(command, options.windowStyle, options.wait);
    }
    
    async getSystemInfo() {
        return await this.wmi.getSystemInfo();
    }
    
    async launchApplication(appPath, args = '') {
        return await this.shell.shellExecute(appPath, args);
    }
    
    async showNotification(title, message) {
        // Use Windows.UI.Notifications
        const { ToastNotificationManager, ToastTemplateType } = 
            require('@nodert-win10-rs4/windows.ui.notifications');
        
        const template = ToastNotificationManager.getTemplateContent(
            ToastTemplateType.toastText02
        );
        
        const textNodes = template.getElementsByTagName('text');
        textNodes.getAt(0).appendChild(template.createTextNode(title));
        textNodes.getAt(1).appendChild(template.createTextNode(message));
        
        const toast = new (require('@nodert-win10-rs4/windows.ui.notifications')).ToastNotification(template);
        const notifier = ToastNotificationManager.createToastNotifier();
        notifier.show(toast);
    }
    
    dispose() {
        // Clean up all modules
        for (const [name, module] of Object.entries(this.modules)) {
            if (module.dispose) {
                module.dispose();
            }
        }
        
        this.initialized = false;
    }
}

// Export singleton instance
module.exports = new OpenClawWindowsIntegration();
module.exports.OpenClawWindowsIntegration = OpenClawWindowsIntegration;
```

### 10.2 Background Task Registration

```javascript
// background-tasks.js - 24/7 Background Operation
const { 
    BackgroundTaskBuilder,
    SystemTrigger,
    SystemTriggerType,
    TimeTrigger,
    MaintenanceTrigger,
    BackgroundTaskRegistration,
    BackgroundExecutionManager,
    BackgroundAccessStatus
} = require('@nodert-win10-rs4/windows.applicationmodel.background');

class OpenClawBackgroundTasks {
    constructor() {
        this.registeredTasks = new Map();
    }
    
    async requestAccess() {
        const status = await BackgroundExecutionManager.requestAccessAsync();
        return status !== BackgroundAccessStatus.denied &&
               status !== BackgroundAccessStatus.deniedBySystemPolicy &&
               status !== BackgroundAccessStatus.deniedByUser;
    }
    
    async registerHeartbeatTask(intervalMinutes = 15) {
        const taskName = 'OpenClawHeartbeat';
        
        // Unregister existing
        this.unregisterTask(taskName);
        
        const builder = new BackgroundTaskBuilder();
        builder.name = taskName;
        builder.taskEntryPoint = 'OpenClaw.BackgroundTasks.HeartbeatTask';
        
        // Use time trigger
        const trigger = new TimeTrigger(intervalMinutes, false);
        builder.setTrigger(trigger);
        
        // Add conditions
        const { SystemCondition, SystemConditionType } = 
            require('@nodert-win10-rs4/windows.applicationmodel.background');
        builder.addCondition(new SystemCondition(SystemConditionType.internetAvailable));
        
        const registration = builder.register();
        this.registeredTasks.set(taskName, registration);
        
        console.log(`Heartbeat task registered: ${taskName}`);
        return registration;
    }
    
    async registerSystemEventTask(eventType) {
        const taskName = `OpenClawSystemEvent_${eventType}`;
        
        this.unregisterTask(taskName);
        
        const builder = new BackgroundTaskBuilder();
        builder.name = taskName;
        builder.taskEntryPoint = 'OpenClaw.BackgroundTasks.SystemEventTask';
        
        const trigger = new SystemTrigger(SystemTriggerType[eventType], false);
        builder.setTrigger(trigger);
        
        const registration = builder.register();
        this.registeredTasks.set(taskName, registration);
        
        return registration;
    }
    
    async registerMaintenanceTask(frequencyMinutes) {
        const taskName = 'OpenClawMaintenance';
        
        this.unregisterTask(taskName);
        
        const builder = new BackgroundTaskBuilder();
        builder.name = taskName;
        builder.taskEntryPoint = 'OpenClaw.BackgroundTasks.MaintenanceTask';
        
        const trigger = new MaintenanceTrigger(frequencyMinutes, false);
        builder.setTrigger(trigger);
        
        const registration = builder.register();
        this.registeredTasks.set(taskName, registration);
        
        return registration;
    }
    
    unregisterTask(taskName) {
        const existing = BackgroundTaskRegistration.allTasks;
        for (let i = 0; i < existing.size; i++) {
            const task = existing.getAt(i);
            if (task.value.name === taskName) {
                task.value.unregister(true);
                console.log(`Unregistered task: ${taskName}`);
                break;
            }
        }
        this.registeredTasks.delete(taskName);
    }
    
    unregisterAllTasks() {
        for (const taskName of this.registeredTasks.keys()) {
            this.unregisterTask(taskName);
        }
    }
    
    getRegisteredTasks() {
        const tasks = [];
        const allTasks = BackgroundTaskRegistration.allTasks;
        for (let i = 0; i < allTasks.size; i++) {
            const task = allTasks.getAt(i);
            tasks.push({
                name: task.value.name,
                taskId: task.value.taskId
            });
        }
        return tasks;
    }
}

module.exports = { OpenClawBackgroundTasks };
```

### 10.3 Package Dependencies

```json
{
  "name": "openclaw-windows",
  "version": "1.0.0",
  "description": "Windows 10 COM/WinRT Integration for OpenClaw AI Agent",
  "main": "openclaw-windows.js",
  "scripts": {
    "build": "node-gyp rebuild",
    "test": "jest",
    "package": "electron-builder"
  },
  "dependencies": {
    "edge-js": "^19.3.0",
    "ffi-napi": "^4.0.3",
    "ref-napi": "^3.0.3",
    "ref-array-napi": "^1.2.2",
    "ref-struct-napi": "^1.1.1",
    "@echogarden/windows-media-tts": "^1.0.0",
    "@nodert-win10-rs4/windows.media.speechsynthesis": "^0.4.4",
    "@nodert-win10-rs4/windows.media.speechrecognition": "^0.4.4",
    "@nodert-win10-rs4/windows.storage": "^0.4.4",
    "@nodert-win10-rs4/windows.storage.pickers": "^0.4.4",
    "@nodert-win10-rs4/windows.system": "^0.4.4",
    "@nodert-win10-rs4/windows.devices.geolocation": "^0.4.4",
    "@nodert-win10-rs4/windows.ui.notifications": "^0.4.4",
    "@nodert-win10-rs4/windows.applicationmodel.background": "^0.4.4",
    "@nodert-win10-rs4/windows.networking.connectivity": "^0.4.4",
    "@nodert-win10-rs4/nodert-streams": "^0.4.4"
  },
  "devDependencies": {
    "node-gyp": "^9.3.1",
    "jest": "^29.5.0",
    "electron": "^24.0.0",
    "electron-builder": "^24.0.0"
  },
  "engines": {
    "node": ">=16.0.0"
  },
  "os": ["win32"]
}
```

---

## Appendix A: COM Object Reference

| ProgID | CLSID | Description |
|--------|-------|-------------|
| Shell.Application | {13709620-C279-11CE-A49E-444553540000} | Windows Shell automation |
| WScript.Shell | {72C24DD5-D70A-438B-8A42-98424B88AFB8} | Windows Script Host shell |
| Scripting.FileSystemObject | {0D43FE01-F093-11CF-8940-00A0C9054228} | File system operations |
| WScript.Network | {093FF999-1EA0-4079-9525-9614C3504B74} | Network operations |
| WMI.WinMgmt | {8BC3F05E-D86B-11D0-A075-00C04FB68820} | Windows Management |
| SAPI.SpVoice | {96749377-3391-11D2-9EE3-00C04F797396} | Speech synthesis (SAPI 5) |
| InternetExplorer.Application | {0002DF01-0000-0000-C000-000000000046} | Internet Explorer |
| Excel.Application | {00024500-0000-0000-C000-000000000046} | Microsoft Excel |
| Word.Application | {00020970-0000-0000-C000-000000000046} | Microsoft Word |
| Outlook.Application | {0006F03A-0000-0000-C000-000000000046} | Microsoft Outlook |

## Appendix B: WinRT Namespace Availability

| Namespace | Desktop Support | Package Required |
|-----------|-----------------|------------------|
| Windows.Media.SpeechSynthesis | Full | No |
| Windows.Media.SpeechRecognition | Full | No |
| Windows.Media.Playback | Full | No |
| Windows.Storage | Full | No |
| Windows.Storage.Pickers | Partial | No (IInitializeWithWindow) |
| Windows.System | Partial | No (IInitializeWithWindow) |
| Windows.System.Diagnostics | Full | No |
| Windows.Networking | Full | No |
| Windows.Devices | Full | No |
| Windows.ApplicationModel | Partial | Yes |
| Windows.UI.Notifications | Partial | Yes |
| Windows.Data.Json | Full | No |
| Windows.Data.Xml | Full | No |
| Windows.Security.Credentials | Full | No |
| Windows.Globalization | Full | No |

## Appendix C: Error Handling

```javascript
// Common COM/WinRT Error Codes
const COM_ERRORS = {
    S_OK: 0x00000000,
    S_FALSE: 0x00000001,
    E_UNEXPECTED: 0x8000FFFF,
    E_NOTIMPL: 0x80004001,
    E_OUTOFMEMORY: 0x8007000E,
    E_INVALIDARG: 0x80070057,
    E_NOINTERFACE: 0x80004002,
    E_POINTER: 0x80004003,
    E_HANDLE: 0x80070006,
    E_ABORT: 0x80004004,
    E_FAIL: 0x80004005,
    E_ACCESSDENIED: 0x80070005,
    E_PENDING: 0x8000000A,
    CO_E_NOTINITIALIZED: 0x800401F0,
    CO_E_ALREADYINITIALIZED: 0x800401F1,
    REGDB_E_CLASSNOTREG: 0x80040154,
    CLASS_E_NOAGGREGATION: 0x80040110,
    E_NO_AGGREGATION: 0x80040110,
    RPC_E_CHANGED_MODE: 0x80010106,
    RPC_E_SERVERFAULT: 0x80010105
};

function getErrorMessage(hr) {
    const code = hr < 0 ? hr >>> 0 : hr; // Convert to unsigned
    for (const [name, value] of Object.entries(COM_ERRORS)) {
        if (value === code) return name;
    }
    return `Unknown Error (0x${code.toString(16).toUpperCase()})`;
}
```

---

*Document Version: 1.0*
*Last Updated: 2024*
*Author: OpenClaw Windows Integration Team*
