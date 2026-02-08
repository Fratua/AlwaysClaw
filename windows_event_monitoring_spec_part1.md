# Windows Event Log and System Monitoring Integration Specification
## OpenClaw-Inspired AI Agent System for Windows 10

---

## Executive Summary

This document provides comprehensive technical specifications for integrating Windows Event Log, ETW (Event Tracing for Windows), and system monitoring capabilities into a Windows 10 AI agent system. The design enables real-time system awareness, event-driven agentic behavior, and comprehensive telemetry collection.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Windows Event Log APIs](#2-windows-event-log-apis)
3. [ETW Integration](#3-etw-integration)
4. [Event Subscription and Notification](#4-event-subscription-and-notification)
5. [Performance Counter Access](#5-performance-counter-access)
6. [System Notification Handling](#6-system-notification-handling)
7. [Custom Event Source Creation](#7-custom-event-source-creation)
8. [Event Filtering and Querying](#8-event-filtering-and-querying)
9. [Real-Time Event Monitoring](#9-real-time-event-monitoring)
10. [Integration with Agent Loops](#10-integration-with-agent-loops)
11. [Security Considerations](#11-security-considerations)
12. [Implementation Reference](#12-implementation-reference)

---

## 1. Architecture Overview

### 1.1 System Architecture

```
AI AGENT SYSTEM CORE
+------------------+------------------+------------------+
|   Soul Engine    | Identity Service |   User Mgmt      |
+------------------+------------------+------------------+
          |                |                |
          +----------------+----------------+
                         |
              AGENT ORCHESTRATOR (15 Loops)
                         |
WINDOWS MONITORING LAYER |
+------------------------+------------------------+
| Windows Event Log API  | ETW Subsystem          |
| - ReadEventLog         | - Event Providers      |
| - ReportEvent          | - Event Consumers      |
| - NotifyChange         | - Session Controllers  |
+------------------------+------------------------+
| Performance Counters (PDH API)                  |
| - PdhOpenQuery, PdhAddCounter, PdhCollect      |
+------------------------+------------------------+
          |
EVENT PROCESSING ENGINE
- Event Filtering
- Pattern Recognition
- Correlation Analysis
- Notification Dispatch
          |
AGENT NOTIFICATION BUS
(Triggers Agentic Behavior)
```

### 1.2 Component Interaction Flow

```
Windows System      Monitoring Layer      Agent System
--------------      ----------------      ------------
     |                     |                     |
[Event Source] ---> [Event Capture] ---> [Agent Handler]
     |                     |                     |
     |              [Filter Engine] ---> [Action Trigger]
     |                     |                     |
     |              [Notify Queue] ---> [Agent Loop]
```

---

## 2. Windows Event Log APIs

### 2.1 Legacy Event Logging API

| Function | Purpose | Use Case |
|----------|---------|----------|
| `OpenEventLog` | Opens handle to event log | Reading events |
| `RegisterEventSource` | Registers as event source | Writing events |
| `ReadEventLog` | Reads events from log | Event retrieval |
| `ReportEvent` | Writes event to log | Custom logging |
| `NotifyChangeEventLog` | Change notification | Real-time monitoring |
| `GetOldestEventLogRecord` | Gets oldest record | Pagination |
| `GetNumberOfEventLogRecords` | Gets record count | Progress |

### 2.2 Modern Event Log API (wevtapi.dll)

| Function | Purpose | Use Case |
|----------|---------|----------|
| `EvtQuery` | Queries events | Structured retrieval |
| `EvtSubscribe` | Subscribes to events | Real-time monitoring |
| `EvtNext` | Gets next event | Result iteration |
| `EvtRender` | Renders to XML/text | Formatting |
| `EvtCreateBookmark` | Creates bookmark | Resume position |

### 2.3 XPath Query Examples

```cpp
// Query events from last hour
*[System[TimeCreated[timediff(@SystemTime) <= 3600000]]]

// Query error events
*[System[Level=1 or Level=2]]

// Query specific event IDs
*[System[(EventID=4624 or EventID=4625)]]

// Query by provider
*[System[Provider[@Name='MyApplication']]]
```

---

## 3. ETW Integration

### 3.1 ETW Architecture

```
CONTROLLER
- Start/Stop Tracing Sessions
- Enable/Disable Providers
- Configure Buffer Pool
      |
+-----+-----+-----+
|     |     |     |
Kernel User  MOF
Provider Provider Provider
|     |     |
+-----+-----+
      |
ETW SESSION
- In-Memory Buffers
- Log File (.etl)
- Real-time Delivery
      |
CONSUMER
- Process Events
- Real-time Processing
- Event Parsing (TDH)
```

### 3.2 ETW Core APIs

| Component | Function | Purpose |
|-----------|----------|---------|
| Controller | `StartTrace` | Starts tracing session |
| Controller | `EnableTraceEx2` | Enables provider |
| Controller | `StopTrace` | Stops session |
| Provider | `EventRegister` | Registers provider |
| Provider | `EventWrite` | Writes event |
| Consumer | `OpenTrace` | Opens log file |
| Consumer | `ProcessTrace` | Processes events |
| TDH | `TdhGetEventInformation` | Gets metadata |
| TDH | `TdhFormatProperty` | Formats property |

### 3.3 Important ETW Provider GUIDs

```cpp
// Kernel Providers
SystemTraceControlGuid = {0x9e814aad, 0x3204, 0x11d2, ...}
ProcessGuid = {0x3d6fa8d1, 0xfe05, 0x11d0, ...}
ThreadGuid = {0x3d6fa8d0, 0xfe05, 0x11d0, ...}
ImageLoadGuid = {0x2cb15d1d, 0x5fc1, 0x11d2, ...}
FileIoGuid = {0x90cbdc39, 0x4a3e, 0x11d1, ...}
DiskIoGuid = {0x3d6fa8d4, 0xfe05, 0x11d0, ...}
TcpIpGuid = {0x9a280ac0, 0xc8d0, 0x11d1, ...}
RegistryGuid = {0xae53722e, 0xc863, 0x11d2, ...}

// Security Providers
MicrosoftWindowsSecurityAuditing = {0x54849625, ...}
MicrosoftWindowsKernelProcess = {0x22fb2cd6, ...}
MicrosoftWindowsPowerShell = {0xa0c1853b, ...}
```

---

## 4. Event Subscription and Notification

### 4.1 Subscription Architecture

```
EVENT SOURCES
+----------+----------+----------+----------+
| Windows  |   ETW    |   WMI    |  Custom  |
| EventLog | Provider |  Events  | Sources  |
+-----+----+----+-----+-----+----+-----+----+
      |         |         |         |
      +---------+---------+---------+
                |
      SUBSCRIPTION MANAGER
      +---------------------+
      | Subscription Registry|
      | - Sub ID, Event Type |
      | - Filter, Callback   |
      +---------------------+
      | Event Dispatcher     |
      | - Route, Filter,     |
      |   Lifecycle          |
      +---------------------+
                |
      NOTIFICATION SYSTEM
      +----------+----------+----------+
      | Synchronous| Asynchronous| Buffered|
      | (Immediate)| (Queued)    | (Batch) |
      +-----+------+------+------+----+---+
            |             |            |
            +-------------+------------+
                          |
                  AGENT HANDLERS
```

### 4.2 Event Types

```cpp
enum class EventSourceType {
    WindowsEventLog,
    ETW,
    WMI,
    FileSystem,
    PerformanceCounter,
    Custom
};

struct SystemEvent {
    std::wstring eventId;
    EventSourceType sourceType;
    std::chrono::system_clock::time_point timestamp;
    std::wstring sourceName;
    std::wstring message;
    std::map<std::wstring, std::wstring> properties;
    int severity;  // 0=Info, 1=Warning, 2=Error, 3=Critical
    std::vector<BYTE> rawData;
};
```

### 4.3 Filter Functions

```cpp
namespace EventFilters {
    // Severity filters
    auto SeverityAtLeast(int minSeverity);
    auto SeverityExactly(int severity);
    
    // Source name filter
    auto SourceNameIs(const std::wstring& name);
    
    // Event ID filter
    auto EventIdIn(const std::vector<std::wstring>& ids);
    
    // Time-based filters
    auto WithinLastMinutes(int minutes);
    
    // Property filter
    auto PropertyEquals(const std::wstring& key, const std::wstring& value);
    
    // Composite filters
    auto And(filter_a, filter_b);
    auto Or(filter_a, filter_b);
}
```

---

## 5. Performance Counter Access

### 5.1 PDH API Flow

```
PERFORMANCE OBJECTS
+----------+----------+----------+----------+
| Processor|  Memory  |  Process |  Disk    |
+-----+----+----+-----+-----+----+-----+----+
      |         |         |         |
      +---------+---------+---------+
                |
         PDH QUERY API
         1. PdhOpenQuery()
         2. PdhAddCounter()
         3. PdhCollectQueryData()
         4. PdhGetFormattedCounterValue()
         5. PdhCloseQuery()
```

### 5.2 Common Counter Paths

```cpp
// Processor counters
PROCESSOR_TIME_TOTAL = L"\\Processor(_Total)\\% Processor Time"
PROCESSOR_USER_TIME = L"\\Processor(_Total)\\% User Time"

// Memory counters
MEMORY_AVAILABLE_BYTES = L"\\Memory\\Available Bytes"
MEMORY_COMMITTED_BYTES = L"\\Memory\\Committed Bytes"
MEMORY_PAGES_PER_SEC = L"\\Memory\\Pages/sec"

// Process counters
PROCESS_CPU_TIME(process) = L"\\Process(process)\\% Processor Time"
PROCESS_WORKING_SET(process) = L"\\Process(process)\\Working Set"
PROCESS_PRIVATE_BYTES(process) = L"\\Process(process)\\Private Bytes"

// Disk counters
DISK_QUEUE_LENGTH = L"\\PhysicalDisk(_Total)\\Avg. Disk Queue Length"
DISK_READ_BYTES_PER_SEC = L"\\PhysicalDisk(_Total)\\Disk Read Bytes/sec"

// Network counters
NETWORK_BYTES_SENT = L"\\Network Interface(*)\\Bytes Sent/sec"
NETWORK_BYTES_RECEIVED = L"\\Network Interface(*)\\Bytes Received/sec"

// System counters
SYSTEM_UPTIME = L"\\System\\System Up Time"
SYSTEM_PROCESSES = L"\\System\\Processes"
SYSTEM_CONTEXT_SWITCHES = L"\\System\\Context Switches/sec"
```

---

## 6. System Notification Handling

### 6.1 Notification Sources

| Source | Messages | Description |
|--------|----------|-------------|
| Power | PBT_APMSUSPEND, PBT_APMRESUMESUSPEND | Power events |
| Session | WTS_SESSION_LOGON, WTS_SESSION_LOCK | Session events |
| Device | DBT_DEVICEARRIVAL, DBT_DEVICEREMOVE | Device events |
| System | WM_QUERYENDSESSION, WM_TIMECHANGE | System events |

### 6.2 Power Events

```cpp
PBT_APMQUERYSUSPEND      // System about to suspend
PBT_APMSUSPEND           // System suspending
PBT_APMRESUMESUSPEND     // System resumed
PBT_APMPOWERSTATUSCHANGE // Power status changed
PBT_POWERSETTINGCHANGE   // Power setting changed
```

### 6.3 Session Events

```cpp
WTS_CONSOLE_CONNECT      // Console connected
WTS_CONSOLE_DISCONNECT   // Console disconnected
WTS_SESSION_LOGON        // Session logon
WTS_SESSION_LOGOFF       // Session logoff
WTS_SESSION_LOCK         // Session locked
WTS_SESSION_UNLOCK       // Session unlocked
WTS_REMOTE_CONNECT       // Remote connected
WTS_REMOTE_DISCONNECT    // Remote disconnected
```

---

## 7. Custom Event Source Creation

### 7.1 Agent Event IDs

```cpp
namespace AgentEventIds {
    // System Events (1000-1099)
    AGENT_STARTED = 1000;
    AGENT_STOPPED = 1001;
    AGENT_ERROR = 1002;
    AGENT_WARNING = 1003;
    CONFIG_RELOADED = 1004;
    
    // Loop Events (1100-1199)
    LOOP_STARTED = 1100;
    LOOP_COMPLETED = 1101;
    LOOP_ERROR = 1102;
    LOOP_TIMEOUT = 1103;
    
    // Task Events (1200-1299)
    TASK_STARTED = 1200;
    TASK_COMPLETED = 1201;
    TASK_FAILED = 1202;
    TASK_CANCELLED = 1203;
    
    // Communication Events (1300-1399)
    EMAIL_SENT = 1300;
    EMAIL_RECEIVED = 1301;
    SMS_SENT = 1302;
    CALL_INITIATED = 1303;
    CALL_COMPLETED = 1304;
    
    // Security Events (1400-1499)
    AUTH_SUCCESS = 1400;
    AUTH_FAILURE = 1401;
    PERMISSION_DENIED = 1402;
    
    // Performance Events (1500-1599)
    HIGH_CPU_USAGE = 1500;
    HIGH_MEMORY_USAGE = 1501;
    DISK_SPACE_LOW = 1502;
    NETWORK_ISSUE = 1503;
}
```

---

## 8. Event Filtering and Querying

### 8.1 XPath Query Builder Pattern

```cpp
class XPathQueryBuilder {
    XPathQueryBuilder& AddQuery(const std::wstring& channel);
    XPathQueryBuilder& WhereEventId(int eventId);
    XPathQueryBuilder& WhereEventIdIn(const std::vector<int>& eventIds);
    XPathQueryBuilder& WhereLevel(int level);
    XPathQueryBuilder& WhereLevelAtLeast(int minLevel);
    XPathQueryBuilder& WhereProvider(const std::wstring& providerName);
    XPathQueryBuilder& WhereTimeCreatedWithinMinutes(int minutes);
    XPathQueryBuilder& WhereTimeCreatedWithinHours(int hours);
    XPathQueryBuilder& WhereTimeCreatedWithinDays(int days);
    XPathQueryBuilder& WhereComputer(const std::wstring& computerName);
    XPathQueryBuilder& WhereSecurityUser(const std::wstring& userName);
    XPathQueryBuilder& And();
    XPathQueryBuilder& Or();
    XPathQueryBuilder& Not();
    std::wstring Build();
};
```

### 8.2 Pre-built Query Examples

```cpp
// Security logon events in last 24 hours
QueryExamples::GetLogonEventsQuery()
    -> *[System[(EventID=4624 or EventID=4625 or EventID=4634)]]
       [System[TimeCreated[timediff(@SystemTime) <= 86400000]]]

// System errors and warnings
QueryExamples::GetSystemErrorsQuery()
    -> *[System[Level<=2]]

// Application events from specific provider
QueryExamples::GetApplicationEventsQuery(providerName)
    -> *[System[Provider[@Name='providerName']]]
       [System[TimeCreated[timediff(@SystemTime) <= 604800000]]]
```

---

## 9. Real-Time Event Monitoring

### 9.1 Monitor Architecture

```
EVENT SOURCES
+----------+----------+----------+----------+
| Event Log| ETW      | Performance| File   |
| Subscription| Session | Counters  | System |
+-----+----+----+-----+-----+----+-----+----+
      |         |         |         |
      +---------+---------+---------+
                |
         EVENT AGGREGATOR
         - Normalize events
         - Add timestamps
         - Deduplicate
         - Buffer high-volume
                |
    +-----------+-----------+
    |           |           |
Real-time   Buffer     Alert
Processor   Manager    Engine
- Filters   - Ring buf - Thresholds
- Patterns  - Persist  - Rate limits
- Correlation          - Escalation
    |           |           |
    +-----------+-----------+
                |
         AGENT DISPATCHER
         - Route to agents
         - Priority dispatch
         - Load balancing
         - Circuit breaker
```

### 9.2 Monitor Statistics

```cpp
struct MonitorStats {
    uint64_t totalEventsProcessed;
    uint64_t totalEventsDropped;
    std::chrono::seconds uptime;
    double eventsPerSecond;
};
```

---

## 10. Integration with Agent Loops

### 10.1 Agent Event Handler Interface

```cpp
class IAgentEventHandler {
public:
    virtual ~IAgentEventHandler() = default;
    virtual void OnEvent(const SystemEvent& event) = 0;
    virtual bool CanHandle(EventSourceType sourceType, 
                          const std::wstring& eventId) = 0;
    virtual int GetPriority() const = 0;
    virtual std::wstring GetName() const = 0;
};
```

### 10.2 Agent Handler Types

| Agent | Priority | Sources | Events |
|-------|----------|---------|--------|
| SecurityAgent | 100 | EventLog, ETW | 4624, 4625, 4634, 4647, 4648 |
| SystemHealthAgent | 50 | Performance, EventLog | CPU, Memory, Disk |
| ProcessMonitorAgent | 75 | ETW | Process start/end |

---

## 11. Security Considerations

### 11.1 Required Privileges

| Operation | Privilege | Notes |
|-----------|-----------|-------|
| Read Security Log | SeSecurityPrivilege | Admin required |
| Write to Event Log | Standard user | Source registered |
| ETW Kernel Tracing | SeSystemProfilePrivilege | Admin required |
| ETW User Tracing | Standard user | No special privileges |
| Performance Counters | Standard user | Some need elevation |
| Session Notifications | Standard user | No special privileges |

### 11.2 Security Manager

```cpp
class SecurityManager {
public:
    static bool HasRequiredPrivilege(const std::wstring& privilegeName);
    static bool EnablePrivilege(const std::wstring& privilegeName);
    static bool IsRunningAsAdmin();
};
```

---

## 12. Implementation Reference

### 12.1 Header Dependencies

```cpp
// Windows Event Log APIs
#include <windows.h>
#include <winevt.h>      // Modern Evt* functions
#include <sddl.h>        // Security descriptor functions

// ETW APIs
#include <evntrace.h>    // Event tracing
#include <evntcons.h>    // Event consumer
#include <evntprov.h>    // Event provider
#include <tdh.h>         // Trace data helper

// Performance Counters
#include <pdh.h>
#include <pdhmsg.h>

// System Notifications
#include <wtsapi32.h>    // Terminal services
#include <dbt.h>         // Device broadcast

// Standard libraries
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <queue>
#include <sstream>
#include <algorithm>
```

### 12.2 Library Dependencies

```cmake
target_link_libraries(AgentMonitoring
    advapi32.lib    # Event log, privileges
    wevtapi.lib     # Modern event log API
    tdh.lib         # Trace data helper
    pdh.lib         # Performance counters
    wtsapi32.lib    # Terminal services
    kernel32.lib
    user32.lib
    shell32.lib
)
```

---

## Appendix A: Event ID Reference

### Security Event IDs

| ID | Description | Severity |
|----|-------------|----------|
| 4624 | Successful logon | Info |
| 4625 | Failed logon | Warning |
| 4634 | Account logoff | Info |
| 4647 | User logoff | Info |
| 4648 | Explicit credential logon | Info |
| 4672 | Special privileges assigned | Info |
| 4720 | User account created | Info |
| 4740 | Account locked out | Warning |
| 4768 | Kerberos ticket requested | Info |
| 4769 | Kerberos service ticket | Info |
| 4771 | Kerberos pre-auth failed | Warning |

### System Event IDs

| ID | Description | Severity |
|----|-------------|----------|
| 6005 | Event log service started | Info |
| 6006 | Event log service stopped | Info |
| 6008 | Unexpected shutdown | Error |
| 6009 | System startup | Info |
| 1074 | Shutdown initiated | Info |
| 41 | Unclean reboot | Error |
| 55 | File system corruption | Error |
| 1001 | Application error | Error |

---

## Appendix B: Performance Counter Reference

### Key Performance Objects

| Object | Key Counters |
|--------|--------------|
| Processor | % Processor Time, % User Time, % Privileged Time |
| Memory | Available Bytes, Committed Bytes, Pages/sec |
| Process | % Processor Time, Working Set, Private Bytes |
| PhysicalDisk | Avg. Disk Queue Length, Disk Bytes/sec |
| LogicalDisk | % Free Space, Free Megabytes |
| Network Interface | Bytes Sent/sec, Bytes Received/sec |
| System | Processes, Threads, Context Switches/sec |

---

## Document Information

| Property | Value |
|----------|-------|
| Version | 1.0 |
| Date | 2025 |
| Platform | Windows 10 |
| Target Framework | Native C++ / Win32 API |
| Minimum Windows Version | Windows 10 (Build 19041+) |

---

*End of Specification*
