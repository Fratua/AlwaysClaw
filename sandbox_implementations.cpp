// Windows AI Agent Sandboxing Implementation
// Comprehensive sandboxing code for Windows 10 AI Agent System
// Based on OpenClaw security analysis

#include <windows.h>
#include <winnt.h>
#include <userenv.h>
#include <sddl.h>
#include <fwpmu.h>
#include <evntrace.h>
#include <evntcons.h>
#include <tdh.h>
#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <memory>

#pragma comment(lib, "userenv.lib")
#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "fwpuclnt.lib")
#pragma comment(lib, "tdh.lib")

namespace AgentSandbox {

// ============================================================================
// SECTION 1: JOB OBJECT SANDBOXING
// ============================================================================

class JobObjectSandbox {
private:
    HANDLE m_hJob;
    std::wstring m_name;
    
public:
    JobObjectSandbox(const std::wstring& name) : m_hJob(nullptr), m_name(name) {}
    
    ~JobObjectSandbox() {
        Close();
    }
    
    BOOL Create() {
        m_hJob = CreateJobObjectW(nullptr, m_name.c_str());
        if (!m_hJob) return FALSE;
        
        return ApplyLimits();
    }
    
    BOOL ApplyLimits() {
        // Basic limits
        JOBOBJECT_BASIC_LIMIT_INFORMATION basicLimits = {0};
        basicLimits.LimitFlags = 
            JOB_OBJECT_LIMIT_ACTIVE_PROCESS |
            JOB_OBJECT_LIMIT_AFFINITY |
            JOB_OBJECT_LIMIT_PRIORITY_CLASS |
            JOB_OBJECT_LIMIT_PROCESS_TIME |
            JOB_OBJECT_LIMIT_JOB_MEMORY |
            JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION;
        
        basicLimits.ActiveProcessLimit = 10;
        basicLimits.Affinity = 0x0000000F;  // First 4 cores
        basicLimits.PriorityClass = BELOW_NORMAL_PRIORITY_CLASS;
        basicLimits.PerProcessUserTimeLimit.QuadPart = 36000000000LL;  // 1 hour
        basicLimits.PerJobUserTimeLimit.QuadPart = 864000000000LL;     // 24 hours
        
        if (!SetInformationJobObject(m_hJob, JobObjectBasicLimitInformation,
                                      &basicLimits, sizeof(basicLimits))) {
            return FALSE;
        }
        
        // Extended limits
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION extLimits = {0};
        extLimits.BasicLimitInformation = basicLimits;
        extLimits.JobMemoryLimit = 4ULL * 1024 * 1024 * 1024;      // 4GB
        extLimits.ProcessMemoryLimit = 1ULL * 1024 * 1024 * 1024;  // 1GB per process
        
        if (!SetInformationJobObject(m_hJob, JobObjectExtendedLimitInformation,
                                      &extLimits, sizeof(extLimits))) {
            return FALSE;
        }
        
        // UI restrictions
        JOBOBJECT_BASIC_UI_RESTRICTIONS uiRestrictions = {0};
        uiRestrictions.UIRestrictionsClass = 
            JOB_OBJECT_UILIMIT_HANDLES |
            JOB_OBJECT_UILIMIT_READCLIPBOARD |
            JOB_OBJECT_UILIMIT_WRITECLIPBOARD |
            JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS |
            JOB_OBJECT_UILIMIT_DISPLAYSETTINGS |
            JOB_OBJECT_UILIMIT_GLOBALATOMS |
            JOB_OBJECT_UILIMIT_DESKTOP |
            JOB_OBJECT_UILIMIT_EXITWINDOWS;
        
        return SetInformationJobObject(m_hJob, JobObjectBasicUIRestrictions,
                                        &uiRestrictions, sizeof(uiRestrictions));
    }
    
    BOOL AssignProcess(HANDLE hProcess) {
        return AssignProcessToJobObject(m_hJob, hProcess);
    }
    
    void Close() {
        if (m_hJob) {
            CloseHandle(m_hJob);
            m_hJob = nullptr;
        }
    }
};

// ============================================================================
// SECTION 2: PROCESS MITIGATION POLICIES
// ============================================================================

class ProcessMitigationPolicy {
public:
    static BOOL ApplyMitigations(HANDLE hProcess) {
        // Modern exploit mitigations
        DWORD64 mitigationPolicies = 
            PROCESS_CREATION_MITIGATION_POLICY_DEP_ENABLE |
            PROCESS_CREATION_MITIGATION_POLICY_DEP_ATL_THUNK_ENABLE |
            PROCESS_CREATION_MITIGATION_POLICY_SEHOP_ENABLE |
            PROCESS_CREATION_MITIGATION_POLICY_FORCE_RELOCATE_IMAGES_ALWAYS_ON |
            PROCESS_CREATION_MITIGATION_POLICY_HEAP_TERMINATE_ALWAYS_ON |
            PROCESS_CREATION_MITIGATION_POLICY_BOTTOM_UP_ASLR_ALWAYS_ON |
            PROCESS_CREATION_MITIGATION_POLICY_HIGH_ENTROPY_ASLR_ALWAYS_ON |
            PROCESS_CREATION_MITIGATION_POLICY_STRICT_HANDLE_CHECKS_ALWAYS_ON |
            PROCESS_CREATION_MITIGATION_POLICY_EXTENSION_POINT_DISABLE_ALWAYS_ON |
            PROCESS_CREATION_MITIGATION_POLICY_PROHIBIT_DYNAMIC_CODE_ALWAYS_ON |
            PROCESS_CREATION_MITIGATION_POLICY_CONTROL_FLOW_GUARD_ALWAYS_ON |
            PROCESS_CREATION_MITIGATION_POLICY_BLOCK_NON_MICROSOFT_BINARIES_ALWAYS_ON |
            PROCESS_CREATION_MITIGATION_POLICY_FONT_DISABLE_ALWAYS_ON |
            PROCESS_CREATION_MITIGATION_POLICY_IMAGE_LOAD_NO_REMOTE_ALWAYS_ON |
            PROCESS_CREATION_MITIGATION_POLICY_IMAGE_LOAD_NO_LOW_LABEL_ALWAYS_ON |
            PROCESS_CREATION_MITIGATION_POLICY_IMAGE_LOAD_PREFER_SYSTEM32_ALWAYS_ON;
        
        PROCESS_MITIGATION_POLICY policies[] = {
            ProcessDEPPolicy,
            ProcessASLRPolicy,
            ProcessDynamicCodePolicy,
            ProcessStrictHandleCheckPolicy,
            ProcessSystemCallDisablePolicy,
            ProcessExtensionPointDisablePolicy,
            ProcessControlFlowGuardPolicy,
            ProcessSignaturePolicy,
            ProcessFontDisablePolicy,
            ProcessImageLoadPolicy
        };
        
        for (auto policy : policies) {
            DWORD64 value = mitigationPolicies;
            SetProcessMitigationPolicy(policy, &value, sizeof(value));
        }
        
        return TRUE;
    }
    
    static BOOL CreateMitigatedProcess(LPCWSTR commandLine, LPPROCESS_INFORMATION lpPI) {
        STARTUPINFOEX siex = {0};
        siex.StartupInfo.cb = sizeof(STARTUPINFOEX);
        
        DWORD64 mitigationPolicies = 
            PROCESS_CREATION_MITIGATION_POLICY_DEP_ENABLE |
            PROCESS_CREATION_MITIGATION_POLICY_DEP_ATL_THUNK_ENABLE |
            PROCESS_CREATION_MITIGATION_POLICY_SEHOP_ENABLE |
            PROCESS_CREATION_MITIGATION_POLICY_FORCE_RELOCATE_IMAGES_ALWAYS_ON |
            PROCESS_CREATION_MITIGATION_POLICY_HEAP_TERMINATE_ALWAYS_ON |
            PROCESS_CREATION_MITIGATION_POLICY_BOTTOM_UP_ASLR_ALWAYS_ON |
            PROCESS_CREATION_MITIGATION_POLICY_HIGH_ENTROPY_ASLR_ALWAYS_ON |
            PROCESS_CREATION_MITIGATION_POLICY_STRICT_HANDLE_CHECKS_ALWAYS_ON |
            PROCESS_CREATION_MITIGATION_POLICY_EXTENSION_POINT_DISABLE_ALWAYS_ON |
            PROCESS_CREATION_MITIGATION_POLICY_PROHIBIT_DYNAMIC_CODE_ALWAYS_ON |
            PROCESS_CREATION_MITIGATION_POLICY_CONTROL_FLOW_GUARD_ALWAYS_ON |
            PROCESS_CREATION_MITIGATION_POLICY_BLOCK_NON_MICROSOFT_BINARIES_ALWAYS_ON |
            PROCESS_CREATION_MITIGATION_POLICY_FONT_DISABLE_ALWAYS_ON |
            PROCESS_CREATION_MITIGATION_POLICY_IMAGE_LOAD_NO_REMOTE_ALWAYS_ON |
            PROCESS_CREATION_MITIGATION_POLICY_IMAGE_LOAD_NO_LOW_LABEL_ALWAYS_ON |
            PROCESS_CREATION_MITIGATION_POLICY_IMAGE_LOAD_PREFER_SYSTEM32_ALWAYS_ON;
        
        SIZE_T attrSize = 0;
        InitializeProcThreadAttributeList(nullptr, 1, 0, &attrSize);
        siex.lpAttributeList = (LPPROC_THREAD_ATTRIBUTE_LIST)HeapAlloc(
            GetProcessHeap(), HEAP_ZERO_MEMORY, attrSize);
        
        if (!InitializeProcThreadAttributeList(siex.lpAttributeList, 1, 0, &attrSize)) {
            HeapFree(GetProcessHeap(), 0, siex.lpAttributeList);
            return FALSE;
        }
        
        if (!UpdateProcThreadAttribute(siex.lpAttributeList, 0,
            PROC_THREAD_ATTRIBUTE_MITIGATION_POLICY,
            &mitigationPolicies, sizeof(mitigationPolicies), nullptr, nullptr)) {
            DeleteProcThreadAttributeList(siex.lpAttributeList);
            HeapFree(GetProcessHeap(), 0, siex.lpAttributeList);
            return FALSE;
        }
        
        BOOL result = CreateProcessW(nullptr, (LPWSTR)commandLine, nullptr, nullptr, FALSE,
            EXTENDED_STARTUPINFO_PRESENT | CREATE_NEW_CONSOLE,
            nullptr, nullptr, &siex.StartupInfo, lpPI);
        
        DeleteProcThreadAttributeList(siex.lpAttributeList);
        HeapFree(GetProcessHeap(), 0, siex.lpAttributeList);
        
        return result;
    }
};

// ============================================================================
// SECTION 3: TOKEN RESTRICTION AND SANDBOXING
// ============================================================================

class TokenSandbox {
public:
    // Dangerous privileges that agent processes should NEVER have
    static constexpr LPCWSTR DANGEROUS_PRIVILEGES[] = {
        SE_DEBUG_NAME,
        SE_TCB_NAME,
        SE_CREATE_TOKEN_NAME,
        SE_ASSIGNPRIMARYTOKEN_NAME,
        SE_LOAD_DRIVER_NAME,
        SE_SYSTEM_ENVIRONMENT_NAME,
        SE_MANAGE_VOLUME_NAME,
        SE_BACKUP_NAME,
        SE_RESTORE_NAME,
        SE_TAKE_OWNERSHIP_NAME,
        SE_SECURITY_NAME,
        SE_INC_BASE_PRIORITY_NAME,
        SE_SHUTDOWN_NAME,
        SE_UNDOCK_NAME,
        SE_RELABEL_NAME,
        SE_TIME_ZONE_NAME,
        SE_CREATE_PAGEFILE_NAME,
        SE_CREATE_PERMANENT_NAME,
        SE_ENABLE_DELEGATION_NAME,
        SE_LOCK_MEMORY_NAME,
        SE_MACHINE_ACCOUNT_NAME,
        SE_PROF_SINGLE_PROCESS_NAME,
        SE_REMOTE_SHUTDOWN_NAME,
        SE_SYNC_AGENT_NAME,
        SE_TRUSTED_CREDMAN_ACCESS_NAME,
        SE_DELEGATE_SESSION_USER_IMPERSONATE_NAME
    };
    
    static BOOL RemoveAllPrivileges(HANDLE hToken) {
        DWORD tokenInfoLength = 0;
        GetTokenInformation(hToken, TokenPrivileges, nullptr, 0, &tokenInfoLength);
        
        auto* privileges = (TOKEN_PRIVILEGES*)malloc(tokenInfoLength);
        if (!GetTokenInformation(hToken, TokenPrivileges, privileges,
                                  tokenInfoLength, &tokenInfoLength)) {
            free(privileges);
            return FALSE;
        }
        
        for (DWORD i = 0; i < privileges->PrivilegeCount; i++) {
            privileges->Privileges[i].Attributes = SE_PRIVILEGE_REMOVED;
        }
        
        BOOL result = AdjustTokenPrivileges(hToken, FALSE, privileges,
                                            0, nullptr, nullptr);
        free(privileges);
        
        return result && GetLastError() == ERROR_SUCCESS;
    }
    
    static BOOL CreateRestrictedTokenForAgent(HANDLE hOriginalToken,
                                               PHANDLE phRestrictedToken) {
        SID_IDENTIFIER_AUTHORITY ntAuthority = SECURITY_NT_AUTHORITY;
        PSID administratorsSid = nullptr;
        PSID powerUsersSid = nullptr;
        PSID authenticatedUsersSid = nullptr;
        
        AllocateAndInitializeSid(&ntAuthority, 2,
            SECURITY_BUILTIN_DOMAIN_RID, DOMAIN_ALIAS_RID_ADMINS,
            0, 0, 0, 0, 0, 0, &administratorsSid);
        
        AllocateAndInitializeSid(&ntAuthority, 2,
            SECURITY_BUILTIN_DOMAIN_RID, DOMAIN_ALIAS_RID_POWER_USERS,
            0, 0, 0, 0, 0, 0, &powerUsersSid);
        
        AllocateAndInitializeSid(&ntAuthority, 1,
            SECURITY_AUTHENTICATED_USER_RID,
            0, 0, 0, 0, 0, 0, &authenticatedUsersSid);
        
        SID_AND_ATTRIBUTES sidsToDisable[] = {
            { administratorsSid, 0 },
            { powerUsersSid, 0 },
            { authenticatedUsersSid, 0 }
        };
        
        LUID_AND_ATTRIBUTES privilegesToDelete[32];
        DWORD privilegeCount = 0;
        
        for (auto& privName : DANGEROUS_PRIVILEGES) {
            LUID luid;
            if (LookupPrivilegeValueW(nullptr, privName, &luid)) {
                privilegesToDelete[privilegeCount].Luid = luid;
                privilegesToDelete[privilegeCount].Attributes = SE_PRIVILEGE_REMOVED;
                privilegeCount++;
            }
        }
        
        BOOL result = CreateRestrictedToken(hOriginalToken,
            DISABLE_MAX_PRIVILEGE | SANDBOX_INERT,
            ARRAYSIZE(sidsToDisable), sidsToDisable,
            privilegeCount, privilegesToDelete,
            0, nullptr, phRestrictedToken);
        
        FreeSid(administratorsSid);
        FreeSid(powerUsersSid);
        FreeSid(authenticatedUsersSid);
        
        return result;
    }
    
    static BOOL SetLowIntegrityLevel(HANDLE hToken) {
        SID_IDENTIFIER_AUTHORITY mlAuthority = SECURITY_MANDATORY_LABEL_AUTHORITY;
        PSID lowIntegritySid = nullptr;
        
        AllocateAndInitializeSid(&mlAuthority, 1,
            SECURITY_MANDATORY_LOW_RID,
            0, 0, 0, 0, 0, 0, 0, &lowIntegritySid);
        
        TOKEN_MANDATORY_LABEL tml = {0};
        tml.Label.Sid = lowIntegritySid;
        tml.Label.Attributes = SE_GROUP_INTEGRITY;
        
        BOOL result = SetTokenInformation(hToken, TokenIntegrityLevel,
            &tml, sizeof(TOKEN_MANDATORY_LABEL) + GetLengthSid(lowIntegritySid));
        
        FreeSid(lowIntegritySid);
        return result;
    }
    
    static BOOL CreateSandboxedProcess(LPCWSTR commandLine) {
        HANDLE hOriginalToken = nullptr;
        if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ALL_ACCESS, &hOriginalToken)) {
            return FALSE;
        }
        
        HANDLE hRestrictedToken = nullptr;
        if (!CreateRestrictedTokenForAgent(hOriginalToken, &hRestrictedToken)) {
            CloseHandle(hOriginalToken);
            return FALSE;
        }
        
        SetLowIntegrityLevel(hRestrictedToken);
        RemoveAllPrivileges(hRestrictedToken);
        
        STARTUPINFOW si = {0};
        si.cb = sizeof(STARTUPINFOW);
        
        PROCESS_INFORMATION pi = {0};
        
        BOOL result = CreateProcessAsUserW(hRestrictedToken, nullptr, (LPWSTR)commandLine,
            nullptr, nullptr, FALSE, CREATE_NEW_CONSOLE | CREATE_UNICODE_ENVIRONMENT,
            nullptr, nullptr, &si, &pi);
        
        CloseHandle(hOriginalToken);
        CloseHandle(hRestrictedToken);
        
        if (result) {
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
        }
        
        return result;
    }
};

// ============================================================================
// SECTION 4: APPCONTAINER SANDBOXING
// ============================================================================

class AppContainerSandbox {
private:
    PSID m_packageSid;
    std::vector<PSID> m_capabilities;
    HANDLE m_processToken;
    std::wstring m_packageName;
    
public:
    // Well-known capability SIDs
    static constexpr LPCWSTR CAPABILITY_INTERNET_CLIENT = L"S-1-15-3-1";
    static constexpr LPCWSTR CAPABILITY_INTERNET_CLIENT_SERVER = L"S-1-15-3-2";
    static constexpr LPCWSTR CAPABILITY_PRIVATE_NETWORK_CLIENT_SERVER = L"S-1-15-3-3";
    static constexpr LPCWSTR CAPABILITY_DOCUMENTS_LIBRARY = L"S-1-15-3-8";
    static constexpr LPCWSTR CAPABILITY_VIDEOS_LIBRARY = L"S-1-15-3-9";
    static constexpr LPCWSTR CAPABILITY_PICTURES_LIBRARY = L"S-1-15-3-10";
    static constexpr LPCWSTR CAPABILITY_MUSIC_LIBRARY = L"S-1-15-3-11";
    static constexpr LPCWSTR CAPABILITY_ENTERPRISE_AUTHENTICATION = L"S-1-15-3-7";
    static constexpr LPCWSTR CAPABILITY_SHARED_USER_CERTIFICATES = L"S-1-15-3-12";
    static constexpr LPCWSTR CAPABILITY_REMOVABLE_STORAGE = L"S-1-15-3-13";
    static constexpr LPCWSTR CAPABILITY_APPOINTMENTS = L"S-1-15-3-16";
    static constexpr LPCWSTR CAPABILITY_CONTACTS = L"S-1-15-3-17";
    
    AppContainerSandbox() : m_packageSid(nullptr), m_processToken(nullptr) {}
    
    ~AppContainerSandbox() {
        Cleanup();
    }
    
    BOOL Initialize(LPCWSTR packageName, const std::vector<LPCWSTR>& capabilities) {
        m_packageName = packageName;
        
        HRESULT hr = DeriveAppContainerSidFromAppContainerName(packageName, &m_packageSid);
        if (FAILED(hr)) return FALSE;
        
        for (auto cap : capabilities) {
            PSID sid = nullptr;
            if (ConvertStringSidToSidW(cap, &sid)) {
                m_capabilities.push_back(sid);
            }
        }
        
        return TRUE;
    }
    
    BOOL CreateSandboxedProcess(LPCWSTR commandLine, LPCWSTR workingDirectory) {
        SECURITY_CAPABILITIES secCaps = {0};
        secCaps.AppContainerSid = m_packageSid;
        
        if (!m_capabilities.empty()) {
            secCaps.Capabilities = new SID_AND_ATTRIBUTES[m_capabilities.size()];
            secCaps.CapabilityCount = (DWORD)m_capabilities.size();
            
            for (size_t i = 0; i < m_capabilities.size(); i++) {
                secCaps.Capabilities[i].Sid = m_capabilities[i];
                secCaps.Capabilities[i].Attributes = SE_GROUP_ENABLED;
            }
        }
        
        STARTUPINFOEX siex = {0};
        siex.StartupInfo.cb = sizeof(STARTUPINFOEX);
        
        SIZE_T attrSize = 0;
        InitializeProcThreadAttributeList(nullptr, 1, 0, &attrSize);
        siex.lpAttributeList = (LPPROC_THREAD_ATTRIBUTE_LIST)HeapAlloc(
            GetProcessHeap(), HEAP_ZERO_MEMORY, attrSize);
        
        InitializeProcThreadAttributeList(siex.lpAttributeList, 1, 0, &attrSize);
        
        UpdateProcThreadAttribute(siex.lpAttributeList, 0,
            PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES,
            &secCaps, sizeof(secCaps), nullptr, nullptr);
        
        PROCESS_INFORMATION pi = {0};
        
        BOOL result = CreateProcessW(nullptr, (LPWSTR)commandLine, nullptr, nullptr, FALSE,
            EXTENDED_STARTUPINFO_PRESENT | CREATE_NEW_CONSOLE,
            nullptr, workingDirectory, &siex.StartupInfo, &pi);
        
        DeleteProcThreadAttributeList(siex.lpAttributeList);
        HeapFree(GetProcessHeap(), 0, siex.lpAttributeList);
        delete[] secCaps.Capabilities;
        
        if (result) {
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
        }
        
        return result;
    }
    
    BOOL AddFileAccess(LPCWSTR filePath, DWORD accessMask) {
        PSECURITY_DESCRIPTOR sd = nullptr;
        PACL pDacl = nullptr;
        PACL newDacl = nullptr;
        
        GetNamedSecurityInfoW(filePath, SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION, nullptr, nullptr,
            &pDacl, nullptr, &sd);
        
        EXPLICIT_ACCESS_W ea = {0};
        ea.grfAccessPermissions = accessMask;
        ea.grfAccessMode = GRANT_ACCESS;
        ea.grfInheritance = CONTAINER_INHERIT_ACE | OBJECT_INHERIT_ACE;
        ea.Trustee.TrusteeForm = TRUSTEE_IS_SID;
        ea.Trustee.TrusteeType = TRUSTEE_IS_GROUP;
        ea.Trustee.ptstrName = (LPWSTR)m_packageSid;
        
        SetEntriesInAclW(1, &ea, pDacl, &newDacl);
        
        DWORD result = SetNamedSecurityInfoW((LPWSTR)filePath, SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION, nullptr, nullptr, newDacl, nullptr);
        
        LocalFree(newDacl);
        return result == ERROR_SUCCESS;
    }
    
    void Cleanup() {
        if (m_packageSid) {
            FreeSid(m_packageSid);
            m_packageSid = nullptr;
        }
        for (auto sid : m_capabilities) {
            if (sid) LocalFree(sid);
        }
        m_capabilities.clear();
        if (m_processToken) {
            CloseHandle(m_processToken);
            m_processToken = nullptr;
        }
    }
};

// ============================================================================
// SECTION 5: NETWORK SANDBOXING WITH WFP
// ============================================================================

struct NetworkPolicy {
    BOOL allowDNS;
    BOOL allowOutbound;
    BOOL requireProxy;
    std::vector<USHORT> allowedOutboundPorts;
    std::vector<std::wstring> allowedAddresses;
    std::vector<std::wstring> blockedAddresses;
    DWORD rateLimitMbps;
};

class NetworkSandbox {
private:
    HANDLE m_engineHandle;
    GUID m_providerGuid;
    GUID m_sublayerGuid;
    std::vector<GUID> m_filterGuids;
    
public:
    NetworkSandbox() : m_engineHandle(nullptr) {
        UuidCreate(&m_providerGuid);
        UuidCreate(&m_sublayerGuid);
    }
    
    ~NetworkSandbox() {
        Cleanup();
    }
    
    BOOL Initialize() {
        DWORD result = FwpmEngineOpenW(nullptr, RPC_C_AUTHN_WINNT,
            nullptr, nullptr, &m_engineHandle);
        if (result != ERROR_SUCCESS) return FALSE;
        
        FWPM_PROVIDER0 provider = {0};
        provider.providerKey = m_providerGuid;
        provider.displayData.name = (PWSTR)L"AI Agent Network Sandbox";
        provider.displayData.description = (PWSTR)L"Network isolation for AI agent processes";
        provider.flags = FWPM_PROVIDER_FLAG_PERSISTENT;
        
        result = FwpmProviderAdd0(m_engineHandle, &provider, nullptr);
        if (result != ERROR_SUCCESS && result != FWP_E_ALREADY_EXISTS) {
            return FALSE;
        }
        
        FWPM_SUBLAYER0 sublayer = {0};
        sublayer.subLayerKey = m_sublayerGuid;
        sublayer.displayData.name = (PWSTR)L"Agent Sandbox Sublayer";
        sublayer.displayData.description = (PWSTR)L"Sublayer for agent sandbox filters";
        sublayer.flags = FWPM_SUBLAYER_FLAG_PERSISTENT;
        sublayer.providerKey = &m_providerGuid;
        sublayer.weight = 0x100;
        
        result = FwpmSubLayerAdd0(m_engineHandle, &sublayer, nullptr);
        if (result != ERROR_SUCCESS && result != FWP_E_ALREADY_EXISTS) {
            return FALSE;
        }
        
        return TRUE;
    }
    
    BOOL AddProcessFilter(DWORD processId, const NetworkPolicy& policy) {
        if (!AddBlockAllFilter(processId)) return FALSE;
        if (!AddLoopbackFilter(processId)) return FALSE;
        
        if (policy.allowDNS) {
            if (!AddDNSFilter(processId)) return FALSE;
        }
        
        for (auto port : policy.allowedOutboundPorts) {
            if (!AddOutboundPortFilter(processId, port)) return FALSE;
        }
        
        for (auto& address : policy.allowedAddresses) {
            if (!AddAddressFilter(processId, address.c_str())) return FALSE;
        }
        
        for (auto& address : policy.blockedAddresses) {
            if (!AddBlockAddressFilter(processId, address.c_str())) return FALSE;
        }
        
        return TRUE;
    }
    
    void Cleanup() {
        for (auto& guid : m_filterGuids) {
            FwpmFilterDeleteByKey0(m_engineHandle, &guid);
        }
        m_filterGuids.clear();
        
        FwpmSubLayerDeleteByKey0(m_engineHandle, &m_sublayerGuid);
        FwpmProviderDeleteByKey0(m_engineHandle, &m_providerGuid);
        
        if (m_engineHandle) {
            FwpmEngineClose0(m_engineHandle);
            m_engineHandle = nullptr;
        }
    }
    
private:
    BOOL AddBlockAllFilter(DWORD processId) {
        FWPM_FILTER0 filter = {0};
        GUID filterGuid;
        UuidCreate(&filterGuid);
        
        filter.filterKey = filterGuid;
        filter.displayData.name = (PWSTR)L"Block All Outbound";
        filter.displayData.description = (PWSTR)L"Block all outbound traffic for agent";
        filter.flags = FWPM_FILTER_FLAG_PERSISTENT;
        filter.providerKey = &m_providerGuid;
        filter.layerKey = FWPM_LAYER_ALE_AUTH_CONNECT_V4;
        filter.subLayerKey = m_sublayerGuid;
        filter.weight.type = FWP_UINT8;
        filter.weight.uint8 = 0xFF;
        filter.action.type = FWP_ACTION_BLOCK;
        
        DWORD result = FwpmFilterAdd0(m_engineHandle, &filter, nullptr, nullptr);
        if (result == ERROR_SUCCESS) {
            m_filterGuids.push_back(filterGuid);
        }
        
        return result == ERROR_SUCCESS;
    }
    
    BOOL AddLoopbackFilter(DWORD processId) {
        FWPM_FILTER0 filter = {0};
        GUID filterGuid;
        UuidCreate(&filterGuid);
        
        filter.filterKey = filterGuid;
        filter.displayData.name = (PWSTR)L"Allow Loopback";
        filter.flags = FWPM_FILTER_FLAG_PERSISTENT;
        filter.providerKey = &m_providerGuid;
        filter.layerKey = FWPM_LAYER_ALE_AUTH_CONNECT_V4;
        filter.subLayerKey = m_sublayerGuid;
        filter.weight.type = FWP_UINT8;
        filter.weight.uint8 = 0x10;
        filter.action.type = FWP_ACTION_PERMIT;
        
        FWP_CONDITION_VALUE0 localIp = {0};
        localIp.type = FWP_UINT32;
        localIp.uint32 = 0x0100007F;  // 127.0.0.1
        
        FWPM_FILTER_CONDITION0 cond = {0};
        cond.fieldKey = FWPM_CONDITION_IP_LOCAL_ADDRESS;
        cond.matchType = FWP_MATCH_EQUAL;
        cond.conditionValue = localIp;
        
        filter.filterCondition = &cond;
        filter.numFilterConditions = 1;
        
        DWORD result = FwpmFilterAdd0(m_engineHandle, &filter, nullptr, nullptr);
        if (result == ERROR_SUCCESS) {
            m_filterGuids.push_back(filterGuid);
        }
        
        return result == ERROR_SUCCESS;
    }
    
    BOOL AddDNSFilter(DWORD processId) {
        FWPM_FILTER0 filter = {0};
        GUID filterGuid;
        UuidCreate(&filterGuid);
        
        filter.filterKey = filterGuid;
        filter.displayData.name = (PWSTR)L"Allow DNS";
        filter.flags = FWPM_FILTER_FLAG_PERSISTENT;
        filter.providerKey = &m_providerGuid;
        filter.layerKey = FWPM_LAYER_ALE_AUTH_CONNECT_V4;
        filter.subLayerKey = m_sublayerGuid;
        filter.weight.type = FWP_UINT8;
        filter.weight.uint8 = 0x10;
        filter.action.type = FWP_ACTION_PERMIT;
        
        FWP_CONDITION_VALUE0 remotePort = {0};
        remotePort.type = FWP_UINT16;
        remotePort.uint16 = 53;
        
        FWPM_FILTER_CONDITION0 cond = {0};
        cond.fieldKey = FWPM_CONDITION_IP_REMOTE_PORT;
        cond.matchType = FWP_MATCH_EQUAL;
        cond.conditionValue = remotePort;
        
        filter.filterCondition = &cond;
        filter.numFilterConditions = 1;
        
        DWORD result = FwpmFilterAdd0(m_engineHandle, &filter, nullptr, nullptr);
        if (result == ERROR_SUCCESS) {
            m_filterGuids.push_back(filterGuid);
        }
        
        return result == ERROR_SUCCESS;
    }
    
    BOOL AddOutboundPortFilter(DWORD processId, USHORT port) {
        FWPM_FILTER0 filter = {0};
        GUID filterGuid;
        UuidCreate(&filterGuid);
        
        filter.filterKey = filterGuid;
        filter.displayData.name = (PWSTR)L"Allow Outbound Port";
        filter.flags = FWPM_FILTER_FLAG_PERSISTENT;
        filter.providerKey = &m_providerGuid;
        filter.layerKey = FWPM_LAYER_ALE_AUTH_CONNECT_V4;
        filter.subLayerKey = m_sublayerGuid;
        filter.weight.type = FWP_UINT8;
        filter.weight.uint8 = 0x10;
        filter.action.type = FWP_ACTION_PERMIT;
        
        FWP_CONDITION_VALUE0 remotePort = {0};
        remotePort.type = FWP_UINT16;
        remotePort.uint16 = port;
        
        FWPM_FILTER_CONDITION0 cond = {0};
        cond.fieldKey = FWPM_CONDITION_IP_REMOTE_PORT;
        cond.matchType = FWP_MATCH_EQUAL;
        cond.conditionValue = remotePort;
        
        filter.filterCondition = &cond;
        filter.numFilterConditions = 1;
        
        DWORD result = FwpmFilterAdd0(m_engineHandle, &filter, nullptr, nullptr);
        if (result == ERROR_SUCCESS) {
            m_filterGuids.push_back(filterGuid);
        }
        
        return result == ERROR_SUCCESS;
    }
    
    BOOL AddAddressFilter(DWORD processId, LPCWSTR address) {
        // Implementation for adding address filter
        return TRUE;
    }
    
    BOOL AddBlockAddressFilter(DWORD processId, LPCWSTR address) {
        // Implementation for adding block address filter
        return TRUE;
    }
};

// ============================================================================
// MAIN FUNCTION - EXAMPLE USAGE
// ============================================================================

int wmain(int argc, wchar_t* argv[]) {
    std::wcout << L"AI Agent Sandbox Implementation" << std::endl;
    std::wcout << L"================================" << std::endl;
    
    // Example: Create Job Object sandbox
    JobObjectSandbox jobSandbox(L"AIAgent_Job");
    if (jobSandbox.Create()) {
        std::wcout << L"Job object sandbox created successfully" << std::endl;
    }
    
    // Example: Create AppContainer sandbox
    AppContainerSandbox appContainer;
    std::vector<LPCWSTR> capabilities = {
        AppContainerSandbox::CAPABILITY_INTERNET_CLIENT
    };
    
    if (appContainer.Initialize(L"AIAgent.SkillExecutor", capabilities)) {
        std::wcout << L"AppContainer initialized successfully" << std::endl;
    }
    
    // Example: Create network sandbox
    NetworkSandbox netSandbox;
    if (netSandbox.Initialize()) {
        std::wcout << L"Network sandbox initialized successfully" << std::endl;
    }
    
    std::wcout << L"Sandbox setup complete" << std::endl;
    
    return 0;
}

} // namespace AgentSandbox
