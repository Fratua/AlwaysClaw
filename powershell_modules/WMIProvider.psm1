# WMIProvider.psm1
#Requires -Version 5.1

<#
.SYNOPSIS
    WMI Integration module for AI Agent system management
.DESCRIPTION
    Provides comprehensive WMI access for system information,
    hardware monitoring, and management operations
#>

class WMIProvider {
    [string]$Namespace
    [hashtable]$QueryCache
    [int]$CacheDurationSeconds
    
    WMIProvider([string]$namespace = 'root\cimv2') {
        $this.Namespace = $namespace
        $this.QueryCache = @{}
        $this.CacheDurationSeconds = 60
    }
    
    #region System Information
    
    [PSCustomObject] GetComputerSystem() {
        return $this.ExecuteQuery('SELECT * FROM Win32_ComputerSystem') | Select-Object -First 1
    }
    
    [PSCustomObject] GetOperatingSystem() {
        return $this.ExecuteQuery('SELECT * FROM Win32_OperatingSystem') | Select-Object -First 1
    }
    
    [PSCustomObject] GetProcessor() {
        return $this.ExecuteQuery('SELECT * FROM Win32_Processor') | Select-Object -First 1
    }
    
    [array] GetPhysicalMemory() {
        return $this.ExecuteQuery('SELECT * FROM Win32_PhysicalMemory')
    }
    
    [PSCustomObject] GetBIOS() {
        return $this.ExecuteQuery('SELECT * FROM Win32_BIOS') | Select-Object -First 1
    }
    
    [array] GetDiskDrives() {
        return $this.ExecuteQuery('SELECT * FROM Win32_DiskDrive')
    }
    
    [array] GetLogicalDisks() {
        return $this.ExecuteQuery('SELECT * FROM Win32_LogicalDisk WHERE DriveType=3')
    }
    
    #endregion
    
    #region Process & Service Management
    
    [array] GetProcesses([hashtable]$Filter) {
        $query = 'SELECT * FROM Win32_Process'
        
        if ($Filter) {
            $conditions = @()
            foreach ($key in $Filter.Keys) {
                $conditions += "$key = '$($Filter[$key])'"
            }
            $query += ' WHERE ' + ($conditions -join ' AND ')
        }
        
        return $this.ExecuteQuery($query)
    }
    
    [array] GetServices([string]$State, [string]$StartMode) {
        $query = 'SELECT * FROM Win32_Service'
        $conditions = @()
        
        if ($State) { $conditions += "State = '$State'" }
        if ($StartMode) { $conditions += "StartMode = '$StartMode'" }
        
        if ($conditions.Count -gt 0) {
            $query += ' WHERE ' + ($conditions -join ' AND ')
        }
        
        return $this.ExecuteQuery($query)
    }
    
    [bool] StartService([string]$ServiceName) {
        try {
            $service = Get-WmiObject -Class Win32_Service -Filter "Name = '$ServiceName'"
            if ($service) {
                $result = $service.StartService()
                return $result.ReturnValue -eq 0
            }
            return $false
        } catch {
            Write-Error "Failed to start service $ServiceName : $_"
            return $false
        }
    }
    
    [bool] StopService([string]$ServiceName) {
        try {
            $service = Get-WmiObject -Class Win32_Service -Filter "Name = '$ServiceName'"
            if ($service) {
                $result = $service.StopService()
                return $result.ReturnValue -eq 0
            }
            return $false
        } catch {
            Write-Error "Failed to stop service $ServiceName : $_"
            return $false
        }
    }
    
    [bool] SetServiceStartMode([string]$ServiceName, [string]$StartMode) {
        $validModes = @('Auto', 'Manual', 'Disabled', 'Boot', 'System')
        if ($StartMode -notin $validModes) {
            throw "Invalid start mode. Valid values: $($validModes -join ', ')"
        }
        
        try {
            $service = Get-WmiObject -Class Win32_Service -Filter "Name = '$ServiceName'"
            if ($service) {
                $result = $service.ChangeStartMode($StartMode)
                return $result.ReturnValue -eq 0
            }
            return $false
        } catch {
            Write-Error "Failed to change service start mode: $_"
            return $false
        }
    }
    
    #endregion
    
    #region Performance Monitoring
    
    [PSCustomObject] GetSystemPerformance() {
        $performance = @{}
        
        # CPU Usage
        $processor = $this.GetProcessor()
        $performance.CpuUsage = $processor.LoadPercentage
        
        # Memory Usage
        $os = $this.GetOperatingSystem()
        $totalMemory = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
        $freeMemory = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
        $performance.TotalMemoryGB = $totalMemory
        $performance.FreeMemoryGB = $freeMemory
        $performance.MemoryUsedPercent = [math]::Round((($totalMemory - $freeMemory) / $totalMemory) * 100, 2)
        
        # Disk Usage
        $disks = $this.GetLogicalDisks() | ForEach-Object {
            [PSCustomObject]@{
                Drive = $_.DeviceID
                TotalGB = [math]::Round($_.Size / 1GB, 2)
                FreeGB = [math]::Round($_.FreeSpace / 1GB, 2)
                UsedPercent = [math]::Round((($_.Size - $_.FreeSpace) / $_.Size) * 100, 2)
            }
        }
        $performance.Disks = $disks
        
        return [PSCustomObject]$performance
    }
    
    [array] GetProcessPerformance() {
        $query = @'
            SELECT Name, ProcessId, WorkingSetSize, PageFileUsage, 
                   UserModeTime, KernelModeTime, ThreadCount, HandleCount
            FROM Win32_Process
'@
        
        $processes = $this.ExecuteQuery($query)
        
        return $processes | ForEach-Object {
            [PSCustomObject]@{
                Name = $_.Name
                ProcessId = $_.ProcessId
                WorkingSetMB = [math]::Round($_.WorkingSetSize / 1MB, 2)
                PageFileMB = [math]::Round($_.PageFileUsage / 1MB, 2)
                ThreadCount = $_.ThreadCount
                HandleCount = $_.HandleCount
            }
        }
    }
    
    #endregion
    
    #region Event Log Access
    
    [array] GetEventLogs(
        [string]$LogName,
        [int]$Newest = 100,
        [string]$Level,
        [datetime]$After,
        [datetime]$Before
    ) {
        # Use Get-WinEvent for better performance
        $filterXPath = @()
        
        if ($Level) {
            $levelMap = @{
                'Critical' = 1
                'Error' = 2
                'Warning' = 3
                'Information' = 4
                'Verbose' = 5
            }
            $filterXPath += "*[System[Level=$($levelMap[$Level])]]"
        }
        
        if ($After) {
            $filterXPath += "*[System[TimeCreated[@SystemTime>='$($After.ToUniversalTime().ToString('o'))']]]"
        }
        
        if ($Before) {
            $filterXPath += "*[System[TimeCreated[@SystemTime<='$($Before.ToUniversalTime().ToString('o'))']]]"
        }
        
        $filterHashtable = @{
            LogName = $LogName
        }
        
        if ($filterXPath.Count -gt 0) {
            $filterHashtable['XPath'] = $filterXPath -join ' and '
        }
        
        try {
            $events = Get-WinEvent -FilterHashtable $filterHashtable -MaxEvents $Newest -ErrorAction Stop
            return $events | Select-Object TimeCreated, Id, LevelDisplayName, Message
        } catch {
            Write-Error "Failed to retrieve event logs: $_"
            return @()
        }
    }
    
    #endregion
    
    #region Network Information
    
    [array] GetNetworkAdapters() {
        return $this.ExecuteQuery('SELECT * FROM Win32_NetworkAdapter WHERE NetEnabled = TRUE')
    }
    
    [array] GetNetworkAdapterConfiguration([bool]$IPEnabled = $true) {
        $query = 'SELECT * FROM Win32_NetworkAdapterConfiguration'
        if ($IPEnabled) {
            $query += ' WHERE IPEnabled = TRUE'
        }
        return $this.ExecuteQuery($query)
    }
    
    [PSCustomObject] GetNetworkStatistics() {
        $tcpStats = $this.ExecuteQuery('SELECT * FROM Win32_PerfFormattedData_TCPv4_TCPv4') | Select-Object -First 1
        $ipStats = $this.ExecuteQuery('SELECT * FROM Win32_PerfFormattedData_TCPIP_IPv4') | Select-Object -First 1
        
        return [PSCustomObject]@{
            ConnectionsEstablished = $tcpStats.ConnectionsEstablished
            ConnectionsActive = $tcpStats.ConnectionsActive
            ConnectionsPassive = $tcpStats.ConnectionsPassive
            DatagramsReceived = $ipStats.DatagramsReceivedPersec
            DatagramsSent = $ipStats.DatagramsSentPersec
        }
    }
    
    #endregion
    
    #region Helper Methods
    
    [array] ExecuteQuery([string]$query) {
        # Check cache
        $cacheKey = [System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($query))
        $cached = $this.QueryCache[$cacheKey]
        
        if ($cached -and ($cached.Timestamp -gt (Get-Date).AddSeconds(-$this.CacheDurationSeconds))) {
            return $cached.Data
        }
        
        # Execute query
        $result = Get-WmiObject -Query $query -Namespace $this.Namespace -ErrorAction Stop
        
        # Cache result
        $this.QueryCache[$cacheKey] = @{
            Data = $result
            Timestamp = Get-Date
        }
        
        return $result
    }
    
    [void] ClearCache() {
        $this.QueryCache.Clear()
    }
    
    #endregion
}

# CIM Alternative (PowerShell 5.1+)
class CIMProvider {
    [string]$ComputerName
    [pscredential]$Credential
    [Microsoft.Management.Infrastructure.CimSession]$Session
    
    CIMProvider([string]$computerName = 'localhost') {
        $this.ComputerName = $computerName
        $this.InitializeSession()
    }
    
    [void] InitializeSession() {
        $sessionOptions = New-CimSessionOption -Protocol Dcom
        
        $sessionParams = @{
            ComputerName = $this.ComputerName
            SessionOption = $sessionOptions
        }
        
        if ($this.Credential) {
            $sessionParams['Credential'] = $this.Credential
        }
        
        $this.Session = New-CimSession @sessionParams
    }
    
    [array] GetInstances([string]$className, [string]$namespace = 'root/cimv2') {
        return Get-CimInstance -CimSession $this.Session -ClassName $className -Namespace $namespace
    }
    
    [array] Query([string]$query, [string]$namespace = 'root/cimv2') {
        return Get-CimInstance -CimSession $this.Session -Query $query -Namespace $namespace
    }
    
    [void] Dispose() {
        if ($this.Session) {
            Remove-CimSession -CimSession $this.Session
        }
    }
}

# Export functions
function New-WMIProvider {
    param([string]$Namespace = 'root\cimv2')
    return [WMIProvider]::new($Namespace)
}

function New-CIMProvider {
    param([string]$ComputerName = 'localhost')
    return [CIMProvider]::new($ComputerName)
}

Export-ModuleMember -Function New-WMIProvider, New-CIMProvider
