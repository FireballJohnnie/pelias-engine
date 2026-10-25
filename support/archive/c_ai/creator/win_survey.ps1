# win_survey.ps1 - READ-ONLY. Windows-side evidence for GPU driver resets, host low-memory events,
# WSL VM crashes and host hangs. Changes nothing.
#
# Run in a normal 64-bit "Windows PowerShell" window:
#   powershell -NoProfile -ExecutionPolicy Bypass -File <project>\win_survey.ps1
# Optional: add  -Days 10  to look further back.
# A few folders (WER reports, LiveKernelReports) need an Administrator window; the output says when access was denied.
param([int]$Days = 6)

$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
$since = (Get-Date).AddDays(-$Days)
$out = Join-Path '<project>\logs' ('winsurvey_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Force -Path $out | Out-Null
$env:WSL_UTF8 = '1'   # makes wsl.exe print UTF-8 instead of UTF-16

function Save-Events([string]$name, [hashtable]$filter) {
    try {
        Get-WinEvent -FilterHashtable $filter -ErrorAction Stop |
            Select-Object TimeCreated, Id, ProviderName, LevelDisplayName, @{n='Message'; e={ ($_.Message -replace '\s+', ' ') }} |
            Export-Csv -Path (Join-Path $out ($name + '.csv')) -NoTypeInformation -Encoding UTF8
        Write-Host ('[ok]   ' + $name)
    } catch {
        Write-Host ('[none] ' + $name + ' : ' + $_.Exception.Message)
    }
}

# --- event logs ---------------------------------------------------------------------------
Save-Events '01_gpu_driver_resets'     @{ LogName='System'; StartTime=$since; ProviderName='nvlddmkm','Display' }
Save-Events '02_low_memory_host'       @{ LogName='System'; StartTime=$since; ProviderName='Microsoft-Windows-Resource-Exhaustion-Detector' }
Save-Events '03_unexpected_shutdowns'  @{ LogName='System'; StartTime=$since; Id=41,6008,1001,1074 }
Save-Events '04_hardware_errors_whea'  @{ LogName='System'; StartTime=$since; ProviderName='Microsoft-Windows-WHEA-Logger' }
Save-Events '05_app_crashes_hangs'     @{ LogName='Application'; StartTime=$since; ProviderName='Application Error','Application Hang','Windows Error Reporting' }
Save-Events '06_wsl_vm_compute'        @{ LogName='Microsoft-Windows-Hyper-V-Compute-Operational'; StartTime=$since }
Save-Events '07_wsl_vm_worker_admin'   @{ LogName='Microsoft-Windows-Hyper-V-Worker-Admin'; StartTime=$since }

# --- Reliability Monitor records (same data as: perfmon /rel) -------------------------------
try {
    Get-CimInstance Win32_ReliabilityRecords -ErrorAction Stop |
        Where-Object { $_.TimeGenerated -gt $since } |
        Select-Object TimeGenerated, SourceName, EventIdentifier, ProductName, @{n='Message'; e={ ($_.Message -replace '\s+', ' ') }} |
        Export-Csv -Path (Join-Path $out '08_reliability_monitor.csv') -NoTypeInformation -Encoding UTF8
    Write-Host '[ok]   08_reliability_monitor'
} catch { Write-Host ('[none] 08_reliability_monitor : ' + $_.Exception.Message) }

# --- memory, commit limit, pagefile, the WSL VM process -----------------------------------------
$mem = Join-Path $out '10_memory_pagefile_vmmem.txt'
'--- physical memory and commit limit (GB) ---' | Out-File $mem
Get-CimInstance Win32_OperatingSystem |
    Select-Object @{n='TotalRAM_GB'; e={[math]::Round($_.TotalVisibleMemorySize / 1MB, 1)}},
                  @{n='FreeRAM_GB'; e={[math]::Round($_.FreePhysicalMemory / 1MB, 1)}},
                  @{n='CommitLimit_GB'; e={[math]::Round($_.TotalVirtualMemorySize / 1MB, 1)}},
                  @{n='CommitFree_GB'; e={[math]::Round($_.FreeVirtualMemory / 1MB, 1)}} |
    Format-List | Out-File $mem -Append
'--- pagefile (MB; PeakUsage is the highest it has reached since boot) ---' | Out-File $mem -Append
Get-CimInstance Win32_PageFileUsage | Select-Object Name, AllocatedBaseSize, CurrentUsage, PeakUsage |
    Format-Table -AutoSize | Out-String -Width 200 | Out-File $mem -Append
'--- WSL VM process memory right now ---' | Out-File $mem -Append
Get-Process vmmem* -ErrorAction SilentlyContinue |
    Select-Object Name, Id, @{n='WorkingSet_GB'; e={[math]::Round($_.WorkingSet64 / 1GB, 1)}}, @{n='Private_GB'; e={[math]::Round($_.PrivateMemorySize64 / 1GB, 1)}} |
    Format-Table -AutoSize | Out-String -Width 200 | Out-File $mem -Append
'--- top 12 processes by memory ---' | Out-File $mem -Append
Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 12 Name, Id, @{n='WorkingSet_GB'; e={[math]::Round($_.WorkingSet64 / 1GB, 2)}} |
    Format-Table -AutoSize | Out-String -Width 200 | Out-File $mem -Append
'--- drives (free GB) ---' | Out-File $mem -Append
Get-PSDrive -PSProvider FileSystem | Select-Object Name, @{n='Used_GB'; e={[math]::Round($_.Used / 1GB, 1)}}, @{n='Free_GB'; e={[math]::Round($_.Free / 1GB, 1)}} |
    Format-Table -AutoSize | Out-String -Width 200 | Out-File $mem -Append
Write-Host '[ok]   10_memory_pagefile_vmmem'

# --- WSL itself ----------------------------------------------------------------------------
$w = Join-Path $out '11_wsl_info.txt'
'--- wsl --version ---' | Out-File $w;           wsl.exe --version 2>&1 | Out-File $w -Append
'--- wsl -l -v ---' | Out-File $w -Append;       wsl.exe -l -v 2>&1 | Out-File $w -Append
'--- wsl --status ---' | Out-File $w -Append;    wsl.exe --status 2>&1 | Out-File $w -Append
'--- .wslconfig ---' | Out-File $w -Append
Get-Content (Join-Path $env:USERPROFILE '.wslconfig') -ErrorAction SilentlyContinue | Out-File $w -Append
Write-Host '[ok]   11_wsl_info'

# --- NVIDIA, Windows side ------------------------------------------------------------------------
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $g = Join-Path $out '12_nvidia_smi_windows.txt'
    nvidia-smi 2>&1 | Out-File $g
    nvidia-smi -q 2>&1 | Out-File $g -Append
    Write-Host '[ok]   12_nvidia_smi_windows'
}

# --- crash report folders and dump files (names only; nothing is opened) ------------------------------------
$rep = Join-Path $out '13_wer_reports_and_dumps.txt'
'--- WER report folders (the name hints at the type, e.g. Kernel_141 = GPU hang, AppCrash_python = python died) ---' | Out-File $rep
foreach ($d in 'C:\ProgramData\Microsoft\Windows\WER\ReportArchive', 'C:\ProgramData\Microsoft\Windows\WER\ReportQueue') {
    if (Test-Path $d) {
        try {
            Get-ChildItem $d -Directory -ErrorAction Stop | Where-Object { $_.LastWriteTime -gt $since } | Sort-Object LastWriteTime |
                ForEach-Object { '{0}  {1}' -f $_.LastWriteTime.ToString('yyyy-MM-dd HH:mm'), $_.Name } | Out-File $rep -Append
        } catch { ('access denied (try an Administrator window): ' + $d) | Out-File $rep -Append }
    }
}
'--- kernel / GPU watchdog dump files ---' | Out-File $rep -Append
foreach ($d in 'C:\Windows\LiveKernelReports', 'C:\Windows\Minidump') {
    if (Test-Path $d) {
        try {
            Get-ChildItem $d -Recurse -File -ErrorAction Stop | Where-Object { $_.LastWriteTime -gt $since } |
                Select-Object LastWriteTime, Length, FullName | Format-Table -AutoSize | Out-String -Width 250 | Out-File $rep -Append
        } catch { ('access denied (try an Administrator window): ' + $d) | Out-File $rep -Append }
    }
}
Write-Host '[ok]   13_wer_reports_and_dumps'

Write-Host ''
Write-Host ('Done. Files are in: ' + $out)
Get-ChildItem $out | Select-Object Name, Length | Format-Table -AutoSize
