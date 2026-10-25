<#
.SYNOPSIS
  Windows side of the pelias probe suite: read-only machine facts, then (optionally) build and run the probes.

.DESCRIPTION
  What it does:   reads system facts, the display-driver event log and per-app GPU memory; looks for the build tools;
                  builds pelias_probe.exe with them; runs the same probes as the WSL script; saves everything to a folder.
  What it never does: change any setting, install anything, write to the registry, or delete anything.

  Run it like this (the -ExecutionPolicy flag applies to this one run only and changes no setting):
    powershell -ExecutionPolicy Bypass -File .\run_windows_all.ps1 -Mode facts      # step 1: facts only, nothing built, no GPU load
    powershell -ExecutionPolicy Bypass -File .\run_windows_all.ps1 -Mode quick      # step 2: build + short pass over every probe
    powershell -ExecutionPolicy Bypass -File .\run_windows_all.ps1 -Mode full       # the real measurements (about 70 minutes)
    powershell -ExecutionPolicy Bypass -File .\run_windows_all.ps1 -Mode tdr        # display-timeout test, asks before each stage

  Useful options:  -Gguf <path to model file>   -Only "h5 h6"   -Skip "m3 h4"   -Tag stock   -Yes   -NoBuild
  Memory-spike test:  -Settle           before every probe, wait (up to -SettleMaxSec, default 60) until GPU memory used is back to the
                                        run's starting level + -VramTolMiB (default 64): the "flush between runs" test
                      -RetryOnFail      if a short probe fails (exit 2 or 3), rerun it plain and then settled and say which one passes
                                        (h4, m3 and h13 are never retried: too long). Results are tagged retry_plain / retry_settled.
                      Always on: vram_log.tsv in the results folder (GPU memory used and free before and after every probe).
#>
[CmdletBinding()]
param(
  [ValidateSet('facts', 'quick', 'full', 'tdr')]
  [string]$Mode = 'quick',
  [string]$Gguf = '',
  [string]$Only = '',
  [string]$Skip = '',
  [string]$Tag = '',
  [string]$OutRoot = 'C:\AI\cl-sndbx-bkp\stuart-creator\2026-10-23\probe_results\windows',
  [string]$Arch = 'sm_89',
  [int]$EventHours = 24,
  [string]$Exe = '',
  [switch]$Yes,
  [switch]$NoBuild,
  [switch]$NoFacts,
  [switch]$Settle,
  [switch]$RetryOnFail,
  [int]$SettleMaxSec = 60,
  [int]$VramTolMiB = 64
)

$ErrorActionPreference = 'Continue'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$RunName = "${Stamp}_${Mode}"
if ($Tag) { $RunName = "${RunName}_${Tag}" }
$RunStart = Get-Date

# ------------------------------------------------------------------ output folder
$Out = Join-Path $OutRoot $RunName
try {
  New-Item -ItemType Directory -Force -Path $Out -ErrorAction Stop | Out-Null
} catch {
  $fallbackRoot = Join-Path $env:USERPROFILE 'pelias_probe_results'
  $Out = Join-Path $fallbackRoot $RunName
  New-Item -ItemType Directory -Force -Path $Out | Out-Null
  Write-Host "Could not create $OutRoot - using $Out instead."
}
$LogPath = Join-Path $Out 'run.log'
$FactsPath = Join-Path $Out 'facts_windows.txt'
$SummaryPath = Join-Path $Out 'summary.tsv'
Set-Content -Path $SummaryPath -Value "probe`texit_code`tseconds" -Encoding UTF8

function Say([string]$Text) {
  Write-Host $Text
  Add-Content -Path $LogPath -Value $Text -Encoding UTF8
}

# ------------------------------------------------------------------ small helpers
function Get-Sha256Hex([byte[]]$Bytes) {
  $h = [System.Security.Cryptography.SHA256]::Create()
  return (($h.ComputeHash($Bytes) | ForEach-Object { $_.ToString('x2') }) -join '')
}

function Get-SourcesSha {
  $files = Get-ChildItem -Path $Here -File | Where-Object { $_.Extension -in '.h', '.cpp', '.cu' } | Sort-Object Name
  $ms = New-Object System.IO.MemoryStream
  foreach ($f in $files) {
    $b = [System.IO.File]::ReadAllBytes($f.FullName)
    $ms.Write($b, 0, $b.Length)
  }
  return (Get-Sha256Hex $ms.ToArray()).Substring(0, 16)
}

function Split-Words([string]$Text) {
  if (-not $Text) { return @() }
  return @($Text -split '[\s,]+' | Where-Object { $_ })
}
$OnlyList = Split-Words $Only
$SkipList = Split-Words $Skip
$script:TensorBlocked = $false
function Wanted([string]$Name) {
  if ($script:TensorBlocked -and ($Name -eq 'h3' -or $Name -eq 'h10' -or $Name -eq 'h4')) { return $false }
  if ($OnlyList.Count -gt 0) { return ($OnlyList -contains $Name) }
  return (-not ($SkipList -contains $Name))
}

# ------------------------------------------------------------------ tools
$NvSmi = $null
$cmdNv = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if ($cmdNv) { $NvSmi = $cmdNv.Source }
elseif (Test-Path 'C:\Windows\System32\nvidia-smi.exe') { $NvSmi = 'C:\Windows\System32\nvidia-smi.exe' }

function Find-Nvcc {
  $c = Get-Command nvcc.exe -ErrorAction SilentlyContinue
  if ($c) { return $c.Source }
  if ($env:CUDA_PATH) {
    $p = Join-Path $env:CUDA_PATH 'bin\nvcc.exe'
    if (Test-Path $p) { return $p }
  }
  $root = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
  if (Test-Path $root) {
    $d = Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
    if ($d) {
      $p = Join-Path $d.FullName 'bin\nvcc.exe'
      if (Test-Path $p) { return $p }
    }
  }
  return $null
}

function Find-VsDevCmd {
  $pf86 = ${env:ProgramFiles(x86)}
  if (-not $pf86) { return $null }
  $vw = Join-Path $pf86 'Microsoft Visual Studio\Installer\vswhere.exe'
  if (-not (Test-Path $vw)) { return $null }
  $inst = & $vw -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  if (-not $inst) { return $null }
  $dev = Join-Path ([string]$inst) 'Common7\Tools\VsDevCmd.bat'
  if (Test-Path $dev) { return $dev }
  return $null
}

function Import-VsEnvironment([string]$DevCmd) {
  # Runs VsDevCmd in a helper batch file and copies the resulting variables into THIS PowerShell process only.
  $bat = Join-Path $Out 'vsenv.cmd'
  Set-Content -Path $bat -Encoding ASCII -Value @('@echo off', ('call "' + $DevCmd + '" -arch=x64 -host_arch=x64 -no_logo >nul 2>&1'), 'set')
  $lines = & cmd.exe /c $bat
  foreach ($l in $lines) {
    if ($l -match '^([^=]+)=(.*)$') {
      [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
  }
}

# ------------------------------------------------------------------ facts (read-only)
function Add-Fact([string]$Title, [scriptblock]$Body) {
  Add-Content -Path $FactsPath -Value "### $Title" -Encoding UTF8
  try {
    $text = ((& $Body 2>&1) | Out-String -Width 220).TrimEnd()
  } catch {
    $text = "(failed: $($_.Exception.Message))"
  }
  Add-Content -Path $FactsPath -Value $text -Encoding UTF8
  Add-Content -Path $FactsPath -Value '' -Encoding UTF8
}

function Save-GpuEvents([datetime]$Since, [datetime]$Until, [string]$Path) {
  $providers = @('nvlddmkm', 'Display', 'Microsoft-Windows-WHEA-Logger', 'Microsoft-Windows-Kernel-Power', 'Microsoft-Windows-DxgKrnl')
  $rows = @()
  $notes = @()
  foreach ($prov in $providers) {
    try {
      $ev = Get-WinEvent -FilterHashtable @{ LogName = 'System'; ProviderName = $prov; StartTime = $Since; EndTime = $Until } -ErrorAction Stop
      foreach ($e in $ev) {
        $msg = ($e.Message -replace '\s+', ' ')
        if ($msg.Length -gt 220) { $msg = $msg.Substring(0, 220) }
        $rows += [pscustomobject]@{ Time = $e.TimeCreated; Line = ('{0:yyyy-MM-dd HH:mm:ss}  id={1}  {2}  [{3}]  {4}' -f $e.TimeCreated, $e.Id, $e.ProviderName, $e.LevelDisplayName, $msg) }
      }
    } catch {
      if ($_.Exception.Message -notmatch 'No events were found') { $notes += "(query for $prov failed: $($_.Exception.Message))" }
    }
  }
  $lines = @("# System log events from: $($providers -join ', ')", "# window: $Since  to  $Until")
  $sorted = @($rows | Sort-Object Time)
  if ($sorted.Count -eq 0) { $lines += '(no events from these providers in this window)' }
  foreach ($r in $sorted) { $lines += $r.Line }
  $lines += $notes
  Set-Content -Path $Path -Value $lines -Encoding UTF8
}

function Collect-Facts {
  Set-Content -Path $FactsPath -Value "# Windows facts - read only - $(Get-Date -Format o)" -Encoding UTF8
  Add-Fact 'Operating system' { Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber, OSArchitecture, @{n = 'TotalMemoryGiB'; e = { [math]::Round($_.TotalVisibleMemorySize / 1MB, 1) } }, @{n = 'FreeMemoryGiB'; e = { [math]::Round($_.FreePhysicalMemory / 1MB, 1) } } | Format-List }
  Add-Fact 'CPU' { Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed, L2CacheSize, L3CacheSize | Format-List }
  Add-Fact 'Memory modules' { Get-CimInstance Win32_PhysicalMemory | Select-Object BankLabel, @{n = 'GiB'; e = { [math]::Round($_.Capacity / 1GB, 0) } }, Speed, ConfiguredClockSpeed, Manufacturer, PartNumber | Format-Table -AutoSize }
  Add-Fact 'Graphics adapters' { Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, DriverDate, CurrentRefreshRate, VideoProcessor | Format-List }
  Add-Fact 'Page file' {
    Get-CimInstance Win32_PageFileUsage | Select-Object Name, AllocatedBaseSize, CurrentUsage, PeakUsage | Format-List
    Get-CimInstance Win32_ComputerSystem | Select-Object AutomaticManagedPagefile, @{n = 'TotalPhysicalGiB'; e = { [math]::Round($_.TotalPhysicalMemory / 1GB, 1) } } | Format-List
  }
  Add-Fact 'Power plan' { powercfg /getactivescheme }
  Add-Fact 'Display-timeout (TDR) and GPU scheduling settings - read only; "(not set)" means the Windows default (TdrDelay default is 2 seconds)' {
    $p = Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers' -ErrorAction Stop
    foreach ($n in 'HwSchMode', 'TdrDelay', 'TdrLevel', 'TdrDdiDelay', 'TdrLimitTime', 'TdrLimitCount') {
      $v = $p.$n
      if ($null -eq $v) { "$n = (not set)" } else { "$n = $v" }
    }
  }
  if ($NvSmi) {
    Add-Fact 'nvidia-smi (summary)' { & $NvSmi }
    Add-Fact 'nvidia-smi (csv query)' { & $NvSmi '--query-gpu=name,driver_version,pstate,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,power.draw,power.limit,temperature.gpu,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,memory.total,memory.used,memory.free,utilization.gpu' '--format=csv' }
    Add-Fact 'nvidia-smi -q (full, includes BAR1 and PCIe sections)' { & $NvSmi -q }
    Add-Fact 'nvidia-smi compute apps' { & $NvSmi '--query-compute-apps=pid,process_name,used_memory' '--format=csv' }
  } else {
    Add-Fact 'nvidia-smi' { 'nvidia-smi.exe not found' }
  }
  Add-Fact 'GPU memory by process (Windows performance counters - shows who holds VRAM right now)' {
    $ded = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples
    $shr = (Get-Counter '\GPU Process Memory(*)\Shared Usage' -ErrorAction Stop).CounterSamples
    $rows = @{}
    foreach ($s in $ded) {
      if ($s.InstanceName -match 'pid_(\d+)_') {
        $id = [int]$Matches[1]
        if (-not $rows.ContainsKey($id)) { $rows[$id] = @{ Ded = 0.0; Shr = 0.0 } }
        $rows[$id].Ded += $s.CookedValue
      }
    }
    foreach ($s in $shr) {
      if ($s.InstanceName -match 'pid_(\d+)_') {
        $id = [int]$Matches[1]
        if (-not $rows.ContainsKey($id)) { $rows[$id] = @{ Ded = 0.0; Shr = 0.0 } }
        $rows[$id].Shr += $s.CookedValue
      }
    }
    $rows.GetEnumerator() | ForEach-Object {
      $proc = Get-Process -Id $_.Key -ErrorAction SilentlyContinue
      $pname = '(gone)'
      if ($proc) { $pname = $proc.ProcessName }
      [pscustomobject]@{ PID = $_.Key; Process = $pname; DedicatedMiB = [math]::Round($_.Value.Ded / 1MB, 1); SharedMiB = [math]::Round($_.Value.Shr / 1MB, 1) }
    } | Sort-Object DedicatedMiB -Descending | Select-Object -First 25 | Format-Table -AutoSize
  }
  Add-Fact 'GPU adapter memory totals (counters)' {
    $a = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples
    $a | ForEach-Object { [pscustomobject]@{ Adapter = $_.InstanceName; DedicatedMiB = [math]::Round($_.CookedValue / 1MB, 1) } } | Format-Table -AutoSize
  }
  Add-Fact 'WSL version' {
    $old = [Console]::OutputEncoding
    try {
      [Console]::OutputEncoding = [System.Text.Encoding]::Unicode
      & wsl.exe --version
    } finally {
      [Console]::OutputEncoding = $old
    }
  }
  Add-Fact '.wslconfig (read only)' {
    $f = Join-Path $env:USERPROFILE '.wslconfig'
    if (Test-Path $f) { Get-Content $f } else { '(no .wslconfig)' }
  }
  Save-GpuEvents (Get-Date).AddHours(-$EventHours) (Get-Date) (Join-Path $Out "events_last_${EventHours}h.txt")
}

function Get-StableState([string]$NvccPath) {
  $os = Get-CimInstance Win32_OperatingSystem
  $cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
  $mem = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
  $gpu = ''
  if ($NvSmi) { $gpu = [string](& $NvSmi '--query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.limit' '--format=csv,noheader' | Select-Object -First 1) }
  $tdr = ''
  try {
    $p = Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers' -ErrorAction Stop
    $tdr = "HwSchMode=$($p.HwSchMode);TdrDelay=$($p.TdrDelay);TdrLevel=$($p.TdrLevel)"
  } catch { $tdr = 'unreadable' }
  $nv = ''
  if ($NvccPath) { $nv = [string](& $NvccPath --version | Select-Object -Last 1) }
  return (@("os_build=$($os.BuildNumber)", "cpu=$cpu", "mem_total_gib=$mem", "gpu=$gpu", "tdr=$tdr", "nvcc=$nv", "probe_sources_sha=$(Get-SourcesSha)") -join "`n")
}

# ------------------------------------------------------------------ go: facts
Say "run: $RunName   (results folder: $Out)"
if (-not $NoFacts) {
  Say 'Collecting read-only Windows facts ...'
  Collect-Facts
  Say "Facts saved: $FactsPath"
}

$Nvcc = Find-Nvcc
$VsDev = Find-VsDevCmd
Say ''
Say 'Build tools:'
Say ('  nvcc (CUDA toolkit): ' + $(if ($Nvcc) { $Nvcc } else { 'NOT FOUND' }))
Say ('  Visual Studio C++ tools: ' + $(if ($VsDev) { $VsDev } else { 'NOT FOUND' }))
Say ('  nvidia-smi: ' + $(if ($NvSmi) { $NvSmi } else { 'NOT FOUND' }))
Add-Content -Path $FactsPath -Value "### Build tools`nnvcc: $Nvcc`nVsDevCmd: $VsDev`n" -Encoding UTF8

$StableText = ''
try { $StableText = Get-StableState $Nvcc } catch { $StableText = "state unreadable: $($_.Exception.Message)" }
Set-Content -Path (Join-Path $Out 'machine_state.txt') -Value $StableText -Encoding UTF8
$Sha = (Get-Sha256Hex ([System.Text.Encoding]::UTF8.GetBytes($StableText))).Substring(0, 16)
Say "machine_state_sha: $Sha"

if ($Mode -eq 'facts') {
  Save-GpuEvents $RunStart (Get-Date) (Join-Path $Out 'events_during_run.txt')
  Say ''
  Say 'Facts only - nothing was built or run. Tell Claude it is done.'
  exit 0
}

# ------------------------------------------------------------------ build
if ($Exe) { $ExePath = $Exe } else { $ExePath = Join-Path $Here 'build\pelias_probe.exe' }

function Invoke-Step([string]$Label, [string]$Tool, [string[]]$ToolArgs) {
  Say "  $Label"
  $buildLog = Join-Path $Out 'build.log'
  & $Tool @ToolArgs 2>&1 | Tee-Object -FilePath $buildLog -Append | Out-Host
  if ($LASTEXITCODE -ne 0) { throw "$Label failed (exit code $LASTEXITCODE). See $buildLog" }
}

function Build-Probes {
  if (-not $Nvcc) { return 'nvcc not found' }
  if (-not $VsDev) { return 'Visual Studio C++ build tools not found' }
  Import-VsEnvironment $VsDev
  if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { return 'cl.exe not available after loading the Visual Studio environment' }
  $bd = Join-Path $Here 'build'
  New-Item -ItemType Directory -Force -Path $bd | Out-Null
  $cudaInc = Join-Path (Split-Path (Split-Path $Nvcc -Parent) -Parent) 'include'
  Push-Location $Here
  try {
    Say 'Building:'
    Invoke-Step '[1/4] kernels.cu (nvcc)' $Nvcc @('-std=c++20', '-O2', "-arch=$Arch", '-c', 'kernels.cu', '-o', (Join-Path $bd 'kernels.obj'))
    Invoke-Step '[2/4] cpu_bw.cpp (cl /arch:AVX2)' 'cl.exe' @('/nologo', '/std:c++20', '/O2', '/EHsc', '/arch:AVX2', '/c', 'cpu_bw.cpp', ('/Fo' + (Join-Path $bd 'cpu_bw.obj')))
    Invoke-Step '[3/4] main.cpp (cl)' 'cl.exe' @('/nologo', '/std:c++20', '/O2', '/EHsc', '/D_CRT_SECURE_NO_WARNINGS', ('/I' + $cudaInc), '/c', 'main.cpp', ('/Fo' + (Join-Path $bd 'main.obj')))
    Invoke-Step '[4/4] link (nvcc)' $Nvcc @("-arch=$Arch", (Join-Path $bd 'main.obj'), (Join-Path $bd 'cpu_bw.obj'), (Join-Path $bd 'kernels.obj'), '-o', $ExePath)
  } catch {
    Pop-Location
    return $_.Exception.Message
  }
  Pop-Location
  return ''
}

if (-not $NoBuild -and -not $Exe) {
  $err = Build-Probes
  if ($err) {
    Say ''
    Say "Could not build: $err"
    Say 'The facts above are still saved. To build, these are needed (this script installs nothing):'
    Say '  1. Visual Studio 2022 Build Tools, workload "Desktop development with C++"'
    Say '  2. NVIDIA CUDA Toolkit 13.0 for Windows - choose Custom and tick only the CUDA development parts; do not reinstall the display driver'
    Say 'Then run this script again.'
    exit 2
  }
}
if (-not (Test-Path $ExePath)) {
  Say "Program not found: $ExePath  (build first, or pass -Exe)"
  exit 2
}
if ($Gguf) { $env:PELIAS_GGUF = $Gguf }

# ------------------------------------------------------------------ running probes
function Show-NewText([string]$Path, [long]$Pos) {
  if (-not (Test-Path $Path)) { return $Pos }
  try {
    $fs = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
    try {
      if ($fs.Length -le $Pos) { return $Pos }
      [void]$fs.Seek($Pos, [System.IO.SeekOrigin]::Begin)
      $sr = New-Object System.IO.StreamReader($fs)
      $text = $sr.ReadToEnd()
      $newPos = $fs.Length
      if ($text.Length -gt 0) {
        Write-Host -NoNewline $text
        Add-Content -Path $LogPath -Value $text.TrimEnd("`r", "`n") -Encoding UTF8
      }
      return $newPos
    } finally {
      $fs.Dispose()
    }
  } catch {
    return $Pos
  }
}

function Run-ProbeOnce([string]$Name, [int]$TimeoutSec, [string[]]$Extra, [string]$RunTag) {
  $label = $Name
  if ($RunTag -match 'retry_(plain|settled)$') { $label = "$Name [$RunTag]" }
  Say ''
  Say ("=== {0}  ({1}, timeout {2}s)" -f $label, (Get-Date -Format 'HH:mm:ss'), $TimeoutSec)
  Note-Vram 'before' $Name
  $argList = @($Name)
  if ($Mode -eq 'quick') { $argList += '--quick' }
  $argList += $Extra
  $argList += @('--out', $Out, '--machine-sha', $Sha)
  if ($RunTag) { $argList += @('--tag', $RunTag) }
  $quoted = $argList | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
  $argLine = $quoted -join ' '
  $tag2 = $Name
  if ($Extra.Count -ge 2 -and $Extra[0] -eq '--stage') { $tag2 = "${Name}_$($Extra[1])" }
  if ($RunTag -match 'retry_(plain|settled)$') { $tag2 = "${tag2}_$($Matches[0])" }
  $so = Join-Path $Out "stdout_$tag2.txt"
  $se = Join-Path $Out "stderr_$tag2.txt"
  $t0 = Get-Date
  $p = Start-Process -FilePath $ExePath -ArgumentList $argLine -NoNewWindow -PassThru -RedirectStandardOutput $so -RedirectStandardError $se
  $null = $p.Handle
  $pos = [long]0
  $timedOut = $false
  while (-not $p.HasExited) {
    Start-Sleep -Milliseconds 700
    $pos = Show-NewText $so $pos
    if (((Get-Date) - $t0).TotalSeconds -gt $TimeoutSec) {
      $timedOut = $true
      try { $p.Kill() } catch { }
      break
    }
  }
  [void]$p.WaitForExit(30000)
  $pos = Show-NewText $so $pos
  if ((Test-Path $se) -and ((Get-Item $se).Length -gt 0)) {
    $errText = (Get-Content -Path $se -Raw)
    Say ('[stderr] ' + $errText.TrimEnd())
  }
  $code = 0
  if ($timedOut) {
    $code = 124
    Say "[TIMEOUT] $Name was stopped after ${TimeoutSec}s"
  } else {
    $code = $p.ExitCode
    if ($code -ne 0 -and $code -ne 3 -and $code -ne 4) { Say "[EXIT $code] $Name ended abnormally" }
  }
  Note-Vram 'after' $Name
  $secs = [int]((Get-Date) - $t0).TotalSeconds
  Add-Content -Path $SummaryPath -Value ("{0}`t{1}`t{2}" -f $label, $code, $secs) -Encoding UTF8
  return [int]$code
}

# >>> GPU MEMORY LOG, SETTLE, RETRY
$VramLog = Join-Path $Out 'vram_log.tsv'
Set-Content -Path $VramLog -Value "time`tevent`tprobe`tused_mib`tfree_mib`ttotal_mib" -Encoding UTF8
$script:VramUsed = $null
$script:VramBase = $null
function Note-Vram([string]$Event, [string]$Probe) {
  $u = '?'; $f = '?'; $tt = '?'
  if ($NvSmi) {
    try {
      $line = & $NvSmi '--query-gpu=memory.used,memory.free,memory.total' '--format=csv,noheader,nounits' 2>$null | Select-Object -First 1
      $parts = ("$line" -replace '\s', '') -split ','
      if ($parts.Count -ge 3) { $u = $parts[0]; $f = $parts[1]; $tt = $parts[2] }
    } catch { }
  }
  Add-Content -Path $VramLog -Value ("{0}`t{1}`t{2}`t{3}`t{4}`t{5}" -f (Get-Date -Format 'HH:mm:ss'), $Event, $Probe, $u, $f, $tt) -Encoding UTF8
  if ($u -match '^\d+$') { $script:VramUsed = [int]$u } else { $script:VramUsed = $null }
}

function Wait-Settle([string]$Probe) {
  if ($null -eq $script:VramBase) {
    Say "[SETTLE] ${Probe}: no starting level recorded (nvidia-smi gave no reading) - not waiting"
    return $false
  }
  Note-Vram 'settle_start' $Probe
  $waited = 0
  while (-not (($null -ne $script:VramUsed) -and ($script:VramUsed -le ($script:VramBase + $VramTolMiB)))) {
    if ($waited -ge $SettleMaxSec) {
      Note-Vram 'settle_timeout' $Probe
      Say ("[SETTLE] {0}: GPU memory used is still {1} MiB (starting level {2} MiB) after {3}s - going on" -f $Probe, $script:VramUsed, $script:VramBase, $SettleMaxSec)
      return $false
    }
    Start-Sleep -Seconds 2
    $waited += 2
    Note-Vram 'settle_wait' $Probe
  }
  Start-Sleep -Seconds 1
  Note-Vram 'settle_done' $Probe
  Say ("[SETTLE] {0}: GPU memory back to the starting level ({1} MiB, start {2} MiB) after {3}s" -f $Probe, $script:VramUsed, $script:VramBase, $waited)
  return $true
}

function Retry-Probe([string]$Name, [int]$TimeoutSec, [int]$Rc0, [string[]]$Extra) {
  $pre = ''
  if ($Tag) { $pre = "${Tag}_" }
  Say "[RETRY] $Name failed (exit $Rc0). Rerunning it twice to see whether a memory spike or leftover memory was the cause."
  Note-Vram 'retry_start' $Name
  Say "[RETRY] ${Name}: plain rerun, no waiting"
  $rcp = Run-ProbeOnce $Name $TimeoutSec $Extra "${pre}retry_plain"
  Say "[RETRY] ${Name}: settled rerun, waiting first for GPU memory to return to the starting level"
  [void](Wait-Settle $Name)
  $rcs = Run-ProbeOnce $Name $TimeoutSec $Extra "${pre}retry_settled"
  if ($rcp -eq 0) { $verdict = 'passed on the plain rerun: a short outside spike (or just the time passing) is the likely cause, not leftover memory from the earlier probe' }
  elseif ($rcs -eq 0) { $verdict = 'passed only after settling: memory still held after the earlier probe was the cause, so a flush between runs helps' }
  else { $verdict = 'failed every time: a real limit, not a spike' }
  Say "[RETRY] ${Name}: first=$Rc0 plain=$rcp settled=$rcs -> $verdict"
  $retryLog = Join-Path $Out 'retry_log.tsv'
  if (-not (Test-Path $retryLog)) { Set-Content -Path $retryLog -Value "probe`tfirst_exit`tplain_exit`tsettled_exit`tverdict" -Encoding UTF8 }
  Add-Content -Path $retryLog -Value ("{0}`t{1}`t{2}`t{3}`t{4}" -f $Name, $Rc0, $rcp, $rcs, $verdict) -Encoding UTF8
}

function Run-Probe([string]$Name, [int]$TimeoutSec, [string[]]$Extra = @()) {
  if ($Settle) { [void](Wait-Settle $Name) }
  $rc = Run-ProbeOnce $Name $TimeoutSec $Extra $Tag
  if ($RetryOnFail -and ($rc -eq 2 -or $rc -eq 3)) {
    if (@('h4', 'm3', 'h13') -contains $Name) {
      Say "[RETRY] $Name failed (exit $rc) but is a long probe, so it is not rerun automatically"
    } else {
      Retry-Probe $Name $TimeoutSec $rc $Extra
    }
  }
  return [int]$rc    # the first result is what the rest of the script acts on
}
# <<< GPU MEMORY LOG, SETTLE, RETRY

function Confirm-Start([string]$Prompt) {
  if ($Yes) { return }
  [void](Read-Host "$Prompt [Enter to start, Ctrl-C to cancel]")
}

# timeouts (seconds)
if ($Mode -eq 'quick') { $tShort = 300; $tH4 = 300; $tMem = 600; $tM3 = 600 } else { $tShort = 900; $tH4 = 2700; $tMem = 1800; $tM3 = 3000 }

$mem = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)
Say ''
Say "Free memory now: $mem GiB. Close browsers, games, video and any other GPU or LLM program before this run."
if ($Mode -eq 'full' -and $mem -lt 35) { Say 'WARNING: under 35 GiB free - the pinned-memory probes (P-H7, P-M3) will stop early to protect the machine.' }
if (-not $Gguf -and -not $env:PELIAS_GGUF) { Say 'Note: no -Gguf given, so the model-file read test (m4) will be skipped.' }

Note-Vram 'baseline' 'run'
$script:VramBase = $script:VramUsed

if ($Mode -eq 'tdr') {
  Say ''
  Say 'DISPLAY-TIMEOUT TEST. Each stage runs one long GPU kernel. Windows may reset the display driver: the screen can flicker or go black for a few seconds and programs using the GPU can crash. Save your work first.'
  foreach ($st in @('2s', '5s', '30s', 'queued', 'graph')) {
    if (-not (Wanted "h13_$st")) { continue }
    $ans = Read-Host "Run stage '$st'? Type YES to run, anything else to stop"
    if ($ans -ne 'YES') { Say "stopped before stage $st"; break }
    $rc = Run-Probe 'h13' 240 @('--stage', $st)
    if ($rc -ne 0) { Say "Stopped: stage $st did not pass. Do not run the longer stages. Results are saved."; break }
  }
} else {
  if ($Mode -eq 'full') { Confirm-Start 'About to run the FULL suite (about 70 minutes, runs the GPU hard).' }
  $rcSelf = Run-Probe 'selftest' 300
  if ($rcSelf -eq 4) {
    $script:TensorBlocked = $true
    Say 'The tensor-core self-checks failed (everything else passed). Skipping h3, h10 and h4 and running the rest.'
  }
  if ($rcSelf -ne 0 -and $rcSelf -ne 4) {
    Say 'The self-test failed, so the measurements cannot be trusted. Stopping.'
  } else {
    if (Wanted 'state') { [void](Run-Probe 'state' 60) }
    if (Wanted 'h8') { [void](Run-Probe 'h8' 120) }
    if (Wanted 'h1') { [void](Run-Probe 'h1' 120) }
    if (Wanted 'h2') { [void](Run-Probe 'h2' $tShort) }
    if (Wanted 'h3') { [void](Run-Probe 'h3' $tShort) }
    if (Wanted 'h10') { [void](Run-Probe 'h10' $tShort) }
    if (Wanted 'h4') { [void](Run-Probe 'h4' $tH4) }
    if (Wanted 'h5') { [void](Run-Probe 'h5' $tShort) }
    if (Wanted 'h6') { [void](Run-Probe 'h6' $tShort) }
    if (Wanted 'h12') { [void](Run-Probe 'h12' $tShort) }
    if (Wanted 'h7') { [void](Run-Probe 'h7' $tMem) }
    if (Wanted 'm3') { [void](Run-Probe 'm3' $tM3) }
    if (Wanted 'm4') { [void](Run-Probe 'm4' $tMem) }
    if (Wanted 'h9') { [void](Run-Probe 'h9' $tShort) }
  }
}

# ------------------------------------------------------------------ after the run
Save-GpuEvents $RunStart (Get-Date) (Join-Path $Out 'events_during_run.txt')
Say ''
Say "================ summary ($RunName)"
$sumLines = Select-String -Path $LogPath -Pattern '^\[(PASS|FAIL|REC|ERROR|SKIP|TIMEOUT|EXIT|RETRY|SETTLE)' | ForEach-Object { $_.Line }
foreach ($l in $sumLines) { Write-Host $l }
Set-Content -Path (Join-Path $Out 'summary.txt') -Value $sumLines -Encoding UTF8
Say ''
Say "Results folder: $Out"
Say 'Events from the display driver during this run: events_during_run.txt (look for Display id 4101 or nvlddmkm entries).'
Say 'Tell Claude when the run is done.'
