# ============================================================
#  AI Xiaofang - Python bootstrap   (xiaofang_bootstrap.ps1)
# ------------------------------------------------------------
#  Kept 100% ASCII on purpose: Windows PowerShell 5.1 decodes a
#  .ps1 file without a BOM using the active ANSI code page, so any
#  non-ASCII character in here can turn into garbage and break
#  parsing. Every Chinese message the user reads lives in
#  xiaofang_launcher.py instead (Python always reads its source as
#  UTF-8), and that file runs right after this one finishes.
#
#  Called by the launcher .bat, but only when no usable Python was
#  found on the machine:
#    1. picks a python.org installer that matches the CPU
#    2. downloads it into %TEMP%\xiaofang_python_setup
#    3. installs it silently, per-user, no UAC prompt, no shortcuts
#    4. reports where the fresh python.exe landed
#
#  Manual use (mainly for debugging)
#    powershell -NoProfile -ExecutionPolicy Bypass -File xiaofang_bootstrap.ps1 -Arch amd64
#      -Arch amd64 | arm64 | win32   which installer flavour to fetch
#      -DryRun                       resolve + reachability only, no download
#
#  Exit codes
#    0   python.exe is available after the run
#    3   nothing could be downloaded
#    4   installer ran, but python.exe is still not where we expect it
# ============================================================

[CmdletBinding()]
param(
    [ValidateSet('amd64', 'arm64', 'win32')]
    [string]$Arch = 'amd64',

    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
# the progress bar makes Invoke-WebRequest crawl on big files
$ProgressPreference = 'SilentlyContinue'
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch { }

# ---- which versions to try, in order --------------------------------
#  3.12.x is what this project is tuned for, so those come first; the
#  newer minors are only a fallback for machines where 3.12 is gone.
#  Every entry below was reachability-checked on python.org.
$Candidates = @('3.12.10', '3.12.9', '3.13.14', '3.13.7', '3.14.6')

$Base = 'https://www.python.org/ftp/python'
$TmpDir = Join-Path $env:TEMP 'xiaofang_python_setup'
$Log = Join-Path $TmpDir 'bootstrap.log'

if (-not (Test-Path $TmpDir)) {
    New-Item -ItemType Directory -Path $TmpDir -Force | Out-Null
}

function Say {
    param([string]$Message)
    Write-Host ('   [env] ' + $Message)
    try {
        Add-Content -Path $Log -Value ((Get-Date).ToString('HH:mm:ss') + '  ' + $Message)
    } catch { }
}

Say ('machine architecture : ' + $Arch)
Say ('temp folder          : ' + $TmpDir)

# ---- 1) download an installer that actually exists -------------------
$Setup = $null
foreach ($v in $Candidates) {
    $leaf = 'python-' + $v + '-' + $Arch + '.exe'
    $url = $Base + '/' + $v + '/' + $leaf
    $dst = Join-Path $TmpDir $leaf

    if ($DryRun) {
        try {
            $r = Invoke-WebRequest -Uri $url -Method Head -UseBasicParsing -TimeoutSec 25
            $mb = [math]::Round([double]$r.Headers['Content-Length'] / 1MB, 2)
            Say ('candidate ' + $v + '  ->  HTTP ' + $r.StatusCode + '  ' + $mb + ' MB')
            Say ('url       ' + $url)
            $Setup = 'dryrun'
            break
        } catch {
            Say ('candidate ' + $v + '  ->  not available')
            continue
        }
    }

    Say ('trying ' + $v + '  (' + $leaf + ')')
    if (Test-Path $dst) { Remove-Item $dst -Force -ErrorAction SilentlyContinue }
    try {
        Invoke-WebRequest -Uri $url -OutFile $dst -UseBasicParsing -TimeoutSec 900
    } catch {
        Say ('  download failed: ' + $_.Exception.Message)
        continue
    }
    if ((Test-Path $dst) -and ((Get-Item $dst).Length -gt 5MB)) {
        $Setup = $dst
        Say ('  downloaded ' + $leaf + '  (' + [math]::Round((Get-Item $dst).Length / 1MB, 2) + ' MB)')
        break
    }
    Say '  downloaded file looks too small, trying the next version'
}

if (-not $Setup) {
    Say 'no installer could be downloaded'
    Write-Host 'RESULT=NODOWNLOAD'
    exit 3
}

if ($DryRun) {
    Write-Host 'RESULT=DRYRUN'
    exit 0
}

# ---- 2) silent, per-user install (no admin rights required) ----------
$InstallArgs = @(
    '/quiet',
    'InstallAllUsers=0',
    'InstallLauncherAllUsers=0',
    'PrependPath=1',
    'Include_pip=1',
    'Include_launcher=1',
    'Include_lib=1',
    'Include_tcltk=1',
    'Include_test=0',
    'Include_doc=0',
    'Include_dev=0',
    'Include_symbols=0',
    'Include_debug=0',
    'AssociateFiles=0',
    'Shortcuts=0'
)

Say 'running the silent installer, this usually takes 1-3 minutes ...'
try {
    $proc = Start-Process -FilePath $Setup -ArgumentList $InstallArgs -Wait -PassThru
    Say ('installer exit code ' + $proc.ExitCode)
} catch {
    Say ('installer failed to start: ' + $_.Exception.Message)
    Write-Host 'RESULT=INSTALLFAIL'
    exit 3
}

Start-Sleep -Seconds 2

# ---- 3) where did it land? ------------------------------------------
#  PrependPath=1 only edits the stored PATH, so the already running
#  cmd.exe would not see python yet. We look on disk instead.
$Roots = @(
    (Join-Path $env:LOCALAPPDATA 'Programs\Python'),
    (Join-Path $env:ProgramFiles 'Python'),
    (Join-Path ${env:ProgramFiles(x86)} 'Python')
)

$Found = @()
foreach ($root in $Roots) {
    if ([string]::IsNullOrEmpty($root)) { continue }
    if (-not (Test-Path $root)) { continue }
    $Found += Get-ChildItem -Path $root -Filter 'python.exe' -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch 'WindowsApps' } |
        Select-Object -ExpandProperty FullName
}

if ($Found.Count -gt 0) {
    $python = $Found | Sort-Object -Descending | Select-Object -First 1
    Say ('python.exe found at ' + $python)
    Write-Host ('PYTHON=' + $python)
    Write-Host 'RESULT=OK'
    exit 0
}

Say 'installer finished, but python.exe was not found in the usual folders'
Write-Host 'RESULT=NOTFOUND'
exit 4
