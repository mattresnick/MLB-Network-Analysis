param(
  [string]$PythonExe = "python",
  [string]$VenvDir = ".venv",
  [string]$ReqFile = "requirements.txt",
  [switch]$Recreate,
  [switch]$UseConda
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $RepoRoot
$venvPath = Join-Path $RepoRoot $VenvDir

# Helper: detect whether a target folder is a Conda environment
function Test-CondaEnv {
  param([string]$Path)
  return Test-Path (Join-Path $Path 'conda-meta')
}

# Force recreate if requested
if ($Recreate -and (Test-Path $venvPath)) {
  Write-Host "Recreating environment: removing $venvPath"
  try { Remove-Item -Recurse -Force $venvPath } catch {
    Write-Warning "Standard remove failed, attempting to take ownership and retry"
    try { & takeown /F $venvPath /R /D Y | Out-Null; & icacls $venvPath /grant $env:USERNAME`:F /T | Out-Null; Remove-Item -Recurse -Force $venvPath } catch {}
  }
}

# Try to find a python.exe if 'python' alias is disabled
function Resolve-Python {
  param([string]$Preferred)
  try {
    $v = & $Preferred -V 2>$null
    if ($LASTEXITCODE -eq 0) { return $Preferred }
  } catch {}
  $candidates = @(
    "$env:LOCALAPPDATA\\Programs\\Python\\Python312\\python.exe",
    "$env:LOCALAPPDATA\\Programs\\Python\\Python311\\python.exe",
    "$env:LOCALAPPDATA\\Programs\\Python\\Python310\\python.exe",
    "$env:ProgramFiles\\Python312\\python.exe",
    "$env:ProgramFiles\\Python311\\python.exe",
    "$env:ProgramFiles\\Python310\\python.exe",
    "C:\\Python312\\python.exe",
    "C:\\Python311\\python.exe",
    "C:\\Python310\\python.exe",
    "$env:USERPROFILE\\anaconda3\\python.exe",
    "$env:USERPROFILE\\miniconda3\\python.exe",
    "C:\\ProgramData\\Anaconda3\\python.exe",
    "C:\\Miniconda3\\python.exe"
  )
  # Registry-based discovery
  function Get-PythonFromRegistry {
    $roots = @(
      'HKCU:Software\Python\PythonCore',
      'HKLM:Software\Python\PythonCore',
      'HKLM:Software\WOW6432Node\Python\PythonCore'
    )
    $candidates = @()
    foreach ($root in $roots) {
      try {
        if (Test-Path $root) {
          Get-ChildItem $root | ForEach-Object {
            $verKey = $_.PsPath
            $ip = Join-Path $verKey 'InstallPath'
            try {
              $props = Get-ItemProperty -Path $ip -ErrorAction Stop
              if ($props.ExecutablePath -and (Test-Path $props.ExecutablePath)) { $candidates += $props.ExecutablePath }
              if ($props.'(default)') {
                $base = $props.'(default)'
                $exe = Join-Path $base 'python.exe'
                if (Test-Path $exe) { $candidates += $exe }
              }
            } catch {}
          }
        }
      } catch {}
    }
    return $candidates | Select-Object -Unique
  }
  $regFound = Get-PythonFromRegistry
  foreach ($p in $regFound) {
    try { & $p -V 2>$null | Out-Null; if ($LASTEXITCODE -eq 0) { return $p } } catch {}
  }
  foreach ($p in $candidates) {
    if (Test-Path $p) { return $p }
  }
  return $Preferred
}

$Python = Resolve-Python $PythonExe
Write-Host "Using Python at: $Python"

# Helper to attempt creating a venv using a given command tuple
function New-VenvTry {
  param([string[]]$Cmd)
  try {
    & $Cmd[0] @($Cmd[1..($Cmd.Count-1)])
    return $true
  } catch {
    return $false
  }
}

function New-Venv {
  param([string]$TargetDir, [string]$PrimaryPython)
  $created = $false
  # Prefer the Windows Python launcher first to avoid conda base python
  foreach ($ver in @('3.11','3.12','3.10')) {
    if (New-VenvTry @('py', "-$ver", '-m','venv', $TargetDir)) { $created = $true; break }
  }
  if (-not $created) {
    # Default launcher without pin
    $created = New-VenvTry @('py','-m','venv', $TargetDir)
  }
  if (-not $created) {
    # Try python3
    $created = New-VenvTry @('python3','-m','venv', $TargetDir)
  }
  if (-not $created) {
    # Last resort: whatever $PrimaryPython resolves to
    $created = New-VenvTry @($PrimaryPython, '-m','venv', $TargetDir)
  }
  return $created
}

function Find-Conda {
  param([string]$BasePython)
  $root = Split-Path -Parent $BasePython
  $candidates = @(
    (Join-Path $root 'condabin/conda.bat'),
    (Join-Path $root 'Scripts/conda.exe'),
    'conda'
  )
  foreach ($c in $candidates) {
    try {
      $p = (Resolve-Path $c -ErrorAction Stop).Path
      if (Test-Path $p) { return $p }
    } catch {}
  }
  return $null
}

# Only use conda if explicitly requested, otherwise default to venv
$condaExe = $null
if ($UseConda) {
  $isConda = ($Python -match '(?i)anaconda|miniconda')
  $condaExe = if ($isConda) { Find-Conda -BasePython $Python } else { $null }
  if ($condaExe) { Write-Host "Detected Conda at: $condaExe" } else { Write-Warning "UseConda was specified but conda was not found; falling back to venv." }
}

if (-not (Test-Path $VenvDir)) {
  Write-Host "Creating virtual environment in $VenvDir"
  $created = $false
  if ($condaExe) {
    Write-Host "Creating conda env at $VenvDir (python=3.11)"
    & $condaExe create -y -p $VenvDir python=3.11
    if ($LASTEXITCODE -eq 0) { $created = $true } else { Write-Warning "Conda env creation failed, falling back to venv" }
  }
  if (-not $created) {
    if (-not (New-Venv -TargetDir $VenvDir -PrimaryPython $Python)) {
      Write-Error "Failed to create virtual environment. Please ensure Python 3.10+ is installed and accessible, or pass -PythonExe <path> to this script."
      exit 1
    }
  }
}


$VenvRootPy = Join-Path $VenvDir 'python.exe'
$VenvScriptsPy = Join-Path $VenvDir 'Scripts/python.exe'
$isCondaEnv = Test-CondaEnv -Path $VenvDir

# If env exists but is missing python, recreate it now
if ((-not (Test-Path $VenvRootPy)) -and (-not (Test-Path $VenvScriptsPy))) {
  Write-Warning "No python executable found in $VenvDir; recreating environment..."
  try { Remove-Item -Recurse -Force $VenvDir } catch {}
  $created = $false
  if ($condaExe) { & $condaExe create -y -p $VenvDir python=3.11; if ($LASTEXITCODE -eq 0) { $created = $true } }
  if (-not $created) { if (-not (New-Venv -TargetDir $VenvDir -PrimaryPython $Python)) { Write-Error "Failed to (re)create environment."; exit 1 } }
  $VenvRootPy = Join-Path $VenvDir 'python.exe'
  $VenvScriptsPy = Join-Path $VenvDir 'Scripts/python.exe'
}

# If this is a conda env, do NOT copy python.exe into Scripts (breaks relative paths)
if (-not $isCondaEnv) {
  if ((-not (Test-Path $VenvScriptsPy)) -and (Test-Path $VenvRootPy)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $VenvScriptsPy) | Out-Null
    Copy-Item -Force $VenvRootPy $VenvScriptsPy
  }
}

$VenvPy = if ($isCondaEnv -and (Test-Path $VenvRootPy)) {
  # Conda env: always use root python.exe
  $VenvRootPy
} elseif (Test-Path $VenvScriptsPy) {
  $VenvScriptsPy
} elseif (Test-Path $VenvRootPy) {
  $VenvRootPy
} else {
  $null
}
if (-not $VenvPy) {
  Write-Warning "Virtual environment appears incomplete: missing python.exe"
  Write-Host "Recreating virtual environment..."
  try { Remove-Item -Recurse -Force $VenvDir } catch {}
  if ($condaExe) {
    & $condaExe create -y -p $VenvDir python=3.11
  } else {
    if (-not (New-Venv -TargetDir $VenvDir -PrimaryPython $Python)) {
      Write-Error "Failed to (re)create virtual environment. Please ensure Python is installed or pass -PythonExe <path>."
      exit 1
    }
  }
  $VenvRootPy = Join-Path $VenvDir 'python.exe'
  $VenvScriptsPy = Join-Path $VenvDir 'Scripts/python.exe'
  if (-not $isCondaEnv) {
    if ((-not (Test-Path $VenvScriptsPy)) -and (Test-Path $VenvRootPy)) { New-Item -ItemType Directory -Force -Path (Split-Path $VenvScriptsPy) | Out-Null; Copy-Item -Force $VenvRootPy $VenvScriptsPy }
  }
  $VenvPy = if ($isCondaEnv -and (Test-Path $VenvRootPy)) { $VenvRootPy } elseif (Test-Path $VenvScriptsPy) { $VenvScriptsPy } elseif (Test-Path $VenvRootPy) { $VenvRootPy } else { $null }
  if (-not $VenvPy) {
    Write-Error "Virtual environment Python still not found after recreation at $VenvDir"
    exit 1
  }
}

# Verify SSL module
# Optional: SSL probe only if using conda
if ($condaExe -and (Test-CondaEnv -Path $VenvDir)) {
  & $condaExe run -p $VenvDir python -c "import ssl, sys; print('OK SSL:', ssl.OPENSSL_VERSION)" | Out-Null
  if ($LASTEXITCODE -ne 0) {
    Write-Warning "SSL probe failed inside conda env; attempting to update openssl/certifi"
    & $condaExe install -y -p $VenvDir -c conda-forge openssl certifi ca-certificates
  }
}
if (-not $VenvPy) {
  Write-Error "Unable to provision a functional Python environment in $VenvDir"
  exit 1
}

# Install dependencies using the environment's own Python (more reliable than conda run)
Write-Host "Installing dependencies with env Python: $VenvPy"
if ($isCondaEnv -and $condaExe) {
  # Ensure pip is present in conda env
  try {
    & $VenvPy -m pip --version | Out-Null
  } catch {
    Write-Host "pip not found in conda env; installing via conda..."
    & $condaExe install -y -p $VenvDir pip
  }
}

& $VenvPy -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Write-Error "pip upgrade failed in $VenvDir"; exit 1 }

& $VenvPy -m pip install -r $ReqFile
if ($LASTEXITCODE -ne 0) { Write-Error "pip install -r $ReqFile failed in $VenvDir"; exit 1 }

# Ensure debugpy is present for VS Code debugging
& $VenvPy -m pip install debugpy
if ($LASTEXITCODE -ne 0) { Write-Error "pip install debugpy failed in $VenvDir"; exit 1 }

Write-Host "Virtual environment is ready: $VenvDir"
