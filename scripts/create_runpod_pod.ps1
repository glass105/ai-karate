param(
    [switch]$Deploy
)

$ErrorActionPreference = "Stop"

function Import-DotEnv {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing local environment file: $Path"
    }

    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^\s*(?:#|$)') {
            continue
        }

        $name, $value = $line -split '=', 2
        if ($name -and -not [Environment]::GetEnvironmentVariable($name)) {
            [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), "Process")
        }
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Import-DotEnv (Join-Path $repoRoot ".env")

if (-not $env:RUNPOD_API_KEY) {
    throw "Set RUNPOD_API_KEY in $repoRoot\.env before deploying."
}

$publicKeyPath = if ($env:RUNPOD_PUBLIC_KEY_FILE) {
    $env:RUNPOD_PUBLIC_KEY_FILE
} else {
    Join-Path $env:USERPROFILE ".ssh\ai_karate_runpod_ed25519.pub"
}

$bootstrapCommand = @"
set -e
/start.sh &
start_pid=`$!
if [ ! -d /workspace/ai-karate/.git ]; then
  git clone https://github.com/glass105/ai-karate.git /workspace/ai-karate
else
  git -C /workspace/ai-karate pull --ff-only
fi
cd /workspace/ai-karate
mkdir -p models runs videos/input videos/output
venv=/root/.venvs/ai-karate
mkdir -p /root/.venvs
if [ ! -x "`$venv/bin/python" ]; then
  python -m venv --system-site-packages "`$venv"
fi
"`$venv/bin/python" -m pip install --upgrade pip
"`$venv/bin/python" -m pip install --upgrade torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
"`$venv/bin/python" -m pip install --no-cache-dir -r requirements.txt "lap>=0.5.12,<1"
"`$venv/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available to PyTorch'; print(f'CUDA ready: {torch.cuda.get_device_name(0)}')"
wait "`$start_pid"
"@

$body = @{
    name              = $env:RUNPOD_POD_NAME
    imageName         = $env:RUNPOD_IMAGE
    cloudType         = "SECURE"
    computeType       = "GPU"
    gpuTypeIds        = @($env:RUNPOD_GPU_ID)
    gpuCount          = 1
    containerDiskInGb = 40
    volumeInGb        = 50
    volumeMountPath   = "/workspace"
    ports             = @("8888/http", "22/tcp")
    dockerStartCmd    = @("bash", "-lc", $bootstrapCommand)
}

if (Test-Path -LiteralPath $publicKeyPath) {
    $body.env = @{
        PUBLIC_KEY = (Get-Content -Raw -LiteralPath $publicKeyPath).Trim()
    }
}

if ($env:RUNPOD_JUPYTER_PASSWORD) {
    if (-not $body.env) {
        $body.env = @{}
    }
    $body.env.JUPYTER_PASSWORD = $env:RUNPOD_JUPYTER_PASSWORD
}

if ($env:RUNPOD_NETWORK_VOLUME_ID) {
    $body.Remove("volumeInGb")
    $body.networkVolumeId = $env:RUNPOD_NETWORK_VOLUME_ID
}

if (-not $Deploy) {
    Write-Host "Dry run only. Add -Deploy to create a billable Runpod Pod."
    $displayBody = $body | ConvertTo-Json -Depth 6 | ConvertFrom-Json
    if ($displayBody.env.PUBLIC_KEY) {
        $displayBody.env.PUBLIC_KEY = "<redacted>"
    }
    if ($displayBody.env.JUPYTER_PASSWORD) {
        $displayBody.env.JUPYTER_PASSWORD = "<redacted>"
    }
    $displayBody | ConvertTo-Json -Depth 6
    exit 0
}

$headers = @{
    Authorization = "Bearer $env:RUNPOD_API_KEY"
}

$response = Invoke-RestMethod `
    -Method Post `
    -Uri "https://rest.runpod.io/v1/pods" `
    -Headers $headers `
    -ContentType "application/json" `
    -Body ($body | ConvertTo-Json -Depth 4)

$response | ConvertTo-Json -Depth 8
