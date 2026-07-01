param(
    [switch]$Deploy,
    [ValidateSet("bytetrack", "botsort", "ocsort", "deepocsort", "hybridsort", "strongsort", "boosttrack")]
    [string]$TrackerName,
    [string]$ExperimentName,
    [string[]]$GpuTypeIds
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

$tracker = if ($TrackerName) {
    $TrackerName
} elseif ($env:RUNPOD_TRACKER) {
    $env:RUNPOD_TRACKER.ToLowerInvariant()
} else {
    "bytetrack"
}

if ($tracker -notin @("bytetrack", "botsort", "ocsort", "deepocsort", "hybridsort", "strongsort", "boosttrack")) {
    throw "Set RUNPOD_TRACKER to bytetrack, botsort, or a supported BoxMOT tracker."
}

$storageType = if ($env:RUNPOD_NETWORK_VOLUME_ID) {
    "persistent"
} elseif ($env:RUNPOD_ALLOW_EPHEMERAL_VOLUME -eq "true") {
    "disposable"
} else {
    throw @"
Set RUNPOD_NETWORK_VOLUME_ID in $repoRoot\.env before deploying.
This automatically reconnects the existing network volume at /workspace.
For a disposable Pod volume instead, explicitly set RUNPOD_ALLOW_EPHEMERAL_VOLUME=true.
"@
}

$podName = "$($env:RUNPOD_POD_NAME)-$storageType-$tracker"
if ($ExperimentName) {
    $podName = "$podName-$ExperimentName"
}

$publicKeyPath = if ($env:RUNPOD_PUBLIC_KEY_FILE) {
    $env:RUNPOD_PUBLIC_KEY_FILE
} else {
    Join-Path $env:USERPROFILE ".ssh\ai_karate_runpod_ed25519.pub"
}

$gitRef = if ($env:RUNPOD_GIT_REF) {
    $env:RUNPOD_GIT_REF
} else {
    git -C $repoRoot rev-parse --abbrev-ref HEAD
}

$bootstrapCommand = @"
set -e
/start.sh &
start_pid=`$!
if [ ! -d /workspace/ai-karate/.git ]; then
  git clone --branch "$gitRef" https://github.com/glass105/ai-karate.git /workspace/ai-karate
else
  git -C /workspace/ai-karate fetch origin "$gitRef"
  git -C /workspace/ai-karate checkout "$gitRef"
  git -C /workspace/ai-karate pull --ff-only origin "$gitRef"
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
"`$venv/bin/python" -m pip install --no-cache-dir -r requirements-boxmot.txt "lap>=0.5.12,<1"
"`$venv/bin/python" -m pip uninstall -y onnxruntime
"`$venv/bin/python" -m pip install --no-cache-dir --force-reinstall onnxruntime-gpu==1.23.2
"`$venv/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available to PyTorch'; print(f'CUDA ready: {torch.cuda.get_device_name(0)}')"
"`$venv/bin/python" -c "import onnxruntime as ort; assert 'CUDAExecutionProvider' in ort.get_available_providers(), ort.get_available_providers(); print(f'ONNXRuntime providers: {ort.get_available_providers()}')"
wait "`$start_pid"
"@

$body = @{
    name              = $podName
    imageName         = $env:RUNPOD_IMAGE
    cloudType         = "SECURE"
    computeType       = "GPU"
    gpuTypeIds        = @($(if ($GpuTypeIds) { $GpuTypeIds } else { $env:RUNPOD_GPU_ID }))
    gpuTypePriority   = "custom"
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

if (-not $body.env) {
    $body.env = @{}
}
$body.env.AI_KARATE_TRACKER = "$tracker.yaml"

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

$displayResponse = $response | ConvertTo-Json -Depth 8 | ConvertFrom-Json
if ($displayResponse.env.PUBLIC_KEY) {
    $displayResponse.env.PUBLIC_KEY = "<redacted>"
}
if ($displayResponse.env.JUPYTER_PASSWORD) {
    $displayResponse.env.JUPYTER_PASSWORD = "<redacted>"
}
$displayResponse | ConvertTo-Json -Depth 8
