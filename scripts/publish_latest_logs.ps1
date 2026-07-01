param(
    [string]$SourceDirectory,
    [switch]$Commit,
    [switch]$Push
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$outputRoot = Join-Path $repoRoot "videos\output"
$destination = Join-Path $repoRoot "review\latest"

if (-not $SourceDirectory) {
    $summary = Get-ChildItem -LiteralPath $outputRoot -Recurse -File -Filter "*_summary.json" |
        Where-Object { $_.FullName -notmatch '[\\/]old\.[^\\/]+[\\/]' } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $summary) {
        throw "No current summary JSON was found below $outputRoot."
    }
    $source = $summary.Directory
} else {
    $sourcePath = if ([IO.Path]::IsPathRooted($SourceDirectory)) {
        $SourceDirectory
    } else {
        Join-Path $repoRoot $SourceDirectory
    }
    $source = Get-Item -LiteralPath $sourcePath
    $summary = Get-ChildItem -LiteralPath $source.FullName -File -Filter "*_summary.json" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $summary) {
        throw "No summary JSON was found in $($source.FullName)."
    }
}

$stem = $summary.BaseName -replace '_summary$', ''
$files = [ordered]@{
    "summary.json" = $summary.FullName
    "tracks.csv" = Join-Path $source.FullName "${stem}_tracks.csv"
    "process.log" = Join-Path $source.FullName "process.log"
}

foreach ($entry in $files.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath $entry.Value)) {
        throw "Missing required review artifact: $($entry.Value)"
    }
}

New-Item -ItemType Directory -Force -Path $destination | Out-Null
foreach ($entry in $files.GetEnumerator()) {
    Copy-Item -LiteralPath $entry.Value -Destination (Join-Path $destination $entry.Key) -Force
}

$manifestFiles = foreach ($entry in $files.GetEnumerator()) {
    $copied = Get-Item -LiteralPath (Join-Path $destination $entry.Key)
    [ordered]@{
        name = $entry.Key
        bytes = $copied.Length
        sha256 = (Get-FileHash -LiteralPath $copied.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$manifest = [ordered]@{
    source_directory = $source.FullName.Substring($repoRoot.Length + 1).Replace("\", "/")
    source_stem = $stem
    published_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    files = @($manifestFiles)
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $destination "manifest.json") -Encoding utf8

$relativeDestination = "review/latest"
git -C $repoRoot add -- $relativeDestination
if ($LASTEXITCODE -ne 0) {
    throw "git add failed for $relativeDestination"
}

if ($Push) {
    $Commit = $true
}
if ($Commit) {
    git -C $repoRoot diff --cached --quiet -- $relativeDestination
    if ($LASTEXITCODE -ne 0) {
        git -C $repoRoot commit -m "Update latest pipeline review logs" -- $relativeDestination
        if ($LASTEXITCODE -ne 0) {
            throw "git commit failed for $relativeDestination"
        }
    } else {
        Write-Host "Latest review logs are already committed."
    }
}
if ($Push) {
    $branch = git -C $repoRoot branch --show-current
    git -C $repoRoot push origin $branch
    if ($LASTEXITCODE -ne 0) {
        throw "git push failed for $branch"
    }
}

Write-Host "Published latest review logs from $($source.FullName) to $destination"
