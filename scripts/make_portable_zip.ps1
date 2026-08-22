# Bundle the parts you actually need on another PC into a single ZIP.
# Skips: .venv, node_modules, caches, datasets, lora_jobs (training history).
# Includes: source code, APP/loras, APP/voices, configs, docs.
#
# Usage:
#   .\scripts\make_portable_zip.ps1                       # default: Irodori-TTS_portable.zip in repo root
#   .\scripts\make_portable_zip.ps1 -OutPath D:\backup.zip
#   .\scripts\make_portable_zip.ps1 -IncludeDatasets      # also bundle APP/datasets/ (huge)

param(
  [string]$OutPath = "Irodori-TTS_portable.zip",
  [switch]$IncludeDatasets,
  [switch]$IncludeLoraJobs
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Always excluded
$excludeDirs = @(
  '.git', '.venv',
  'APP\node_modules', 'APP\__pycache__',
  'APP\outputs',
  'APP\datasets\_staging',
  'irodori_tts\__pycache__',
  'scripts\__pycache__',
  'node_modules'
)
# Optional excludes (added unless caller asked to include)
if (-not $IncludeDatasets)  { $excludeDirs += 'APP\datasets' }
if (-not $IncludeLoraJobs)  { $excludeDirs += 'APP\lora_jobs' }

# Stage to a temp dir, then zip. robocopy gives us reliable exclusion.
$stage = Join-Path $env:TEMP "irodori_portable_$([guid]::NewGuid().ToString('N').Substring(0,8))"
New-Item -ItemType Directory -Path $stage | Out-Null
try {
  Write-Host "Staging to: $stage" -ForegroundColor Cyan
  $xdArgs = @()
  foreach ($d in $excludeDirs) { $xdArgs += '/XD'; $xdArgs += (Join-Path $root $d) }
  $robocopyArgs = @($root, $stage, '/E', '/NFL', '/NDL', '/NJH', '/NJS', '/R:1', '/W:1') + $xdArgs + @('/XF', '*.pyc', '*.tmp', 'Thumbs.db')
  robocopy @robocopyArgs | Out-Null
  # robocopy exit codes 0-7 are success-ish; 8+ are real errors
  if ($LASTEXITCODE -ge 8) { throw "robocopy failed with exit $LASTEXITCODE" }

  $sizeMB = [math]::Round((Get-ChildItem $stage -Recurse | Measure-Object Length -Sum).Sum / 1MB, 1)
  Write-Host "Staged content: ${sizeMB}MB" -ForegroundColor Cyan

  # Resolve output path (relative -> under repo root)
  if (-not [System.IO.Path]::IsPathRooted($OutPath)) {
    $OutPath = Join-Path $root $OutPath
  }
  if (Test-Path $OutPath) { Remove-Item $OutPath -Force }

  Write-Host "Compressing to: $OutPath ..." -ForegroundColor Cyan
  Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $OutPath -CompressionLevel Optimal

  $zipMB = [math]::Round((Get-Item $OutPath).Length / 1MB, 1)
  Write-Host ""
  Write-Host "DONE: $OutPath  (${zipMB}MB)" -ForegroundColor Green
}
finally {
  if (Test-Path $stage) {
    Write-Host "Cleaning up stage..." -ForegroundColor DarkGray
    Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
  }
}
