param(
    [switch]$SkipApiPrompt,
    [int]$TimeoutSeconds = 900,
    [int]$MaxAttempts = 6,
    [double]$RetrySleepSeconds = 30
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo

if (-not $env:NVIDIA_API_KEY -and -not $SkipApiPrompt) {
    $env:NVIDIA_API_KEY = (Read-Host "Pega tu NVIDIA API key" -MaskInput).Trim()
}
if ([string]::IsNullOrWhiteSpace($env:NVIDIA_API_KEY)) {
    throw "NVIDIA_API_KEY no está activa en esta sesión de PowerShell."
}
if (-not $env:NVIDIA_API_KEY.StartsWith("nvapi-")) {
    throw "NVIDIA_API_KEY no tiene el prefijo nvapi-."
}

$design = "config\ep1_nvidia_replication_v1.json"
$freezeRecord = Get-Content -Raw -Encoding utf8 "config\ep1_nvidia_replication_v1.freeze.json" | ConvertFrom-Json
$designHash = (Get-FileHash $design -Algorithm SHA256).Hash.ToLower()
if ($designHash -ne $freezeRecord.design_sha256) {
    throw "El diseño E-P1 NVIDIA no coincide con su freeze previo al piloto."
}

$runs = @(
    @{
        Model = "nvidia/nemotron-3-super-120b-a12b"
        Slug = "nemotron3_super"
    },
    @{
        Model = "nvidia/nemotron-3-ultra-550b-a55b"
        Slug = "nemotron3_ultra"
    },
    @{
        Model = "mistralai/mistral-medium-3.5-128b"
        Slug = "mistral_medium_3_5"
    }
)

$root = "run_outputs\ep1_nvidia_r1"
$freezeDir = Join-Path $root "freezes"
New-Item -ItemType Directory -Force -Path $freezeDir | Out-Null

foreach ($run in $runs) {
    $pilotDir = Join-Path $root ("pilot_" + $run.Slug)
    $arguments = @(
        "-3.12", "scripts\collect_ep1_nvidia.py",
        "--design", $design,
        "--mode", "pilot",
        "--model", $run.Model,
        "--out", $pilotDir,
        "--timeout", $TimeoutSeconds.ToString(),
        "--max-attempts", $MaxAttempts.ToString(),
        "--retry-sleep-seconds", $RetrySleepSeconds.ToString([Globalization.CultureInfo]::InvariantCulture)
    )
    if (Test-Path -LiteralPath $pilotDir) {
        $arguments += "--resume"
    }
    Write-Host "`n=== E-P1 pilot: $($run.Model) ===" -ForegroundColor Cyan
    & py @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Falló el piloto de $($run.Model). Ejecuta nuevamente este script para reanudar."
    }

    $modelFreeze = Join-Path $freezeDir ($run.Slug + ".json")
    $replaceFreeze = $false
    if (Test-Path -LiteralPath $modelFreeze) {
        $manifest = Get-Content -Raw -Encoding utf8 (Join-Path $pilotDir "manifest.json") | ConvertFrom-Json
        $existingFreeze = Get-Content -Raw -Encoding utf8 $modelFreeze | ConvertFrom-Json
        $replaceFreeze = (
            -not $existingFreeze.pilot_collection_content_sha256 -or
            $existingFreeze.pilot_collection_content_sha256 -ne $manifest.collection_content_sha256
        )
    }
    if (-not (Test-Path -LiteralPath $modelFreeze) -or $replaceFreeze) {
        $freezeArguments = @(
            "-3.12", "scripts\freeze_ep1_nvidia_pilot.py",
            "--design", $design,
            "--model", $run.Model,
            "--pilot-dir", $pilotDir,
            "--out", $modelFreeze
        )
        if ($replaceFreeze) {
            $freezeArguments += "--replace"
        }
        & py @freezeArguments
        if ($LASTEXITCODE -ne 0) {
            throw "No se pudo congelar el techo confirmatorio de $($run.Model)."
        }
    } else {
        Write-Host "Reusing model freeze: $modelFreeze" -ForegroundColor DarkGray
    }
}

Write-Host "`nPilotos completos. No se inició ningún holdout." -ForegroundColor Green
Write-Host "Revisa y fija los archivos de $freezeDir antes de la corrida confirmatoria."
