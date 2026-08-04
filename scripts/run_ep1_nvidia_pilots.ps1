param(
    [switch]$SkipApiPrompt
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
        Model = "mistralai/mistral-medium-3.5-128b"
        Slug = "mistral_medium_3_5"
    },
    @{
        Model = "nvidia/nemotron-3-ultra-550b-a55b"
        Slug = "nemotron3_ultra"
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
        "--timeout", "1800",
        "--max-attempts", "3",
        "--retry-sleep-seconds", "5"
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
    if (-not (Test-Path -LiteralPath $modelFreeze)) {
        & py -3.12 scripts\freeze_ep1_nvidia_pilot.py `
            --design $design `
            --model $run.Model `
            --pilot-dir $pilotDir `
            --out $modelFreeze
        if ($LASTEXITCODE -ne 0) {
            throw "No se pudo congelar el techo confirmatorio de $($run.Model)."
        }
    } else {
        Write-Host "Reusing model freeze: $modelFreeze" -ForegroundColor DarkGray
    }
}

Write-Host "`nPilotos completos. No se inició ningún holdout." -ForegroundColor Green
Write-Host "Revisa y fija los archivos de $freezeDir antes de la corrida confirmatoria."
