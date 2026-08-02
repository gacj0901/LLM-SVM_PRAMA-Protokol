param(
    [switch]$DryRun,
    [switch]$SkipApiPrompt
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Protocol = "C:\Users\THINKPAD\Desktop\Aptadynamik Cybernetics\Aptadynamik Logoprobs PRAMA Protokol\protocols\minimal_structural_perturbations.yaml"

Set-Location $Repo

if (-not $DryRun -and -not $env:NVIDIA_API_KEY -and -not $SkipApiPrompt) {
    $env:NVIDIA_API_KEY = (Read-Host "Pega tu NVIDIA API key" -MaskInput).Trim()
}
if (-not $DryRun -and [string]::IsNullOrWhiteSpace($env:NVIDIA_API_KEY)) {
    throw "NVIDIA_API_KEY no está activa en esta sesión de PowerShell."
}

$Arguments = @(
    "-3.12",
    "scripts\run_minimal_structural_perturbations_nvidia.py",
    "--protocol", $Protocol
)
if ($DryRun) {
    $Arguments += "--dry-run"
}

& py @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Falló la corrida. Ejecuta este mismo script para reanudar desde la última respuesta guardada."
}
