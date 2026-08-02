param([switch]$SkipApiPrompt)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Protocol = "C:\Users\THINKPAD\Desktop\Aptadynamik Cybernetics\Aptadynamik Logoprobs PRAMA Protokol\protocols\minimal_structural_perturbations.yaml"
Set-Location -LiteralPath $Repo

if (-not $env:NVIDIA_API_KEY -and -not $SkipApiPrompt) {
    $env:NVIDIA_API_KEY = (Read-Host "Pega tu NVIDIA API key" -MaskInput).Trim()
}
if ([string]::IsNullOrWhiteSpace($env:NVIDIA_API_KEY)) {
    throw "NVIDIA_API_KEY no está activa en esta sesión de PowerShell."
}
if (-not $env:NVIDIA_API_KEY.StartsWith("nvapi-")) {
    throw "NVIDIA_API_KEY no tiene el prefijo nvapi-."
}
$env:OPENAI_BASE_URL = "https://integrate.api.nvidia.com/v1"

$Common = @(
    "--protocol", $Protocol,
    "--max-tokens", "700",
    "--top-logprobs", "5",
    "--seed", "1337",
    "--timeout", "600",
    "--max-attempts", "3",
    "--retry-sleep", "5",
    "--fixed-ready-horizon", "16"
)

$Runs = @(
    @{
        Model = "nvidia/nemotron-3-super-120b-a12b"
        Output = "run_outputs\minimal_structural_nemotron_super_dynamic_v4"
        Profile = @("--temperature", "1.0", "--top-p", "0.95")
        Reuse = "run_outputs\minimal_structural_perturbations_nemotron_super_v1"
    },
    @{
        Model = "mistralai/mistral-medium-3.5-128b"
        Output = "run_outputs\minimal_structural_mistral_medium_3_5_dynamic_v4"
        Profile = @("--temperature", "0.7", "--top-p", "1.0", "--reasoning-effort", "none")
        Reuse = $null
    },
    @{
        Model = "nvidia/nemotron-3-ultra-550b-a55b"
        Output = "run_outputs\minimal_structural_nemotron_ultra_dynamic_v4"
        Profile = @("--temperature", "1.0", "--top-p", "0.95")
        Reuse = $null
    }
)

foreach ($Run in $Runs) {
    Write-Host "`n=== $($Run.Model) ===" -ForegroundColor Cyan
    $Arguments = @(
        "-3.12", "scripts\run_minimal_structural_perturbations_dynamic.py",
        "--model", $Run.Model,
        "--output-dir", $Run.Output
    ) + $Common + $Run.Profile
    if ($Run.Reuse) {
        $Arguments += @("--reuse-responses-from", $Run.Reuse)
    }
    & py @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Falló $($Run.Model). Ejecuta nuevamente este script para reanudar."
    }
}

$SummaryPath = "run_outputs\minimal_structural_three_model_v4_summary.json"
& py -3.12 scripts\summarize_minimal_structural_three_models_v4.py --out $SummaryPath
if ($LASTEXITCODE -ne 0) {
    throw "Falló la síntesis final de los tres modelos."
}
Write-Host "`nFinalizado: $SummaryPath" -ForegroundColor Green
