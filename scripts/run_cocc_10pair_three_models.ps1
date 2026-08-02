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
$env:OPENAI_BASE_URL = "https://integrate.api.nvidia.com/v1"

$designPath = "config\cocc_10pair_three_model_prospective_v1.json"
$design = Get-Content -Raw -Encoding utf8 $designPath | ConvertFrom-Json
$designHash = (Get-FileHash $designPath -Algorithm SHA256).Hash.ToLower()
$freeze = Get-Content -Raw -Encoding utf8 "config\cocc_10pair_three_model_prospective_v1.freeze.json" | ConvertFrom-Json
if ($designHash -ne $freeze.design_sha256) {
    throw "El SHA del diseño no coincide con su freeze record."
}

function Assert-Hash([string]$Path, [string]$Expected) {
    $actual = (Get-FileHash $Path -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $Expected) {
        throw "Hash distinto para ${Path}: esperado=$Expected observado=$actual"
    }
}

Assert-Hash $design.dataset $design.dataset_sha256
Assert-Hash $design.dataset_manifest $design.dataset_manifest_sha256
Assert-Hash $design.observer_contract $design.observer_contract_sha256
Assert-Hash $design.kernel.declaration $design.kernel.declaration_sha256
Assert-Hash $design.kernel.recertification $design.kernel.recertification_sha256
Assert-Hash "scripts\run_break_the_chain_prama_eval_nvidia.py" $design.interface_hashes.nemotron_runner_sha256
Assert-Hash "scripts\run_break_the_chain_prama_eval_nvidia_mistral.py" $design.interface_hashes.mistral_runner_sha256
Assert-Hash "scripts\project_cocc_prama_dynamic.py" $design.interface_hashes.projector_sha256
Assert-Hash "scripts\cocc_external_verifier.py" $design.interface_hashes.verifier_sha256
Assert-Hash "scripts\evaluate_cocc_10pair_prospective.py" $design.interface_hashes.evaluator_sha256
Assert-Hash "src\aptadynamic_llm\model_payload.py" $design.interface_hashes.model_payload_sha256

$verifier = @(
    "py", "-3.12", "scripts\cocc_external_verifier.py",
    "--dataset", $design.dataset,
    "--timeout", "20"
) | ConvertTo-Json -Compress
$projector = @(
    "py", "-3.12", "scripts\project_cocc_prama_dynamic.py",
    "--observer-contract", $design.observer_contract,
    "--declaration", $design.kernel.declaration,
    "--recertification", $design.kernel.recertification
) | ConvertTo-Json -Compress

function Invoke-ProspectiveRun(
    [string]$Runner,
    [string]$Model,
    [string]$OutputDir,
    [string[]]$ProfileArgs
) {
    $arguments = @(
        "-3.12", $Runner,
        "--dataset", $design.dataset,
        "--dataset-manifest", $design.dataset_manifest,
        "--output-dir", $OutputDir,
        "--model", $Model,
        "--n", "20",
        "--prospective-only",
        "--max-tokens", "16384",
        "--top-logprobs", "5",
        "--seed", "1337",
        "--max-attempts", "3",
        "--retry-sleep-seconds", "5",
        "--timeout", "1800",
        "--confirmatory-design-sha256", $designHash,
        "--verifier-command-json", $verifier,
        "--projector-command-json", $projector,
        "--projector-observer-sha256", $design.observer_contract_sha256
    ) + $ProfileArgs
    if (Test-Path -LiteralPath $OutputDir) {
        $arguments += "--resume"
    }
    Write-Host "`n=== $Model ===" -ForegroundColor Cyan
    & py @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Falló $Model. Ejecuta de nuevo este mismo script para reanudar."
    }
    $report = Join-Path $OutputDir "evaluation\report.json"
    & py -3.12 scripts\evaluate_cocc_10pair_prospective.py `
        --run-dir $OutputDir `
        --design $designPath `
        --out $report
    if ($LASTEXITCODE -ne 0) {
        throw "Falló la evaluación local de $Model."
    }
}

$runs = @(
    @{
        Runner = "scripts\run_break_the_chain_prama_eval_nvidia.py"
        Model = "nvidia/nemotron-3-super-120b-a12b"
        Output = "run_outputs\cocc_10pair_nemotron3_super_v1"
        Profile = @("--temperature", "1.0", "--top-p", "0.95", "--no-enable-thinking")
    },
    @{
        Runner = "scripts\run_break_the_chain_prama_eval_nvidia_mistral.py"
        Model = "mistralai/mistral-medium-3.5-128b"
        Output = "run_outputs\cocc_10pair_mistral_medium_3_5_v1"
        Profile = @("--temperature", "0.7", "--top-p", "1.0", "--reasoning-effort", "none")
    },
    @{
        Runner = "scripts\run_break_the_chain_prama_eval_nvidia.py"
        Model = "nvidia/nemotron-3-ultra-550b-a55b"
        Output = "run_outputs\cocc_10pair_nemotron3_ultra_v1"
        Profile = @("--temperature", "1.0", "--top-p", "0.95", "--no-enable-thinking")
    }
)

foreach ($run in $runs) {
    Invoke-ProspectiveRun $run.Runner $run.Model $run.Output $run.Profile
}

$summary = [ordered]@{
    schema = "LLM-SVM-CoCC-10pair-three-model-summary/1"
    design_sha256 = $designHash
    generated_at = [DateTimeOffset]::UtcNow.ToString("o")
    models = @()
}
foreach ($run in $runs) {
    $reportPath = Join-Path $run.Output "evaluation\report.json"
    $report = Get-Content -Raw -Encoding utf8 $reportPath | ConvertFrom-Json
    $summary.models += [ordered]@{
        model = $report.model
        verdict = $report.verdict
        pass_n = $report.outcomes.pass_n
        fail_n = $report.outcomes.fail_n
        outcome_discordant_pairs = $report.paired_exact_inference.outcome_discordant_pair_count
        primary_exact_p = $report.paired_exact_inference.primary.exact_one_sided_sign_p
        auroc_delta = $report.full_trajectory_metrics.max_delta
        auroc_xi = $report.full_trajectory_metrics.max_xi
        auroc_balance = $report.full_trajectory_metrics.max_negative_balance
        prama_minus_delta = $report.full_trajectory_metrics.prama_minus_delta
    }
}
$summaryPath = "run_outputs\cocc_10pair_three_model_summary_v1.json"
$summary | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 $summaryPath
Write-Host "`nFinalizado: $summaryPath" -ForegroundColor Green
