param(
    [ValidateSet("nemotron3_super", "nemotron3_ultra", "mistral_medium_3_5")]
    [string[]]$Models = @("nemotron3_super", "nemotron3_ultra"),
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

$designPath = "config\ep1_nvidia_replication_v1.json"
$design = Get-Content -Raw -Encoding utf8 $designPath | ConvertFrom-Json
$designRecord = Get-Content -Raw -Encoding utf8 "config\ep1_nvidia_replication_v1.freeze.json" | ConvertFrom-Json
$designHash = (Get-FileHash $designPath -Algorithm SHA256).Hash.ToLower()
if ($designHash -ne $designRecord.design_sha256) {
    throw "El diseño E-P1 NVIDIA no coincide con su freeze previo al piloto."
}

$profiles = @{
    nemotron3_super = @{
        Model = "nvidia/nemotron-3-super-120b-a12b"
    }
    nemotron3_ultra = @{
        Model = "nvidia/nemotron-3-ultra-550b-a55b"
    }
    mistral_medium_3_5 = @{
        Model = "mistralai/mistral-medium-3.5-128b"
    }
}

$root = "run_outputs\ep1_nvidia_r1"
foreach ($slug in $Models) {
    $profile = $profiles[$slug]
    $modelFreeze = Join-Path $root ("freezes\" + $slug + ".json")
    if (-not (Test-Path -LiteralPath $modelFreeze)) {
        throw "No existe el freeze confirmatorio de ${slug}: $modelFreeze"
    }
    & git ls-files --error-unmatch -- $modelFreeze *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "El freeze de ${slug} debe estar committed antes del holdout: $modelFreeze"
    }
    & git diff --quiet -- $modelFreeze
    if ($LASTEXITCODE -ne 0) {
        throw "El freeze de ${slug} tiene cambios sin fijar: $modelFreeze"
    }
    $freeze = Get-Content -Raw -Encoding utf8 $modelFreeze | ConvertFrom-Json
    if ($freeze.status -ne "CONFIRMATORY_FROZEN" -or $freeze.design_sha256 -ne $designHash) {
        throw "Freeze confirmatorio inválido para ${slug}."
    }

    $runDir = Join-Path $root ("confirmatory_" + $slug)
    $n = [int]$design.confirmatory.initial_n
    while ($true) {
        $collectorArguments = @(
            "-3.12", "scripts\collect_ep1_nvidia.py",
            "--design", $designPath,
            "--mode", "confirmatory",
            "--model", $profile.Model,
            "--freeze", $modelFreeze,
            "--out", $runDir,
            "--n", $n.ToString(),
            "--timeout", $TimeoutSeconds.ToString(),
            "--max-attempts", $MaxAttempts.ToString(),
            "--retry-sleep-seconds", $RetrySleepSeconds.ToString([Globalization.CultureInfo]::InvariantCulture)
        )
        if (Test-Path -LiteralPath $runDir) {
            $collectorArguments += "--resume"
        }
        Write-Host "`n=== E-P1 confirmatory: $($profile.Model), N=$n ===" -ForegroundColor Cyan
        & py @collectorArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Adquisición interrumpida para $($profile.Model). Ejecuta nuevamente este script para reanudar."
        }

        $analysisDir = Join-Path $runDir ("analysis_n" + $n)
        if (-not (Test-Path -LiteralPath $analysisDir)) {
            & py -3.12 scripts\analyze_ep1.py `
                --sessions-dir $runDir `
                --out $analysisDir `
                --replication-freeze $modelFreeze
            $analysisExit = $LASTEXITCODE
        } else {
            $analysisExit = 0
        }
        $verdictPath = Join-Path $analysisDir "ep1_verdict.json"
        if (-not (Test-Path -LiteralPath $verdictPath)) {
            throw "El análisis no produjo verdict: $verdictPath"
        }
        $verdict = Get-Content -Raw -Encoding utf8 $verdictPath | ConvertFrom-Json
        if ($verdict.verdict -eq "underpowered" -and -not $verdict.terminal) {
            $nextN = [int]$verdict.gates.power.next_total_n
            if ($nextN -le $n) {
                throw "El power gate devolvió una extensión no creciente."
            }
            Write-Host "Power gate: extensión preregistrada N=$n -> N=$nextN" -ForegroundColor Yellow
            $n = $nextN
            continue
        }
        if ($analysisExit -ne 0 -and $verdict.verdict -ne "underpowered") {
            throw "El análisis terminó con código $analysisExit y verdict $($verdict.verdict)."
        }
        Write-Host "Terminal verdict for $($profile.Model): $($verdict.verdict)" -ForegroundColor Green
        break
    }
}
