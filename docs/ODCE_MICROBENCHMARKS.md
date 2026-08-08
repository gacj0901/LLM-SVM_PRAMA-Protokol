# Microbenchmarks exploratorios para ODCE-v0

Esta batería está diseñada para adquisición rápida y reanudable antes de una
calibración empírica. No congela ODCE, PRAMA ni D_O v9, y no constituye una
configuración de leaderboard.

## Contenido

- `smoke`: 2 casos (1 GSM8K + 1 IFEval).
- `pilot`: 8 casos (4 + 4), incluyendo `smoke`.
- `full`: 24 casos (12 + 12), incluyendo `pilot`.

GSM8K usa una adaptación zero-shot con una línea final numérica. IFEval usa
los prompts seleccionados sin cambios y verificadores locales deterministas
para restricciones sencillas. Su puntuación se declara compatible, pero no es
la puntuación del harness oficial de leaderboard.

Cada solicitud está limitada por defecto a 384 tokens, 120 segundos y dos
intentos. La corrida se detiene al alcanzar dos timeouts, el primer caso que
agote sus intentos con HTTP 500, o 30 minutos. Cada
caso se guarda inmediatamente y `--resume` reutiliza únicamente artefactos con
la misma identidad de suite, prompt y perfil de generación.

## Preparación de PowerShell

```powershell
Set-Location 'C:\Users\THINKPAD\Desktop\Aptadynamik Cybernetics\AtadynamiK-GIT\LLM-SVM_PRAMA-Protokol'
py -3.12 scripts\run_odce_microbenchmarks.py --model nvidia/nemotron-3-super-120b-a12b --tier pilot --output-dir run_outputs\odce_microbenchmarks_v1\nemotron-super --dry-run
```

El segundo comando sólo valida el plan; no llama al modelo ni crea resultados.

Para NVIDIA, carga la clave sin mostrarla:

```powershell
$secureKey = Read-Host 'NVIDIA_API_KEY' -AsSecureString
$env:NVIDIA_API_KEY = [System.Net.NetworkCredential]::new('', $secureKey).Password
Remove-Variable secureKey
if (-not $env:NVIDIA_API_KEY) { throw 'NVIDIA_API_KEY no fue configurada' }
```

## NVIDIA: smoke y pilot

Nota de disponibilidad del 7 de agosto de 2026:

- `mistralai/mistral-medium-3.5-128b` llegó a EOL a las 09:00 UTC y el runner
  impide nuevas adquisiciones con esa identidad;
- el endpoint de Nemotron Super devolvió HTTP 500 incluso en el preflight de
  ocho tokens;
- Nemotron Ultra continúa publicado como endpoint gratuito y debe verificarse
  con el preflight antes de iniciar su smoke.

```powershell
py -3.12 scripts\preflight_nvidia_nemotron.py --execute --model nvidia/nemotron-3-ultra-550b-a55b --timeout 120 --top-logprobs 5
```

Si el preflight de Ultra termina correctamente, ejecuta su `smoke`:

```powershell
py -3.12 scripts\run_odce_microbenchmarks.py --model nvidia/nemotron-3-ultra-550b-a55b --tier smoke --output-dir run_outputs\odce_microbenchmarks_v1\nemotron-ultra
```

Si termina sin timeouts, amplía el mismo directorio a `pilot`; los dos casos
ya guardados se reutilizan:

```powershell
py -3.12 scripts\run_odce_microbenchmarks.py --model nvidia/nemotron-3-ultra-550b-a55b --tier pilot --output-dir run_outputs\odce_microbenchmarks_v1\nemotron-ultra --resume
```

No ejecutes `full` por rutina. Úsalo sólo para los modelos cuyo `pilot` quede
`COMPLETE` con latencias aceptables:

```powershell
py -3.12 scripts\run_odce_microbenchmarks.py --model nvidia/nemotron-3-ultra-550b-a55b --tier full --output-dir run_outputs\odce_microbenchmarks_v1\nemotron-ultra --resume
```

## Hermes local

Comprueba primero que el modelo y el servidor Ollama estén disponibles:

```powershell
ollama show hermes3:8b
Invoke-RestMethod http://localhost:11434/api/tags | Select-Object -ExpandProperty models | Select-Object name
```

Si el segundo comando no conecta, ejecuta `ollama serve` en otra ventana de
PowerShell. Después corre `smoke` y, sólo si termina, `pilot`:

```powershell
py -3.12 scripts\run_odce_microbenchmarks.py --model hermes3:8b --tier smoke --output-dir run_outputs\odce_microbenchmarks_v1\hermes3-8b
py -3.12 scripts\run_odce_microbenchmarks.py --model hermes3:8b --tier pilot --output-dir run_outputs\odce_microbenchmarks_v1\hermes3-8b --resume
```

## Reanudación y diagnóstico

Repite exactamente el último comando con `--resume` después de una
interrupción. Para inspeccionar el estado:

```powershell
Get-Content run_outputs\odce_microbenchmarks_v1\nemotron-super\manifest.json
Get-ChildItem run_outputs\odce_microbenchmarks_v1 -Recurse -Filter manifest.json | ForEach-Object { Get-Content $_.FullName | ConvertFrom-Json } | Select-Object model_profile, tier, status, completed_count, failed_count, timeout_count, provider_server_error_count, halted_reason
```

Cada directorio contiene:

- `sessions/<session_id>/raw.json`: respuesta y observables token-level;
- `sessions/<session_id>/verification.json`: evaluación post-generación;
- `domain_return_observations.jsonl`: outcomes compatibles con ODCE;
- `manifest.json`: checkpoint y resumen de ejecución.

`functional_gain` es la fracción de condiciones verificadas; en GSM8K es
binaria. `verified_outcome` vale uno sólo cuando el caso completo pasa. Ambos
se vuelven disponibles en el último índice de ventana observado; no se
retroinyectan en índices anteriores. En esta primera suite,
`external_integration` permanece `UNAVAILABLE`.

Para repetir aisladamente un caso truncado sin cambiar el perfil de la corrida
principal, usa otro directorio y `--item-id`. Por ejemplo, una sensibilidad a
768 tokens para `ifeval-1075` en Hermes:

```powershell
py -3.12 scripts\run_odce_microbenchmarks.py --model hermes3:8b --tier pilot --item-id ifeval-1075 --max-tokens 768 --timeout 180 --max-attempts 1 --output-dir run_outputs\odce_microbenchmarks_v1\hermes3-8b-cap768
```

## Criterio de avance

Avanza de `smoke` a `pilot` sólo si ambos casos terminan y no hay timeout. No
continúes con `full` si el pilot agota el presupuesto de timeouts, alcanza el
deadline, o muestra latencias cercanas al timeout en una fracción sustancial de
los casos. En ese punto conviene analizar primero los artefactos ya guardados.

## Banco de rango operacional

`data/odce_operational_calibration_v1.json` complementa la suite anterior; no
la sustituye. Contiene cuatro trayectorias largas `dynamic_range` y cuatro
casos cortos `evidence_integration`. Sus tamaños siguen siendo pequeños:

- `smoke`: 2 casos;
- `pilot`: 4 casos, incluidos los de smoke;
- `full`: 8 casos, incluidos los de pilot.

Las trayectorias largas intentan recorrer condiciones basales, transición con
recuperación, perturbación persistente y organización explícita. El prompt no
garantiza que esos estados aparezcan: sólo la proyección numérica posterior
puede demostrarlo. Los casos de evidencia producen
`external_integration` como fracción de anclas externas conservadas. Esa
evaluación ocurre después de PRAMA y D_O v9 y no modifica su estado.

Valida primero el plan y usa 768 tokens sólo porque el caso dinámico necesita
espacio para aproximarse a 32 ventanas de 16 tokens:

```powershell
py -3.12 scripts\run_odce_microbenchmarks.py --suite data\odce_operational_calibration_v1.json --model nvidia/nemotron-3-ultra-550b-a55b --tier smoke --max-tokens 768 --output-dir run_outputs\odce_operational_calibration_v1\nemotron-ultra --dry-run
```

El calibrador exploratorio exige ahora cobertura además del número bruto de
índices. Los valores predeterminados son 20 observaciones, 10% de índices,
cinco sesiones y 10% de sesiones. Cada correspondencia queda marcada como
`CALIBRATED`, `PARTIALLY_CALIBRATED` o `UNCALIBRATED`; sólo la primera permite
interpretación instrumental. Ninguno de esos estados crea un freeze.
