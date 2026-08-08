# DSEB-v0 smoke conversacional

El perfil `smoke` conserva `DSEB_v0`, `DSEB-v0` y la partición
`exploratory`. Ejecuta 12 turnos de una sola conversación y no declara ningún
freeze.

## Etapas

1. `run_dseb_smoke.py` ejecuta primero el offline-preflight como compuerta,
   adquiere los 12 turnos y preserva logprobs por token.
2. Sus verificaciones son sólo previews: declaran
   `eligible_for_odce=false`, `available_at_index=null` y
   `PENDING_D_O_V9`.
3. `project_dseb_smoke.py` proyecta exclusivamente el canal numérico por
   PRAMA y D_O v9.
4. Después de D_O, vuelve a ejecutar el verificador determinista, registra
   cada outcome en la ventana terminal y ejecuta ODCE-v0.

El orden efectivo es:

```text
cierre terminal -> O_D -> PRAMA -> D_O v9 -> verificador
                -> registro outcome -> ODCE
```

`event_index` y `available_at_index` son ordinales de la secuencia canónica de
pares `(turn_index, window_index)`. No se reconstruyen desde el turno.

## PowerShell: validación sin llamadas

```powershell
py -3.12 scripts\run_dseb_offline_preflight.py `
  --protocol benchmarks\dseb_v0\configs\dseb_v0_smoke.json `
  --seed 7 `
  --output-dir run_outputs\dseb_v0\offline-preflight-smoke-seed0007

py -3.12 scripts\run_dseb_smoke.py `
  --model nvidia/nemotron-3-ultra-550b-a55b `
  --output-dir run_outputs\dseb_v0\smoke-dry-run `
  --seed 7 `
  --dry-run
```

El `--dry-run` no contacta al proveedor ni escribe el directorio de salida.

## PowerShell: Nemotron Ultra

```powershell
py -3.12 scripts\run_dseb_smoke.py `
  --model nvidia/nemotron-3-ultra-550b-a55b `
  --output-dir run_outputs\dseb_v0\smoke\nemotron-ultra-seed0007 `
  --seed 7 `
  --max-tokens 256 `
  --max-attempts 1
```

Si la adquisición se interrumpe, repetir el mismo comando añadiendo
`--resume`.

## PowerShell: Hermes 3 local

```powershell
py -3.12 scripts\run_dseb_smoke.py `
  --model hermes3:8b `
  --output-dir run_outputs\dseb_v0\smoke\hermes3-8b-seed0007-plain-v1 `
  --seed 7 `
  --max-tokens 256 `
  --max-attempts 1
```

## Proyección causal completa

La declaración actual está vinculada al checkout local de PRAMA en commit
`cb41d590207a09d498532b8c599e12ecab7a0dcb`.

Para Nemotron:

```powershell
py -3.12 scripts\project_dseb_smoke.py `
  --acquisition-run run_outputs\dseb_v0\smoke\nemotron-ultra-seed0007 `
  --output-dir run_outputs\dseb_v0\projected\nemotron-ultra-seed0007 `
  --prama-source-root run_outputs\prama_kernel_cb41d590\PRAMA-Protokol-py
```

Para Hermes:

```powershell
py -3.12 scripts\project_dseb_smoke.py `
  --acquisition-run run_outputs\dseb_v0\smoke\hermes3-8b-seed0007-plain-v1 `
  --output-dir run_outputs\dseb_v0\projected\hermes3-8b-seed0007-plain-v1 `
  --prama-source-root run_outputs\prama_kernel_cb41d590\PRAMA-Protokol-py
```

La salida completa contiene:

- `prama_trajectory.jsonl`;
- `structural_observations.jsonl`;
- `domain_return_observations.jsonl`;
- `structural_conversion_differentials.jsonl`;
- `report.json`.

El reporte final debe declarar `COMPLETE_EXPLORATORY_CAUSAL`. La normalización
sigue siendo identidad exploratoria y `contract_freeze_sha256` permanece
`null` deliberadamente.
