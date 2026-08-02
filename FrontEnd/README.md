# Frontend NVIDIA

El servidor mantiene `NVIDIA_API_KEY` fuera del navegador y sólo envía al
modelo el historial conversacional con campos `role` y `content`. No añade un
mensaje de sistema, etiquetas experimentales ni datos del monitor.

Desde PowerShell, en la raíz del repositorio:

```powershell
$env:NVIDIA_API_KEY = Read-Host "Pega tu NVIDIA API key" -MaskInput
py -3.12 FrontEnd\server.py
```

Después abre <http://127.0.0.1:8000/>. Elige el modelo, inicia una sesión y
escribe libremente o carga una de las pruebas sugeridas.

Modelos habilitados:

- `nvidia/nemotron-3-super-120b-a12b`
- `mistralai/mistral-medium-3.5-128b`
- `nvidia/nemotron-3-ultra-550b-a55b`

La métrica `Response time` se inicia inmediatamente antes de crear la
solicitud al proveedor y termina cuando finaliza su stream. Los resultados se
exportan bajo `FrontEnd/results/`.
