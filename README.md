# Seoul Bikes: predicción de la demanda de bicicletas compartidas

Trabajo final de Machine Learning (metodología CRISP-DM).

**Integrantes:** Juan Esteban Giraldo Giraldo y Federico Avalos Zuleta

- Cuaderno principal en Colab: [ENLACE AL COLAB]
- App desplegada en Render: [ENLACE A RENDER]

## Resultados principales

| Métrica (conjunto de prueba) | Valor |
|---|---|
| RMSE | 227,4 bicicletas/hora |
| MAE | 147,8 bicicletas/hora |
| R² | 0,869 |

Frente a planificar con el promedio de cada hora, el modelo reduce en **64,3 %** las bicicletas faltantes + sobrantes; la meta era 20 %.

## Contenido

### `1_trabajo/`
- `Informe_final_Seoul_Bikes.docx`: informe final del trabajo.
- `EDA_de_Seoul_Bike_Rent_Entrega_v3.ipynb`: cuaderno principal (exploración, preparación, modelado, evaluación y modelo prescriptivo). Está hecho para Colab y se entrega ya ejecutado.
- `EDA_de_Seoul_Bike_Rent_Entrega_v3.html`: el mismo cuaderno en HTML, para verlo sin Jupyter.

### `2_empaquetamiento/`
Empaquetamiento del modelo con `uv` + scikit-learn + joblib, y la app que lo usa.
- `notebooks/01_seoul_entrenamiento_y_persistencia.ipynb` (+ `.html`): entrena el MLP, lo evalúa y guarda los pipelines con joblib; verifica con `assert` que el modelo recargado predice igual.
- `notebooks/02_seoul_recuperacion_y_prediccion.ipynb` (+ `.html`): carga los artefactos sin reentrenar, comprueba el hash del dataset y que las métricas coinciden, y predice nuevas observaciones.
- `artifacts/seoul_pipeline_prep.joblib` y `artifacts/seoul_pipeline_modelo.joblib`: pipelines de preprocesamiento y modelo; cada archivo es `{"modelo": ..., "metadata": ...}`.
- `artifacts/seoul_metadata.json`: copia legible de la metadata (versiones, fecha, hash SHA-256 del dataset, hiperparámetros, métricas).
- `artifacts/seoul_predicciones_referencia.csv`: predicciones de prueba para verificar la integridad entre sesiones.
- `dataset/SeoulBikeData.csv`: dataset original (UCI 560).
- `app/servidor.py` y `app/index.html`: app web "Bicicletas para mañana" (pico predicho + margen de seguridad de 633).
- `requirements.txt` / `requirements-lock.txt`: dependencias directas y versiones exactas.
- `README.md`: detalle del empaquetamiento.

### `3_presentacion/`
- `Seoul Bikes – Trabajo final ML (CRISP-DM).pdf`: diapositivas de la presentación.

## Cómo reconstruir el entorno y ejecutar

Desde la carpeta `2_empaquetamiento`:

```bash
uv python install 3.12
uv venv venv_ml --python 3.12
# Windows PowerShell: .\venv_ml\Scripts\Activate.ps1   |   macOS/Linux: source venv_ml/bin/activate
uv pip sync requirements-lock.txt
python -m ipykernel install --user --name venv_ml --display-name "Python (venv_ml)"
```

Notebooks, en orden y con el kernel `Python (venv_ml)`:

1. `notebooks/01_seoul_entrenamiento_y_persistencia.ipynb`
2. `notebooks/02_seoul_recuperacion_y_prediccion.ipynb`

App (con `venv_ml` activo):

```bash
python app/servidor.py
# abrir http://localhost:8000
```

La carpeta `venv_ml` no se incluye en el repositorio: se reconstruye con `requirements-lock.txt`.
