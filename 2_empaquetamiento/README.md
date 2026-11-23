# Seoul Bikes – Empaquetamiento de modelos con uv + scikit-learn

Integrantes: Juan Esteban Giraldo Giraldo / Federico Avalos Zuleta

Entorno de desarrollo: Visual Studio Code + extensiones Python y Jupyter.

## Reconstruir el entorno

```bash
uv python install 3.12
uv venv venv_ml --python 3.12
# Windows PowerShell: .\venv_ml\Scripts\Activate.ps1   |   macOS/Linux: source venv_ml/bin/activate
uv pip sync requirements-lock.txt
python -m ipykernel install --user --name venv_ml --display-name "Python (venv_ml)"
```

## Ejecución (en orden, con el kernel `Python (venv_ml)`)

1. `notebooks/01_seoul_entrenamiento_y_persistencia.ipynb`: limpia los datos, entrena el MLP con GridSearchCV, evalúa, guarda los pipelines con joblib (cada uno con su metadata dentro del archivo) y verifica con `assert` que el modelo recargado predice igual.
2. `notebooks/02_seoul_recuperacion_y_prediccion.ipynb`: carga los artefactos sin reentrenar, comprueba el SHA-256 del dataset contra el guardado en la metadata, reconstruye el conjunto de prueba, comprueba que las métricas y predicciones coinciden con las del Notebook 1 y predice nuevas observaciones.

## Artefactos generados en `artifacts/`

Como piden las diapositivas de empaquetamiento (4 y 8), cada `.joblib` es un diccionario con el modelo y su metadata dentro del mismo archivo:

```python
artefacto = joblib.load("artifacts/seoul_pipeline_modelo.joblib")
pipeline_modelo = artefacto["modelo"]
metadata = artefacto["metadata"]   # python, sklearn, numpy, fecha_entrenamiento, hash_dataset_sha256, ...
```

- `seoul_pipeline_prep.joblib`: `{"modelo": pipeline_prep, "metadata": {...}}`; preprocesamiento (OneHotEncoder de las categóricas).
- `seoul_pipeline_modelo.joblib`: `{"modelo": pipeline_modelo, "metadata": {...}}`; StandardScaler + MLPRegressor (32, 16) con salida estandarizada; es el mismo modelo del cuaderno principal.
- La metadata incluye versiones de Python (`sys.version`), scikit-learn y NumPy, fecha de entrenamiento, archivo del dataset con su hash SHA-256 y número de filas de entrenamiento, además de variables, categorías, hiperparámetros, partición y métricas.
- `seoul_metadata.json`: copia legible de esa misma metadata (no se necesita para cargar el modelo).
- `seoul_predicciones_referencia.csv`: predicciones de prueba para verificar la integridad entre sesiones.

La carpeta `venv_ml` no se incluye en la entrega: se reconstruye con `requirements-lock.txt`.

## Aplicación: bicicletas para mañana

`app/servidor.py` carga `seoul_pipeline_prep.joblib` y `seoul_pipeline_modelo.joblib` (sin reentrenar; toma cada pipeline de la clave `"modelo"` y la metadata de `"metadata"`) y sirve la página `app/index.html`, donde se escribe el pronóstico del clima de mañana y se obtiene cuántas bicicletas tener disponibles (pico predicho + margen de seguridad de 633).

```bash
# desde la carpeta LabUV, con venv_ml activo y después de ejecutar el Notebook 1
python app/servidor.py
# abrir http://localhost:8000
```

No necesita librerías adicionales: usa el servidor HTTP de la biblioteca estándar de Python.
