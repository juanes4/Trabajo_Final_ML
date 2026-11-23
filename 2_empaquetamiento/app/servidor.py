"""
Servicio de predicción: bicicletas para mañana (Seoul Bikes).

Carga los pipelines empaquetados con joblib en el Notebook 1 (cada archivo es un
diccionario {"modelo": pipeline, "metadata": {...}}) y sirve una página web para predecir cuántas bicicletas tener disponibles mañana.

Uso (desde la carpeta LabUV, con venv_ml activo):
    python app/servidor.py
Luego abrir http://localhost:8000 en el navegador.

Solo usa la biblioteca estándar más las librerías de requirements.txt.
"""
import json
import math
import os
import platform
import sys
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

RAIZ = Path(__file__).resolve().parent.parent
RUTA_ARTIFACTOS = RAIZ / "artifacts"
RUTA_PREP = RUTA_ARTIFACTOS / "seoul_pipeline_prep.joblib"
RUTA_MODELO = RUTA_ARTIFACTOS / "seoul_pipeline_modelo.joblib"
RUTA_PAGINA = Path(__file__).resolve().parent / "index.html"

# Margen de seguridad del modelo prescriptivo: percentil 95 del error en el pico diario
# (sección 4.4 del cuaderno principal EDA_de_Seoul_Bike_Rent_Entrega_v3.ipynb)
MARGEN_SEGURIDAD = 633
# En Render (u otro hosting) PORT viene definido y se escucha en todas las interfaces;
# en local se usa 8000 y solo 127.0.0.1.
PUERTO = int(os.environ.get("PORT", 8000))
HOST = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"

ESTACIONES = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
DIAS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------- Carga de los artefactos (sin fit) ----------
for ruta in (RUTA_PREP, RUTA_MODELO):
    if not ruta.exists():
        sys.exit(f"No se encontró {ruta.name} en artifacts/. Ejecuta primero el Notebook 1.")

# Advertencia: joblib puede ejecutar código al cargar; solo se cargan artefactos propios.
# Cada artefacto es un diccionario {"modelo": pipeline, "metadata": {...}}; la metadata viaja dentro del joblib.
artefacto_prep = joblib.load(RUTA_PREP)
artefacto_modelo = joblib.load(RUTA_MODELO)
pipeline_prep = artefacto_prep["modelo"]
pipeline_modelo = artefacto_modelo["modelo"]
metadata = artefacto_modelo["metadata"]

version_entreno = metadata["versiones"]["scikit-learn"]
if version_entreno != sklearn.__version__:
    print(f"Advertencia: el modelo se entrenó con scikit-learn {version_entreno} "
          f"y el entorno tiene {sklearn.__version__}. Reconstruye venv_ml con requirements-lock.txt.")


def temperatura_hora(hora, tmin, tmax):
    """Curva diaria de temperatura: mínima a las 5 h y máxima a las 15 h."""
    if 5 <= hora <= 15:
        f = (1 - math.cos(math.pi * (hora - 5) / 10)) / 2
    else:
        f = (1 + math.cos(math.pi * (((hora - 15) + 24) % 24) / 14)) / 2
    return tmin + (tmax - tmin) * f


def predecir_dia(fecha, tmin, tmax, humedad, lluvia, nieve, festivo):
    """Arma las 24 horas del día con las variables crudas, predice con los pipelines y aplica la regla prescriptiva."""
    if tmin > tmax:
        raise ValueError(f"la temperatura mínima ({tmin}) no puede ser mayor que la máxima ({tmax})")
    estacion = ESTACIONES[fecha.month]
    dia = DIAS[fecha.weekday()]
    temps = [temperatura_hora(h, tmin, tmax) for h in range(24)]

    X = pd.DataFrame({
        "Hour": range(24),
        "Humidity(%)": float(humedad),
        "Temperature(°C)": temps,
        "Rainfall(mm)": float(lluvia),
        "Snowfall (cm)": float(nieve),
        "Seasons": estacion,
        "Holiday": "Holiday" if festivo else "No Holiday",
        "Day of week": dia,
    })[metadata["variables_entrada"]]

    pred = np.clip(pipeline_modelo.predict(pipeline_prep.transform(X)), 0, None)
    hora_pico = int(np.argmax(pred))
    pico = float(pred[hora_pico])
    return {
        "fecha": fecha.isoformat(),
        "estacion": estacion,
        "dia": dia,
        "predicciones": [round(float(p), 1) for p in pred],
        "temperaturas": [round(t, 1) for t in temps],
        "hora_pico": hora_pico,
        "pico": round(pico, 1),
        "total": round(float(pred.sum()), 1),
        "margen": MARGEN_SEGURIDAD,
        "disponibilidad": int(math.ceil(pico + MARGEN_SEGURIDAD)),
    }


INFO_MODELO = {
    "archivos": [RUTA_PREP.name, RUTA_MODELO.name],
    "capas": metadata["arquitectura"]["capas_ocultas"],
    "test_RMSE": round(metadata["metricas"]["test_RMSE"], 1),
    "test_MAE": round(metadata["metricas"]["test_MAE"], 1),
    "sklearn_entreno": version_entreno,
    "sklearn_actual": sklearn.__version__,
    "python": platform.python_version(),
    "fecha_entrenamiento": metadata["fecha_entrenamiento"],
}


class Manejador(BaseHTTPRequestHandler):
    def _responder(self, codigo, cuerpo, tipo="application/json; charset=utf-8"):
        datos = cuerpo if isinstance(cuerpo, bytes) else json.dumps(cuerpo, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(datos)))
        self.end_headers()
        self.wfile.write(datos)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._responder(200, RUTA_PAGINA.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/modelo":
            self._responder(200, INFO_MODELO)
        else:
            self._responder(404, {"error": "Ruta no encontrada"})

    def do_POST(self):
        if self.path != "/api/prediccion":
            self._responder(404, {"error": "Ruta no encontrada"})
            return
        try:
            largo = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(largo) or b"{}")
            resultado = predecir_dia(
                fecha=date.fromisoformat(p["fecha"]),
                tmin=float(p["tmin"]), tmax=float(p["tmax"]), humedad=float(p["humedad"]),
                lluvia=float(p.get("lluvia", 0)), nieve=float(p.get("nieve", 0)),
                festivo=bool(p.get("festivo", False)),
            )
            self._responder(200, resultado)
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            self._responder(400, {"error": f"Datos de entrada inválidos: {e}"})

    def log_message(self, formato, *args):
        pass  # sin registro por cada petición


if __name__ == "__main__":
    print(f"Modelo cargado desde {RUTA_MODELO.name} y {RUTA_PREP.name} "
          f"(RMSE de prueba {INFO_MODELO['test_RMSE']}).")
    print(f"Abre http://localhost:{PUERTO} en el navegador. Ctrl+C para detener.")
    ThreadingHTTPServer((HOST, PUERTO), Manejador).serve_forever()
