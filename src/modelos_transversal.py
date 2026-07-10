"""
Formulación transversal: predicción del precio del anuncio.

Compara regresión lineal múltiple (línea base), random forest y
gradient boosting sobre el conjunto idealista18 depurado, con:
- partición aleatoria 80/20 (semilla fija),
- selección de hiperparámetros por validación cruzada de 5 particiones
  dentro del tramo de entrenamiento,
- evaluación final única sobre el tramo de prueba.

Uso:
    python -m src.modelos_transversal [--rapido]
La opción --rapido reduce las mallas de hiperparámetros y submuestrea
el conjunto para una ejecución de comprobación.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metricas import resumen
from .preparacion_datos import BINARIAS, CONTINUAS, PROCESSED, RESPUESTA

SEED = 2025
RESULTS = Path(__file__).resolve().parents[1] / "results"

PREDICTORES = CONTINUAS + BINARIAS


def cargar() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED / "transversal.parquet")
    df = df.dropna(subset=PREDICTORES + [RESPUESTA])
    return df[PREDICTORES], df[RESPUESTA]


def construir_modelos(rapido: bool) -> dict:
    """Modelos y mallas de hiperparámetros de la memoria."""
    if rapido:
        malla_rf = {"n_estimators": [100], "max_features": ["sqrt"]}
        malla_gb = {"n_estimators": [100], "learning_rate": [0.1],
                    "max_depth": [3]}
    else:
        malla_rf = {
            "n_estimators": [200, 500],
            "max_features": ["sqrt", 1.0 / 3.0],
        }
        malla_gb = {
            "n_estimators": [200, 500],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
        }
    return {
        # La regresión se entrena sobre variables estandarizadas; los
        # árboles, sobre las variables originales (invariantes a
        # transformaciones monótonas).
        "Regresion lineal": (
            Pipeline([("esc", StandardScaler()),
                      ("mod", LinearRegression())]),
            {},
        ),
        "Random forest": (
            RandomForestRegressor(random_state=SEED, n_jobs=-1),
            malla_rf,
        ),
        "Gradient boosting": (
            GradientBoostingRegressor(random_state=SEED),
            malla_gb,
        ),
    }


def main(rapido: bool = False) -> pd.DataFrame:
    X, y = cargar()
    if rapido:
        X, y = X.iloc[:15000], y.iloc[:15000]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    print(f"Entrenamiento: {len(X_tr)}  |  Prueba: {len(X_te)}")

    filas = []
    for nombre, (modelo, malla) in construir_modelos(rapido).items():
        if malla:
            busqueda = GridSearchCV(
                modelo, malla, cv=5,
                scoring="neg_root_mean_squared_error", n_jobs=-1,
            )
            busqueda.fit(X_tr, y_tr)
            mejor = busqueda.best_estimator_
            print(f"{nombre}: mejores hiperparámetros "
                  f"{busqueda.best_params_}")
        else:
            mejor = modelo.fit(X_tr, y_tr)
        y_hat = mejor.predict(X_te)
        filas.append({"Modelo": nombre, **resumen(y_te, y_hat)})

    tabla = pd.DataFrame(filas).set_index("Modelo").round(3)
    print("\n", tabla, sep="")

    RESULTS.mkdir(exist_ok=True)
    tabla.to_csv(RESULTS / "transversal_metricas.csv")
    return tabla


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    main(rapido=ap.parse_args().rapido)
