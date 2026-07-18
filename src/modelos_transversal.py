"""
Formulación transversal: predicción del precio del anuncio.

Compara regresión lineal múltiple (línea base), random forest y
gradient boosting sobre el conjunto idealista18 depurado, con:
- variable objetivo en escala logarítmica (log del precio), que corrige
  la asimetría de la distribución de precios y evita que los inmuebles de
  precio extremo dominen el error;
- partición aleatoria 80/20 (semilla fija);
- selección de hiperparámetros por validación cruzada de 5 particiones
  dentro del tramo de entrenamiento;
- evaluación final única sobre el tramo de prueba, con métricas dadas
  tanto en escala logarítmica como en euros (revertidas con exp);
- test de robustez de los dos métodos de ensamblado sobre varias
  semillas de partición.

Uso:
    python -m src.modelos_transversal [--rapido]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metricas import rmse, mae, mape, r2
from .preparacion_datos import BINARIAS, CONTINUAS, PROCESSED, RESPUESTA

SEED = 2025
SEMILLAS_ROBUSTEZ = [2025, 7, 123, 2024, 99]
# Modelos sometidos al test de robustez frente a la partición.
MODELOS_ROBUSTEZ = ["Random forest", "Gradient boosting"]
RESULTS = Path(__file__).resolve().parents[1] / "results"

PREDICTORES = CONTINUAS + BINARIAS


def cargar():
    df = pd.read_parquet(PROCESSED / "transversal.parquet")
    df = df.dropna(subset=PREDICTORES + [RESPUESTA])
    df = df[df[RESPUESTA] > 0]                 # log exige precio positivo
    return df[PREDICTORES], df[RESPUESTA]


def metricas_dobles(y_log_real, y_log_pred):
    """Métricas en escala log y en euros (revertidas con exp).

    El modelo se entrena y predice en log(precio); las métricas en euros
    se obtienen aplicando la exponencial a valores reales y predichos, lo
    que las hace directamente interpretables.
    """
    y_eur = np.exp(y_log_real)
    yhat_eur = np.exp(y_log_pred)
    return {
        "R2_log": r2(y_log_real, y_log_pred),
        "RMSE_log": rmse(y_log_real, y_log_pred),
        "MAE_eur": mae(y_eur, yhat_eur),
        "RMSE_eur": rmse(y_eur, yhat_eur),
        "MAPE": mape(y_eur, yhat_eur),
    }


def construir_modelos(rapido):
    """Modelos y mallas de hiperparámetros."""
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


def ajustar(nombre, modelo, malla, X_tr, y_tr):
    """Ajusta un modelo, con GridSearchCV si tiene malla."""
    if malla:
        busqueda = GridSearchCV(
            modelo, malla, cv=5,
            scoring="neg_root_mean_squared_error", n_jobs=-1,
        )
        busqueda.fit(X_tr, y_tr)
        return busqueda.best_estimator_, busqueda.best_params_
    return modelo.fit(X_tr, y_tr), {}


def test_robustez(nombre, modelo_base, X, y_log):
    """Reentrena y evalúa un modelo sobre las semillas de robustez.

    El modelo conserva los hiperparámetros seleccionados en la partición
    principal; solo cambia la semilla del reparto entrenamiento/prueba.
    Devuelve la tabla de métricas por semilla.
    """
    filas = []
    for s in SEMILLAS_ROBUSTEZ:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y_log, test_size=0.2, random_state=s)
        m = modelo_base.__class__(**{**modelo_base.get_params()})
        m.fit(Xtr, ytr)
        d = metricas_dobles(yte, m.predict(Xte))
        filas.append({"Semilla": s, "R2_log": round(d["R2_log"], 4),
                      "MAE_eur": round(d["MAE_eur"], 1),
                      "RMSE_eur": round(d["RMSE_eur"], 1)})
    rob = pd.DataFrame(filas).set_index("Semilla")
    print(f"\nTest de robustez de '{nombre}' sobre "
          f"{len(SEMILLAS_ROBUSTEZ)} semillas de partición:")
    print(rob)
    print(f"R2_log medio: {rob['R2_log'].mean():.4f} "
          f"± {rob['R2_log'].std():.4f}")
    return rob


def main(rapido=False):
    X, y = cargar()
    if rapido:
        X, y = X.iloc[:15000], y.iloc[:15000]
    y_log = np.log(y)                          # objetivo en escala log

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_log, test_size=0.2, random_state=SEED
    )
    print(f"Entrenamiento: {len(X_tr)}  |  Prueba: {len(X_te)}")

    filas, mejores = [], {}
    for nombre, (modelo, malla) in construir_modelos(rapido).items():
        ajustado, params = ajustar(nombre, modelo, malla, X_tr, y_tr)
        mejores[nombre] = (ajustado, params)
        if params:
            print(f"{nombre}: {params}")
        y_hat = ajustado.predict(X_te)
        filas.append({"Modelo": nombre, **metricas_dobles(y_te, y_hat)})

    tabla = pd.DataFrame(filas).set_index("Modelo").round(4)
    print("\nComparativa (objetivo en log-precio):")
    print(tabla)

    RESULTS.mkdir(exist_ok=True)
    tabla.to_csv(RESULTS / "transversal_metricas.csv")

    # Persistir los hiperparámetros seleccionados por GridSearchCV, de
    # modo que las combinaciones ganadoras reportadas en la memoria
    # queden respaldadas por un fichero reproducible.
    hiper = pd.DataFrame(
        [{"Modelo": nombre, **params}
         for nombre, (_, params) in mejores.items() if params]
    ).set_index("Modelo")
    hiper.to_csv(RESULTS / "transversal_hiperparametros.csv")
    print("\nHiperparámetros seleccionados:")
    print(hiper)

    # Test de robustez de los dos métodos de ensamblado: se reentrenan
    # con sus hiperparámetros ya seleccionados sobre las mismas cinco
    # particiones aleatorias, lo que permite comparar su estabilidad y
    # comprobar que el orden de la comparativa se mantiene en todas.
    sufijos = {"Random forest": "rf", "Gradient boosting": "gb"}
    for nombre in MODELOS_ROBUSTEZ:
        modelo_base, _ = mejores[nombre]
        rob = test_robustez(nombre, modelo_base, X, y_log)
        rob.to_csv(RESULTS / f"transversal_robustez_{sufijos[nombre]}.csv")
    return tabla


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    main(rapido=ap.parse_args().rapido)
