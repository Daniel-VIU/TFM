"""
Formulación espacio-temporal (Opción A).

Predice el precio individual del inmueble y evalúa si añadir el contexto
temporal del distrito (nivel de precio y variaciones recientes) mejora la
predicción frente a usar solo los atributos del inmueble.

Compara, en ambas configuraciones (sin contexto / con contexto):
- Random forest
- Gradient boosting
- Red neuronal densa (perceptrón multicapa)

En esta formulación el precio individual es muy volátil y no lineal, y
un modelo univariante de series temporales (ARIMA) no es aplicable, ya
que no incorpora atributos transversales. La comparación relevante es,
por tanto, entre modelos de aprendizaje automático con y sin la señal
temporal del distrito.

Uso:
    python -m src.modelos_espaciotemporal [--rapido]
Genera:
    results/espaciotemporal_metricas.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metricas import resumen
from .preparacion_datos import BINARIAS, CONTINUAS, PROCESSED, RESPUESTA

SEED = 2025
RESULTS = Path(__file__).resolve().parents[1] / "results"

# Atributos del inmueble (sin contexto temporal).
ATRIBUTOS = CONTINUAS + BINARIAS
# Variables de contexto temporal del distrito.
CONTEXTO = ["ctx_precio_distrito", "ctx_var_3m", "ctx_var_12m"]


def cargar() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "espaciotemporal.parquet")
    return df.dropna(subset=ATRIBUTOS + CONTEXTO + [RESPUESTA])


def modelos(rapido: bool) -> dict:
    n = 100 if rapido else 400
    return {
        "Random forest": RandomForestRegressor(
            n_estimators=n, max_features="sqrt",
            random_state=SEED, n_jobs=-1),
        "Gradient boosting": GradientBoostingRegressor(
            n_estimators=n, learning_rate=0.1, max_depth=5,
            random_state=SEED),
        "Red neuronal (MLP)": Pipeline([
            ("esc", StandardScaler()),
            ("mod", MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=300,
                early_stopping=True, random_state=SEED)),
        ]),
    }


def main(rapido: bool = False) -> pd.DataFrame:
    df = cargar()
    if rapido:
        df = df.sample(8000, random_state=SEED)
    print(f"Inmuebles: {len(df)}")

    y = df[RESPUESTA]
    filas = []
    for config, cols in [("Sin contexto", ATRIBUTOS),
                         ("Con contexto", ATRIBUTOS + CONTEXTO)]:
        X = df[cols]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=SEED)
        for nombre, modelo in modelos(rapido).items():
            modelo.fit(X_tr, y_tr)
            y_hat = modelo.predict(X_te)
            filas.append({"Configuración": config, "Modelo": nombre,
                          **resumen(y_te, y_hat)})

    tabla = (pd.DataFrame(filas)
             .set_index(["Configuración", "Modelo"]).round(3))
    print("\n", tabla, sep="")

    # Mejora relativa del MAPE al añadir contexto, por modelo.
    print("\nMejora del MAPE al añadir contexto temporal:")
    for nombre in modelos(rapido):
        sin = tabla.loc[("Sin contexto", nombre), "MAPE"]
        con = tabla.loc[("Con contexto", nombre), "MAPE"]
        print(f"  {nombre:20s}: {sin:.2f}% -> {con:.2f}%  "
              f"({(sin - con) / sin * 100:+.1f}%)")

    RESULTS.mkdir(exist_ok=True)
    tabla.to_csv(RESULTS / "espaciotemporal_metricas.csv")
    return tabla


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    main(rapido=ap.parse_args().rapido)
