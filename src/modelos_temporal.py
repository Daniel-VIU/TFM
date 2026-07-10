"""
Formulación temporal: predicción del precio medio (eur/m2) por distrito.

Compara cuatro modelos sobre el panel mensual de 21 distritos:
1. ARIMA por serie (línea base), ajustado sobre la tasa de variación.
2. LSTM entrenada de forma conjunta sobre las ventanas de las 21 series.
3. CNN-LSTM: capas convolucionales 1D antes de las recurrentes.
4. Híbrido ARIMA + random forest sobre los residuos del ARIMA.

Particiones cronológicas 70/20/10 por serie. La normalización
mínimo-máximo de cada serie se ajusta solo con el tramo de
entrenamiento. Las métricas se calculan en la escala original.

Uso:
    python -m src.modelos_temporal [--rapido] [--ventana K]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .metricas import resumen
from .preparacion_datos import PROCESSED

SEED = 2025
RESULTS = Path(__file__).resolve().parents[1] / "results"
VENTANA = 12          # meses de historia que ve el modelo (lookback)
FR_TRAIN, FR_VAL = 0.70, 0.20


# ---------------------------------------------------------------------------
# Utilidades de partición y ventanas
# ---------------------------------------------------------------------------

def particion_cronologica(serie: np.ndarray) -> tuple[int, int]:
    """Índices de corte train/val/test (70/20/10) de una serie."""
    n = len(serie)
    i_tr = int(n * FR_TRAIN)
    i_va = int(n * (FR_TRAIN + FR_VAL))
    return i_tr, i_va


def ventanas(serie: np.ndarray, k: int, ini: int, fin: int):
    """Pares (X, y) con ventana de k valores dentro de [ini, fin).

    La ventana puede empezar antes de `ini` (usa historia previa), pero
    el objetivo y siempre cae dentro del tramo pedido, de modo que
    ninguna observación futura entra en el entrenamiento del tramo
    anterior.
    """
    X, y = [], []
    for t in range(max(ini, k), fin):
        X.append(serie[t - k:t])
        y.append(serie[t])
    return np.array(X), np.array(y)


class EscaladorMinMax:
    """Mínimo-máximo ajustado solo con el tramo de entrenamiento."""

    def ajustar(self, tramo: np.ndarray) -> "EscaladorMinMax":
        self.lo, self.hi = float(tramo.min()), float(tramo.max())
        return self

    def transformar(self, x: np.ndarray) -> np.ndarray:
        return (x - self.lo) / (self.hi - self.lo)

    def invertir(self, x: np.ndarray) -> np.ndarray:
        return x * (self.hi - self.lo) + self.lo


# ---------------------------------------------------------------------------
# Modelo 1: ARIMA por distrito (línea base)
# ---------------------------------------------------------------------------

def arima_por_distrito(panel: dict, orden=(4, 0, 4)) -> dict:
    """ARIMA sobre la tasa de variación, con predicción rodante a 1 mes.

    Siguiendo a Chen et al. (2017), el modelo se ajusta sobre la tasa de
    variación mensual (serie estacionaria) y la predicción se reconvierte
    a niveles multiplicando por el último precio observado. Devuelve
    también los residuos de entrenamiento para el modelo híbrido.
    """
    from statsmodels.tsa.arima.model import ARIMA

    resultados, residuos = {}, {}
    for distrito, serie in panel.items():
        ret = np.diff(serie) / serie[:-1]          # tasa de variación
        i_tr, i_va = particion_cronologica(serie)
        pred_test, real_test = [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Predicción rodante: en cada mes de prueba se reajusta con
            # toda la historia disponible hasta ese momento.
            for t in range(i_va, len(serie) - 1):
                mod = ARIMA(ret[:t], order=orden).fit()
                r_hat = mod.forecast(1)[0]
                pred_test.append(serie[t] * (1 + r_hat))
                real_test.append(serie[t + 1])
            # Residuos en el tramo de entrenamiento (para el híbrido).
            mod_tr = ARIMA(ret[: i_tr - 1], order=orden).fit()
            residuos[distrito] = ret[: i_tr - 1] - mod_tr.fittedvalues
        resultados[distrito] = (np.array(real_test), np.array(pred_test))
    return resultados, residuos


# ---------------------------------------------------------------------------
# Modelos 2 y 3: LSTM y CNN-LSTM sobre el panel conjunto
# ---------------------------------------------------------------------------

def construir_red(tipo: str, k: int, unidades: int = 20):
    """Arquitecturas de la memoria (Chen 2017; Alhussein 2020)."""
    import tensorflow as tf
    from tensorflow.keras import layers

    tf.random.set_seed(SEED)
    entrada = layers.Input(shape=(k, 1))
    x = entrada
    if tipo == "cnn-lstm":
        x = layers.Conv1D(48, 3, activation="relu", padding="same")(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(16, 3, activation="relu", padding="same")(x)
        x = layers.Dropout(0.25)(x)
    x = layers.LSTM(unidades, return_sequences=True)(x)
    x = layers.LSTM(unidades)(x)
    x = layers.Dropout(0.25)(x)
    salida = layers.Dense(1)(x)
    modelo = tf.keras.Model(entrada, salida)
    modelo.compile(optimizer="adam", loss="mae")
    return modelo


def red_sobre_panel(panel: dict, tipo: str, k: int,
                    epocas: int = 150) -> dict:
    """Entrena una única red con las ventanas de las 21 series."""
    import tensorflow as tf

    Xtr, ytr, Xva, yva = [], [], [], []
    esc, tramos_test = {}, {}
    for distrito, serie in panel.items():
        i_tr, i_va = particion_cronologica(serie)
        e = EscaladorMinMax().ajustar(serie[:i_tr])
        s = e.transformar(serie)
        esc[distrito] = e
        X, y = ventanas(s, k, 0, i_tr)
        Xtr.append(X); ytr.append(y)
        X, y = ventanas(s, k, i_tr, i_va)
        Xva.append(X); yva.append(y)
        tramos_test[distrito] = (s, i_va)
    Xtr = np.concatenate(Xtr)[..., None]
    ytr = np.concatenate(ytr)
    Xva = np.concatenate(Xva)[..., None]
    yva = np.concatenate(yva)

    modelo = construir_red(tipo, k)
    parada = tf.keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )
    modelo.fit(Xtr, ytr, validation_data=(Xva, yva),
               epochs=epocas, batch_size=128, verbose=0,
               callbacks=[parada])

    resultados = {}
    for distrito, (s, i_va) in tramos_test.items():
        X, y = ventanas(s, k, i_va, len(s))
        y_hat = modelo.predict(X[..., None], verbose=0).ravel()
        e = esc[distrito]
        resultados[distrito] = (e.invertir(y), e.invertir(y_hat))
    return resultados


# ---------------------------------------------------------------------------
# Modelo 4: híbrido ARIMA + random forest sobre residuos
# ---------------------------------------------------------------------------

def hibrido_arima_rf(panel: dict, res_arima: dict, residuos: dict,
                     k: int = 6) -> dict:
    """Random forest sobre los residuos del ARIMA (Zhao, 2024).

    El RF aprende a predecir el residuo del ARIMA a partir de los k
    residuos previos; la predicción final suma la del ARIMA y la del RF.
    """
    from sklearn.ensemble import RandomForestRegressor

    resultados = {}
    for distrito, serie in panel.items():
        real, pred_arima = res_arima[distrito]
        r = np.asarray(residuos[distrito], float)
        if len(r) <= k + 5:
            resultados[distrito] = (real, pred_arima)
            continue
        Xr, yr = ventanas(r, k, 0, len(r))
        rf = RandomForestRegressor(
            n_estimators=200, random_state=SEED, n_jobs=-1
        ).fit(Xr, yr)
        # En prueba, el residuo previsto se aproxima con los últimos k
        # residuos de entrenamiento actualizados de forma rodante.
        cola = list(r[-k:])
        pred = []
        for i in range(len(pred_arima)):
            r_hat = rf.predict(np.array(cola[-k:])[None, :])[0]
            precio_prev = real[i - 1] if i > 0 else pred_arima[0]
            pred.append(pred_arima[i] + r_hat * precio_prev)
            cola.append(r_hat)
        resultados[distrito] = (real, np.array(pred))
    return resultados


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def cargar_panel() -> dict:
    df = pd.read_csv(PROCESSED / "panel_distritos.csv",
                     parse_dates=["fecha"])
    return {d: g.sort_values("fecha")["precio_m2"].to_numpy(float)
            for d, g in df.groupby("distrito")}


def agregar(resultados: dict) -> dict:
    """Métricas promediadas sobre los 21 distritos."""
    tablas = [resumen(y, y_hat, con_r2=False)
              for y, y_hat in resultados.values()]
    return {m: float(np.mean([t[m] for t in tablas])) for m in tablas[0]}


def main(rapido: bool = False, k: int = VENTANA) -> pd.DataFrame:
    np.random.seed(SEED)
    panel = cargar_panel()
    if rapido:                       # tres distritos y pocas épocas
        panel = {d: panel[d] for d in list(panel)[:3]}
    epocas = 15 if rapido else 150

    filas = {}
    print("ARIMA por distrito...")
    res_arima, residuos = arima_por_distrito(panel)
    filas["ARIMA"] = agregar(res_arima)

    print("LSTM sobre el panel...")
    filas["LSTM"] = agregar(red_sobre_panel(panel, "lstm", k, epocas))

    print("CNN-LSTM sobre el panel...")
    filas["CNN-LSTM"] = agregar(
        red_sobre_panel(panel, "cnn-lstm", k, epocas)
    )

    print("Híbrido ARIMA + RF...")
    filas["ARIMA + RF"] = agregar(
        hibrido_arima_rf(panel, res_arima, residuos)
    )

    tabla = pd.DataFrame(filas).T.round(3)
    print("\nPromedio sobre los distritos evaluados:")
    print(tabla)
    RESULTS.mkdir(exist_ok=True)
    tabla.to_csv(RESULTS / "temporal_metricas.csv")
    return tabla


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--ventana", type=int, default=VENTANA)
    args = ap.parse_args()
    main(rapido=args.rapido, k=args.ventana)
