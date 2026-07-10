"""
Formulación temporal: predicción del precio medio (eur/m2) por distrito.

Compara cuatro modelos sobre el panel mensual de 21 distritos, en tres
horizontes de predicción (1, 3 y 6 meses):

1. ARIMA por serie (línea base), ajustado sobre la tasa de variación.
2. LSTM entrenada de forma conjunta sobre las ventanas de las 21 series.
3. CNN-LSTM: capas convolucionales 1D antes de las recurrentes.
4. Híbrido ARIMA + random forest sobre los residuos del ARIMA.

Particiones cronológicas 70/20/10 por serie. La normalización
mínimo-máximo de cada serie se ajusta solo con el tramo de
entrenamiento. Las métricas se calculan en la escala original y se
reportan tanto en promedio como por distrito.

Uso:
    python -m src.modelos_temporal [--rapido] [--ventana K]

Genera en results/:
    temporal_metricas.csv          (promedio por modelo y horizonte)
    temporal_por_distrito.csv      (MAPE por distrito, modelo y horizonte 1)
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
VENTANA = 12                 # meses de historia que ve el modelo (lookback)
HORIZONTES = (1, 3, 6)       # meses hacia delante a evaluar
FR_TRAIN, FR_VAL = 0.70, 0.20


# ---------------------------------------------------------------------------
# Utilidades de partición y ventanas
# ---------------------------------------------------------------------------

def particion_cronologica(serie):
    """Índices de corte train/val/test (70/20/10) de una serie."""
    n = len(serie)
    return int(n * FR_TRAIN), int(n * (FR_TRAIN + FR_VAL))


def ventanas(serie, k, ini, fin):
    """Pares (X, y) a un paso, con ventana de k valores en [ini, fin)."""
    X, y = [], []
    for t in range(max(ini, k), fin):
        X.append(serie[t - k:t])
        y.append(serie[t])
    return np.array(X), np.array(y)


class EscaladorMinMax:
    """Mínimo-máximo ajustado solo con el tramo de entrenamiento."""

    def ajustar(self, tramo):
        self.lo, self.hi = float(tramo.min()), float(tramo.max())
        return self

    def transformar(self, x):
        return (x - self.lo) / (self.hi - self.lo)

    def invertir(self, x):
        return x * (self.hi - self.lo) + self.lo


# ---------------------------------------------------------------------------
# Modelo 1: ARIMA por distrito (línea base)
# ---------------------------------------------------------------------------

def arima_por_distrito(panel, horizontes=HORIZONTES, orden=(4, 0, 4)):
    """ARIMA sobre la tasa de variación, con predicción rodante a 1/3/6."""
    from statsmodels.tsa.arima.model import ARIMA

    res = {h: {} for h in horizontes}
    residuos = {}
    hmax = max(horizontes)
    for distrito, serie in panel.items():
        ret = np.diff(serie) / serie[:-1]
        i_tr, i_va = particion_cronologica(serie)
        acum = {h: ([], []) for h in horizontes}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for t in range(i_va, len(serie) - 1):
                mod = ARIMA(ret[:t], order=orden).fit()
                pasos = min(hmax, len(serie) - 1 - t)
                r_hat = np.atleast_1d(mod.forecast(pasos))
                for h in horizontes:
                    if t + h < len(serie) and h <= pasos:
                        nivel = serie[t] * np.prod(1 + r_hat[:h])
                        acum[h][0].append(serie[t + h])
                        acum[h][1].append(nivel)
            mod_tr = ARIMA(ret[: i_tr - 1], order=orden).fit()
            residuos[distrito] = ret[: i_tr - 1] - mod_tr.fittedvalues
        for h in horizontes:
            res[h][distrito] = (np.array(acum[h][0]), np.array(acum[h][1]))
    return res, residuos


# ---------------------------------------------------------------------------
# Modelos 2 y 3: LSTM y CNN-LSTM sobre el panel conjunto
# ---------------------------------------------------------------------------

def construir_red(tipo, k, unidades=20):
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


def _predecir_recursivo(modelo, ventana_ini, k, h):
    """Predicción a h pasos realimentando la propia salida del modelo.

    Usa la llamada directa modelo(x) en lugar de modelo.predict(), mucho
    más rápida para muestras individuales al evitar la sobrecarga por
    lote de Keras.
    """
    import tensorflow as tf

    v = list(ventana_ini)
    for _ in range(h):
        x = tf.constant(np.array(v[-k:])[None, :, None], dtype=tf.float32)
        y = float(modelo(x, training=False).numpy().ravel()[0])
        v.append(y)
    return v[-1]


def red_sobre_panel(panel, tipo, k, horizontes=HORIZONTES, epocas=150):
    """Entrena una red a un paso y evalúa a 1/3/6 por predicción recursiva."""
    import tensorflow as tf

    Xtr, ytr, Xva, yva = [], [], [], []
    esc, tramos = {}, {}
    for distrito, serie in panel.items():
        i_tr, i_va = particion_cronologica(serie)
        e = EscaladorMinMax().ajustar(serie[:i_tr])
        s = e.transformar(serie)
        esc[distrito] = e
        X, y = ventanas(s, k, 0, i_tr); Xtr.append(X); ytr.append(y)
        X, y = ventanas(s, k, i_tr, i_va); Xva.append(X); yva.append(y)
        tramos[distrito] = (s, i_va)
    Xtr = np.concatenate(Xtr)[..., None]; ytr = np.concatenate(ytr)
    Xva = np.concatenate(Xva)[..., None]; yva = np.concatenate(yva)

    modelo = construir_red(tipo, k)
    parada = tf.keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True)
    modelo.fit(Xtr, ytr, validation_data=(Xva, yva),
               epochs=epocas, batch_size=128, verbose=0, callbacks=[parada])

    res = {h: {} for h in horizontes}
    for idx, (distrito, (s, i_va)) in enumerate(tramos.items(), 1):
        e = esc[distrito]
        print(f"    prediccion {tipo} {idx}/{len(tramos)}: {distrito}",
              flush=True)
        for h in horizontes:
            if h == 1:
                # Horizonte 1: predicción directa, todas las ventanas del
                # tramo de prueba en una sola llamada por lotes.
                X, y = ventanas(s, k, i_va, len(s))
                y_hat = modelo.predict(X[..., None], verbose=0).ravel()
                res[h][distrito] = (e.invertir(y), e.invertir(y_hat))
            else:
                # Horizontes mayores: predicción recursiva ventana a ventana.
                reales, preds = [], []
                for t in range(i_va, len(s) - h + 1):
                    if t - k < 0:
                        continue
                    preds.append(_predecir_recursivo(modelo, s[t - k:t], k, h))
                    reales.append(s[t + h - 1])
                res[h][distrito] = (e.invertir(np.array(reales)),
                                    e.invertir(np.array(preds)))
    return res


# ---------------------------------------------------------------------------
# Modelo 4: híbrido ARIMA + random forest sobre residuos
# ---------------------------------------------------------------------------

def hibrido_arima_rf(res_arima, residuos, horizontes=HORIZONTES, k=6):
    """Random forest sobre los residuos del ARIMA (Zhao, 2024)."""
    from sklearn.ensemble import RandomForestRegressor

    res = {h: {} for h in horizontes}
    for distrito in residuos:
        r = np.asarray(residuos[distrito], float)
        rf = None
        if len(r) > k + 5:
            Xr, yr = ventanas(r, k, 0, len(r))
            rf = RandomForestRegressor(
                n_estimators=200, random_state=SEED, n_jobs=-1).fit(Xr, yr)
        for h in horizontes:
            real, pred_a = res_arima[h][distrito]
            if rf is None or len(pred_a) == 0:
                res[h][distrito] = (real, pred_a)
                continue
            cola = list(r[-k:])
            pred = []
            for i in range(len(pred_a)):
                r_hat = rf.predict(np.array(cola[-k:])[None, :])[0]
                nivel = real[i - 1] if i > 0 else pred_a[0]
                pred.append(pred_a[i] + r_hat * nivel)
                cola.append(r_hat)
            res[h][distrito] = (real, np.array(pred))
    return res


# ---------------------------------------------------------------------------
# Orquestación y agregación
# ---------------------------------------------------------------------------

def cargar_panel(dataset="madrid"):
    """Carga el panel elegido como diccionario {serie: array de valores}.

    dataset="madrid": panel de precio (eur/m2) por distrito.
    dataset="case_shiller": índice mensual por ciudad de EE. UU.
    """
    if dataset == "madrid":
        df = pd.read_csv(PROCESSED / "panel_distritos.csv",
                         parse_dates=["fecha"])
        col_serie, col_valor = "distrito", "precio_m2"
    elif dataset == "case_shiller":
        df = pd.read_csv(PROCESSED / "panel_case_shiller.csv",
                         parse_dates=["fecha"])
        col_serie, col_valor = "serie", "indice"
    else:
        raise ValueError(f"dataset desconocido: {dataset}")
    return {s: g.sort_values("fecha")[col_valor].to_numpy(float)
            for s, g in df.groupby(col_serie)}


def agregar(res_h):
    tablas = [resumen(y, yh, con_r2=False)
              for y, yh in res_h.values() if len(y)]
    return {m: float(np.mean([t[m] for t in tablas])) for m in tablas[0]}


def mape_por_distrito(res_h):
    from .metricas import mape
    return {d: mape(y, yh) for d, (y, yh) in res_h.items() if len(y)}


def main(rapido=False, k=VENTANA, dataset="madrid"):
    np.random.seed(SEED)
    panel = cargar_panel(dataset)
    if rapido:
        panel = {d: panel[d] for d in list(panel)[:3]}
    epocas = 15 if rapido else 150
    print(f"Dataset: {dataset}  |  {len(panel)} series")

    print("ARIMA por serie...")
    res_arima, residuos = arima_por_distrito(panel)
    print("LSTM sobre el panel...")
    res_lstm = red_sobre_panel(panel, "lstm", k, epocas=epocas)
    print("CNN-LSTM sobre el panel...")
    res_cnn = red_sobre_panel(panel, "cnn-lstm", k, epocas=epocas)
    print("Híbrido ARIMA + RF...")
    res_hib = hibrido_arima_rf(res_arima, residuos)

    modelos = {"ARIMA": res_arima, "LSTM": res_lstm,
               "CNN-LSTM": res_cnn, "ARIMA + RF": res_hib}

    filas = []
    for nombre, res in modelos.items():
        for h in HORIZONTES:
            filas.append({"Modelo": nombre, "Horizonte": h, **agregar(res[h])})
    tabla = pd.DataFrame(filas).set_index(["Modelo", "Horizonte"]).round(3)
    print("\nPromedio sobre las series:")
    print(tabla)

    por_serie = pd.DataFrame({
        nombre: mape_por_distrito(res[1]) for nombre, res in modelos.items()
    }).round(3)
    por_serie.index.name = "Serie"
    por_serie = por_serie.sort_values("ARIMA")

    RESULTS.mkdir(exist_ok=True)
    sufijo = "" if dataset == "madrid" else f"_{dataset}"
    tabla.to_csv(RESULTS / f"temporal_metricas{sufijo}.csv")
    por_serie.to_csv(RESULTS / f"temporal_por_serie{sufijo}.csv")
    print(f"\nMétricas por serie guardadas ({len(por_serie)} series).")
    return tabla


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--ventana", type=int, default=VENTANA)
    ap.add_argument("--dataset", choices=["madrid", "case_shiller"],
                    default="madrid")
    args = ap.parse_args()
    main(rapido=args.rapido, k=args.ventana, dataset=args.dataset)
