"""
Formulación temporal: predicción del precio medio (eur/m2) por distrito.

Compara tres enfoques sobre el panel mensual de distritos, en varios
horizontes de predicción (1, 3, 6 y 12 meses):

1. ARIMA(1,0,4) por serie (línea base), ajustado sobre log-retornos.
   El orden se selecciona por AIC/BIC en rejilla (ver
   comparar_ordenes_arima.py y el capítulo de desarrollo de la memoria).
2. LSTM entrenada de forma conjunta sobre las ventanas de las series.
3. Híbrido ARIMA + random forest sobre los residuos del ARIMA.

Particiones cronológicas 70/20/10 por serie. La normalización
mínimo-máximo de cada serie se ajusta solo con el tramo de
entrenamiento. Las métricas se calculan en la escala original y se
reportan tanto en promedio como por serie.

Uso:
    python -m src.modelos_temporal [--rapido] [--ventana K]

Genera en results/:
    temporal_metricas.csv          (promedio por modelo y horizonte)
    temporal_por_distrito.csv      (MAPE por distrito, modelo, horizonte 1)
    ljung_box_distritos.csv        (contraste de Ljung-Box por distrito)
"""

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .metricas import resumen
from .preparacion_datos import PROCESSED

SEED = 2025
RESULTS = Path(__file__).resolve().parents[1] / "results"
VENTANA = 12
HORIZONTES = (1, 3, 6, 12)
FR_TRAIN, FR_VAL = 0.70, 0.20  # test = 10%
EPS = 1e-9


# ---------------------------------------------------------------------------
# Utilidades de partición y ventanas
# ---------------------------------------------------------------------------

def particion_cronologica(serie):
    """Índices de corte train/val/test (70/20/10) de una serie."""
    n = len(serie)
    i_tr = int(n * FR_TRAIN)
    i_va = int(n * (FR_TRAIN + FR_VAL))
    return i_tr, i_va


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
        tramo = np.asarray(tramo, dtype=float)
        self.lo = float(np.nanmin(tramo))
        self.hi = float(np.nanmax(tramo))
        self.rango = self.hi - self.lo
        if not np.isfinite(self.rango) or self.rango == 0:
            self.rango = 1.0
        return self

    def transformar(self, x):
        x = np.asarray(x, dtype=float)
        return (x - self.lo) / self.rango

    def invertir(self, x):
        x = np.asarray(x, dtype=float)
        return x * self.rango + self.lo


# ---------------------------------------------------------------------------
# Modelo 1: ARIMA por serie (línea base)
# ---------------------------------------------------------------------------

def _log_retornos(serie):
    serie = np.asarray(serie, dtype=float)
    serie = np.clip(serie, EPS, None)
    return np.diff(np.log(serie))


def arima_por_distrito(panel, horizontes=HORIZONTES, orden=(1, 0, 4)):
    """ARIMA sobre log-retornos, con predicción rodante a varios horizontes."""
    from statsmodels.tsa.arima.model import ARIMA

    res = {h: {} for h in horizontes}
    residuos = {}
    hmax = max(horizontes)

    for distrito, serie in panel.items():
        serie = np.asarray(serie, dtype=float)
        if len(serie) < max(VENTANA + 2, hmax + 5):
            for h in horizontes:
                res[h][distrito] = (np.array([]), np.array([]))
            residuos[distrito] = np.array([])
            continue

        ret = _log_retornos(serie)
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
                        # Reconstrucción desde el precio actual usando la suma
                        # acumulada de los retornos predichos.
                        nivel = float(serie[t] * np.exp(np.sum(r_hat[:h])))
                        acum[h][0].append(float(serie[t + h]))
                        acum[h][1].append(nivel)

            mod_tr = ARIMA(ret[: max(i_tr - 1, 1)], order=orden).fit()
            residuos[distrito] = ret[: max(i_tr - 1, 1)] - mod_tr.fittedvalues

        for h in horizontes:
            res[h][distrito] = (np.array(acum[h][0]), np.array(acum[h][1]))

    return res, residuos


# ---------------------------------------------------------------------------
# Modelos 2 y 3: LSTM y CNN-LSTM sobre el panel conjunto
# ---------------------------------------------------------------------------

def construir_red(tipo, k, unidades=24):
    """Arquitecturas recurrentes con regularización ligera."""
    import tensorflow as tf
    from tensorflow.keras import layers

    tf.random.set_seed(SEED)
    entrada = layers.Input(shape=(k, 1))
    x = entrada

    if tipo == "cnn-lstm":
        x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
        x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(16, 3, activation="relu", padding="same")(x)
        x = layers.Dropout(0.20)(x)

    x = layers.LSTM(unidades, return_sequences=True)(x)
    x = layers.LSTM(max(unidades // 2, 8))(x)
    x = layers.Dropout(0.20)(x)
    salida = layers.Dense(1)(x)

    modelo = tf.keras.Model(entrada, salida)
    modelo.compile(optimizer="adam", loss="mae", metrics=["mae"])
    return modelo


def _predecir_recursivo_lote(modelo, ventanas_ini, k, h):
    """Predicción recursiva a h pasos para un lote de ventanas a la vez."""
    import tensorflow as tf

    v = np.asarray(ventanas_ini, dtype=np.float32)
    for _ in range(h):
        x = tf.constant(v[:, -k:, None], dtype=tf.float32)
        y = modelo(x, training=False).numpy().reshape(-1, 1)
        v = np.concatenate([v, y], axis=1)
    return v[:, -1]


def red_sobre_panel(panel, tipo, k, horizontes=HORIZONTES, epocas=150):
    """Entrena una red a un paso y evalúa a varios horizontes por recursión."""
    import tensorflow as tf

    Xtr, ytr, Xva, yva = [], [], [], []
    esc, tramos = {}, {}

    for distrito, serie in panel.items():
        serie = np.asarray(serie, dtype=float)
        i_tr, i_va = particion_cronologica(serie)
        e = EscaladorMinMax().ajustar(serie[:i_tr])
        s = e.transformar(serie)
        esc[distrito] = e

        X, y = ventanas(s, k, 0, i_tr)
        if len(X):
            Xtr.append(X)
            ytr.append(y)

        X, y = ventanas(s, k, i_tr, i_va)
        if len(X):
            Xva.append(X)
            yva.append(y)

        tramos[distrito] = (s, i_va)

    if not Xtr or not Xva:
        raise ValueError("No hay suficientes ventanas para entrenar/validar.")

    Xtr = np.concatenate(Xtr)[..., None]
    ytr = np.concatenate(ytr)
    Xva = np.concatenate(Xva)[..., None]
    yva = np.concatenate(yva)

    modelo = construir_red(tipo, k)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, monitor="val_loss"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=5, factor=0.5, min_lr=1e-5, monitor="val_loss"
        ),
    ]

    modelo.fit(
        Xtr,
        ytr,
        validation_data=(Xva, yva),
        epochs=epocas,
        batch_size=128,
        verbose=0,
        shuffle=True,
        callbacks=callbacks,
    )

    res = {h: {} for h in horizontes}
    for idx, (distrito, (s, i_va)) in enumerate(tramos.items(), 1):
        e = esc[distrito]
        print(f"    prediccion {tipo} {idx}/{len(tramos)}: {distrito}", flush=True)

        for h in horizontes:
            if h == 1:
                X, y = ventanas(s, k, i_va, len(s))
                if len(X):
                    y_hat = modelo.predict(X[..., None], verbose=0).ravel()
                    res[h][distrito] = (e.invertir(y), e.invertir(y_hat))
                else:
                    res[h][distrito] = (np.array([]), np.array([]))
            else:
                inis, reales = [], []
                for t in range(i_va, len(s) - h + 1):
                    if t - k < 0:
                        continue
                    inis.append(s[t - k:t])
                    reales.append(s[t + h - 1])
                if inis:
                    preds = _predecir_recursivo_lote(modelo, np.array(inis), k, h)
                    res[h][distrito] = (e.invertir(np.array(reales)), e.invertir(preds))
                else:
                    res[h][distrito] = (np.array([]), np.array([]))

    return res


# ---------------------------------------------------------------------------
# Modelo 4: híbrido ARIMA + random forest sobre residuos
# ---------------------------------------------------------------------------

def hibrido_arima_rf(res_arima, residuos, horizontes=HORIZONTES, k=6):
    """Random forest sobre residuos ARIMA sin fuga de información."""
    from sklearn.ensemble import RandomForestRegressor

    res = {h: {} for h in horizontes}

    for distrito in residuos:
        r = np.asarray(residuos[distrito], dtype=float)
        if len(r) <= k + 5:
            for h in horizontes:
                res[h][distrito] = res_arima[h].get(distrito, (np.array([]), np.array([])))
            continue

        Xr, yr = ventanas(r, k, 0, len(r))
        if len(Xr) == 0:
            for h in horizontes:
                res[h][distrito] = res_arima[h].get(distrito, (np.array([]), np.array([])))
            continue

        rf = RandomForestRegressor(
            n_estimators=300,
            random_state=SEED,
            n_jobs=-1,
            min_samples_leaf=2,
        ).fit(Xr, yr)

        for h in horizontes:
            real, pred_a = res_arima[h][distrito]
            if len(real) == 0 or len(pred_a) == 0:
                res[h][distrito] = (real, pred_a)
                continue

            # El RF predice el residuo (en log-retorno) de cada punto de
            # test a partir de los k residuos previos, de forma rodante:
            # tras cada punto se incorpora el residuo real observado, no
            # la predicción, para no acumular error. La corrección se
            # aplica a cada punto de forma independiente (sin cumsum),
            # escalada por el nivel de precio del ARIMA en ese punto.
            cola = list(r[-k:])
            correccion = np.empty(len(pred_a))
            for i in range(len(pred_a)):
                r_hat = float(rf.predict(np.array(cola[-k:])[None, :])[0])
                correccion[i] = r_hat
                # Residuo real aproximado: log-retorno realizado menos el
                # previsto por el ARIMA entre el punto anterior y este.
                if i > 0:
                    ret_real = np.log(max(real[i], EPS) /
                                      max(real[i - 1], EPS))
                    cola.append(ret_real)
                else:
                    cola.append(r_hat)
            # Corrección multiplicativa punto a punto (sin acumulación).
            pred = pred_a * np.exp(correccion)
            res[h][distrito] = (real, pred)

    return res


# ---------------------------------------------------------------------------
# Diagnóstico: contraste de Ljung-Box sobre los residuos del ARIMA
# ---------------------------------------------------------------------------

def tabla_ljung_box(residuos, retardos=12, alfa=0.05):
    """Contraste de Ljung-Box por distrito sobre los residuos ARIMA.

    Recibe el diccionario de residuos de entrenamiento que devuelve
    ``arima_por_distrito`` (los mismos sobre los que se entrena el
    corrector del híbrido) y evalúa el estadístico Q acumulado hasta
    ``retardos`` (12 por defecto: un año completo en datos mensuales).
    Devuelve un DataFrame con el estadístico, el p-valor y si la serie
    es compatible con ruido blanco al nivel ``alfa``.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    filas = []
    for distrito, r in residuos.items():
        r = np.asarray(r, dtype=float)
        if len(r) <= retardos + 1:
            continue
        lb = acorr_ljungbox(r, lags=[retardos])
        q = float(lb["lb_stat"].iloc[0])
        p = float(lb["lb_pvalue"].iloc[0])
        filas.append({"Distrito": distrito, "n": len(r),
                      f"Q({retardos})": round(q, 2),
                      "p-valor": round(p, 3),
                      "Ruido blanco": p > alfa})

    tabla = pd.DataFrame(filas).sort_values("Distrito").reset_index(drop=True)
    return tabla


# ---------------------------------------------------------------------------
# Orquestación y agregación
# ---------------------------------------------------------------------------

def cargar_panel():
    """Panel de distritos de Madrid como diccionario {distrito: valores}."""
    df = pd.read_csv(PROCESSED / "panel_distritos.csv", parse_dates=["fecha"])
    panel = {}
    for s, g in df.groupby("distrito"):
        panel[s] = g.sort_values("fecha")["precio_m2"].to_numpy(float)
    return panel


def agregar(res_h):
    tablas = [resumen(y, yh, con_r2=False) for y, yh in res_h.values() if len(y)]
    if not tablas:
        return {"RMSE": float("nan"), "MAE": float("nan"), "MAPE": float("nan")}
    return {m: float(np.mean([t[m] for t in tablas])) for m in tablas[0]}


def mape_por_distrito(res_h):
    from .metricas import mape
    return {d: mape(y, yh) for d, (y, yh) in res_h.items() if len(y)}


def main(rapido=False, k=VENTANA):
    np.random.seed(SEED)
    panel = cargar_panel()
    if rapido:
        panel = {d: panel[d] for d in list(panel)[:3]}
    epocas = 15 if rapido else 150

    print(f"{len(panel)} distritos")

    print("ARIMA por serie...")
    res_arima, residuos = arima_por_distrito(panel)

    print("Contraste de Ljung-Box sobre los residuos...")
    lb = tabla_ljung_box(residuos)
    n_rb = int(lb["Ruido blanco"].sum())
    print(f"  Ruido blanco (p > 0.05): {n_rb} de {len(lb)} distritos")
    RESULTS.mkdir(exist_ok=True)
    lb.to_csv(RESULTS / "ljung_box_distritos.csv", index=False)

    print("LSTM sobre el panel...")
    res_lstm = red_sobre_panel(panel, "lstm", k, epocas=epocas)
    print("Híbrido ARIMA + RF...")
    res_hib = hibrido_arima_rf(res_arima, residuos)

    modelos = {
        "ARIMA": res_arima,
        "LSTM": res_lstm,
        "ARIMA + RF": res_hib,
    }

    filas = []
    for nombre, res in modelos.items():
        for h in HORIZONTES:
            filas.append({"Modelo": nombre, "Horizonte": h, **agregar(res[h])})

    tabla = pd.DataFrame(filas).set_index(["Modelo", "Horizonte"]).round(3)
    print("\nPromedio sobre los distritos:")
    print(tabla)

    por_dist = pd.DataFrame({nombre: mape_por_distrito(res[1])
                             for nombre, res in modelos.items()}).round(3)
    por_dist.index.name = "Distrito"
    por_dist = por_dist.sort_values("ARIMA")

    RESULTS.mkdir(exist_ok=True)
    tabla.to_csv(RESULTS / "temporal_metricas.csv")
    por_dist.to_csv(RESULTS / "temporal_por_distrito.csv")
    print(f"\nMétricas por distrito guardadas ({len(por_dist)} distritos).")
    return tabla


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--ventana", type=int, default=VENTANA)
    args = ap.parse_args()
    main(rapido=args.rapido, k=args.ventana)
