"""Métricas de evaluación definidas en la memoria (cap. Estado del Arte)."""

import numpy as np


def rmse(y, y_hat) -> float:
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def mae(y, y_hat) -> float:
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    return float(np.mean(np.abs(y - y_hat)))


def mape(y, y_hat) -> float:
    """Error porcentual absoluto medio, en tanto por ciento."""
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    return float(100.0 * np.mean(np.abs((y - y_hat) / y)))


def r2(y, y_hat) -> float:
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def resumen(y, y_hat, con_r2: bool = True) -> dict:
    """Diccionario con todas las métricas para una tanda de predicciones."""
    out = {"RMSE": rmse(y, y_hat), "MAE": mae(y, y_hat),
           "MAPE": mape(y, y_hat)}
    if con_r2:
        out["R2"] = r2(y, y_hat)
    return out
