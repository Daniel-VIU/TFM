"""
Preparación del panel Case-Shiller (segundo caso de estudio temporal).

El índice S&P/Case-Shiller recoge el precio de la vivienda de 20 áreas
metropolitanas de EE. UU. con frecuencia mensual desde finales de los
años ochenta. A diferencia del panel de distritos de Madrid, sus series
atraviesan el ciclo de burbuja y crisis de 2006-2012, con caídas
superiores al 50 % en algunas ciudades, lo que proporciona el régimen de
alta volatilidad y no linealidad en el que la literatura sitúa la ventaja
de los modelos de aprendizaje profundo sobre el ARIMA.

Fuente: S&P/Case-Shiller vía FRED, redistribuido por datahub.io con
licencia PDDL (dominio público). El fichero `cities-month-SA.csv` (serie
desestacionalizada) debe descargarse a data/raw/.

El panel resultante tiene el mismo formato que el de Madrid
(serie -> array de valores mensuales), de modo que los mismos modelos de
`modelos_temporal.py` se aplican sin cambios.

Uso:
    python -m src.preparacion_case_shiller
Genera:
    data/processed/panel_case_shiller.csv
"""

from pathlib import Path

import pandas as pd

from .preparacion_datos import PROCESSED, RAW

CSV_CS = RAW / "cities-month-SA.csv"

# Columnas del CSV que no son ciudades individuales (se excluyen del panel).
NO_CIUDADES = {
    "Date",
    "Composite 10-SA", "Composite 20-SA", "National-US-SA",
    "Composite-10-SA", "Composite-20-SA",
}


def cargar_case_shiller(ruta: Path = CSV_CS) -> pd.DataFrame:
    """Convierte el CSV ancho de Case-Shiller a formato largo de panel.

    El CSV tiene una fila por mes y una columna por ciudad. Se transforma
    a formato (serie, fecha, valor) y se descartan los meses sin dato al
    inicio de cada ciudad, de modo que cada serie conserve su ventana.
    """
    df = pd.read_csv(ruta)
    fecha_col = "Date" if "Date" in df.columns else df.columns[0]
    df[fecha_col] = pd.to_datetime(df[fecha_col])

    ciudades = [c for c in df.columns if c not in NO_CIUDADES]
    largo = df.melt(
        id_vars=[fecha_col], value_vars=ciudades,
        var_name="serie", value_name="indice",
    ).rename(columns={fecha_col: "fecha"})

    # Los ceros iniciales de algunas ciudades (huecos de FRED) y los
    # valores ausentes se tratan como falta de dato.
    largo = largo[(largo["indice"].notna()) & (largo["indice"] > 0)]
    largo = largo.sort_values(["serie", "fecha"]).reset_index(drop=True)
    return largo


def main() -> None:
    if not CSV_CS.exists():
        raise FileNotFoundError(
            f"No se encuentra {CSV_CS}. Descarga 'cities-month-SA.csv' de "
            "https://datahub.io/core/house-prices-us y colócalo en data/raw/."
        )
    panel = cargar_case_shiller()
    resumen = panel.groupby("serie")["fecha"].agg(["min", "max", "count"])
    print(f"Panel Case-Shiller: {resumen.shape[0]} ciudades")
    print(f"  meses por ciudad: min={resumen['count'].min()}, "
          f"max={resumen['count'].max()}, "
          f"media={resumen['count'].mean():.0f}")
    print(f"  rango temporal: {panel['fecha'].min():%Y-%m} a "
          f"{panel['fecha'].max():%Y-%m}")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PROCESSED / "panel_case_shiller.csv", index=False)
    print(f"Guardado en {PROCESSED / 'panel_case_shiller.csv'}")


if __name__ == "__main__":
    main()
