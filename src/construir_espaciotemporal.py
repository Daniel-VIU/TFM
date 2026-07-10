"""
Construcción del conjunto espacio-temporal (formulación de la Opción A).

Predice el precio de cada inmueble a partir de sus atributos más el
contexto temporal de su distrito. Une las dos fuentes del trabajo:

- idealista18 (inmuebles individuales, 2018): precio y atributos.
- panel mensual por distrito (informes de idealista): nivel y dinámica
  reciente del precio por metro cuadrado del distrito.

El distrito de cada inmueble se obtiene por cruce espacial de sus
coordenadas con los polígonos oficiales de los 21 distritos de Madrid.
A cada inmueble se le añade, según su trimestre de observación, el precio
medio del distrito y sus variaciones a 3 y 12 meses, calculadas sobre el
panel temporal.

Uso:
    python -m src.construir_espaciotemporal
Genera:
    data/processed/espaciotemporal.parquet
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from .preparacion_datos import PROCESSED, RAW

GEOJSON = RAW / "distritos_madrid.geojson"

# Correspondencia entre los nombres del GeoJSON y los del panel temporal.
MAPA_NOMBRES = {
    "Chamartin": "Chamartín", "Chamberi": "Chamberí", "Tetuan": "Tetuán",
    "Vicalvaro": "Vicálvaro", "Fuencarral-El Pardo": "Fuencarral",
    "Moncloa-Aravaca": "Moncloa",
}

# El trimestre de idealista18 (AAAAMM) se asocia al mes del panel.
PERIODO_A_MES = {
    201803: "2018-03-01", 201806: "2018-06-01",
    201809: "2018-09-01", 201812: "2018-12-01",
}


def asignar_distrito(df: pd.DataFrame) -> pd.DataFrame:
    """Añade la columna `distrito` por cruce espacial de coordenadas."""
    distritos = gpd.read_file(GEOJSON)[["name", "geometry"]]
    distritos["name"] = distritos["name"].replace(MAPA_NOMBRES)
    pts = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df.LONGITUDE, df.LATITUDE)],
        crs="EPSG:4326",
    )
    unido = gpd.sjoin(pts, distritos, how="left", predicate="within")
    # sjoin puede duplicar por solapes de frontera: quedarse con uno.
    unido = unido[~unido.index.duplicated(keep="first")]
    df = df.copy()
    df["distrito"] = unido["name"].to_numpy()
    n_sin = df["distrito"].isna().sum()
    print(f"Distrito asignado: {len(df) - n_sin}/{len(df)} "
          f"({n_sin} sin distrito, se descartan)")
    return df.dropna(subset=["distrito"]).reset_index(drop=True)


def contexto_temporal(panel: pd.DataFrame) -> pd.DataFrame:
    """Precio del distrito y sus variaciones a 3 y 12 meses, por mes."""
    panel = panel.sort_values(["distrito", "fecha"]).copy()
    g = panel.groupby("distrito")["precio_m2"]
    panel["ctx_precio_distrito"] = panel["precio_m2"]
    panel["ctx_var_3m"] = g.pct_change(3)
    panel["ctx_var_12m"] = g.pct_change(12)
    return panel[["distrito", "fecha", "ctx_precio_distrito",
                  "ctx_var_3m", "ctx_var_12m"]]


def main() -> None:
    df = pd.read_parquet(PROCESSED / "transversal.parquet")
    df = asignar_distrito(df)

    panel = pd.read_csv(PROCESSED / "panel_distritos.csv",
                        parse_dates=["fecha"])
    ctx = contexto_temporal(panel)

    # Fecha del panel asociada a cada inmueble según su trimestre.
    df["fecha"] = pd.to_datetime(df["PERIOD"].map(PERIODO_A_MES))

    fusion = df.merge(ctx, on=["distrito", "fecha"], how="left")
    faltan = fusion["ctx_precio_distrito"].isna().sum()
    if faltan:
        print(f"Aviso: {faltan} inmuebles sin contexto temporal "
              f"(distrito-mes ausente en el panel)")
    fusion = fusion.dropna(subset=["ctx_precio_distrito"])

    fusion.to_parquet(PROCESSED / "espaciotemporal.parquet", index=False)
    print(f"Conjunto espacio-temporal: {len(fusion)} inmuebles, "
          f"{fusion['distrito'].nunique()} distritos")
    print(f"Guardado en {PROCESSED / 'espaciotemporal.parquet'}")


if __name__ == "__main__":
    main()
