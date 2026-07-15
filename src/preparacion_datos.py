"""
Preparación de los datos del TFM.

Implementa la sección "Preparación de los datos" de la memoria:

1. Conjunto transversal (idealista18, Madrid):
   - Lectura del CSV corrigiendo sus dos defectos de formato:
     a) columna índice inicial sin nombre en la cabecera,
     b) columna final `geometry` con una coma interior sin entrecomillar.
   - Depuración: colapso de duplicados por media, descarte del año de
     construcción declarado (se conserva el catastral, completo y
     redundante con él), filtrado de atípicos extremos.

2. Panel temporal (informes de precios de idealista por distrito):
   - Lectura del libro de cálculo (una hoja por distrito).
   - Conversión de los precios en texto ("6.389 €/m2") a numérico.
   - Eliminación de los meses sin dato ("n.d.").

Uso:
    python -m src.preparacion_datos
Genera:
    data/processed/transversal.parquet
    data/processed/panel_distritos.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

CSV_TRANSVERSAL = RAW / "madrid_sale.csv"
XLSX_PANEL = RAW / "Evolución_Precio_Idealista_Madrid.xlsx"

# Percentiles usados para el cribado de valores extremos de precio
# y superficie (se eliminan las colas por debajo/encima de estos cortes).
P_INF, P_SUP = 0.005, 0.995

# Variables binarias de equipamiento del anuncio.
BINARIAS = [
    "HASTERRACE", "HASLIFT", "HASAIRCONDITIONING", "HASPARKINGSPACE",
    "HASNORTHORIENTATION", "HASSOUTHORIENTATION", "HASEASTORIENTATION",
    "HASWESTORIENTATION", "HASBOXROOM", "HASWARDROBE", "HASSWIMMINGPOOL",
    "HASDOORMAN", "HASGARDEN", "ISDUPLEX", "ISSTUDIO", "ISINTOPFLOOR",
]

# Predictores continuos / discretos.
CONTINUAS = [
    "CONSTRUCTEDAREA", "ROOMNUMBER", "BATHNUMBER",
    "FLOORCLEAN", "CADCONSTRUCTIONYEAR", "CADMAXBUILDINGFLOOR",
    "CADDWELLINGCOUNT", "CADASTRALQUALITYID",
    "DISTANCE_TO_CITY_CENTER", "DISTANCE_TO_METRO", "DISTANCE_TO_CASTELLANA",
    "LONGITUDE", "LATITUDE",
]

RESPUESTA = "PRICE"


def cargar_transversal(ruta: Path = CSV_TRANSVERSAL) -> pd.DataFrame:
    """Lee el CSV de idealista18 corrigiendo sus defectos de formato.

    La cabecera contiene una primera columna vacía (índice de fila
    exportado desde R) y la última columna, `geometry`, incluye una coma
    interior sin comillas que desplaza el parseo automático. La lectura
    fija la primera columna como índice y descarta `geometry`, redundante
    con LONGITUDE/LATITUDE.
    """
    import csv

    with open(ruta, newline="") as f:
        lector = csv.reader(f)
        cabecera = next(lector)          # ['', 'ASSETID', ..., 'geometry']
        nombres = cabecera[1:-1]         # sin índice ni geometry
        filas = []
        for fila in lector:
            # fila = [idx, ASSETID..LATITUDE (41 valores), geom_a, geom_b]
            filas.append(fila[1:42])
    df = pd.DataFrame(filas, columns=nombres)

    # Conversión de tipos: todo salvo el identificador es numérico.
    for col in df.columns:
        if col != "ASSETID":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["PERIOD"] = df["PERIOD"].astype(int)
    return df


def depurar_transversal(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la depuración descrita en la memoria."""
    n0 = len(df)

    # El mismo anuncio puede aparecer varias veces dentro de un trimestre
    # con precios distintos (revisiones de precio del vendedor). El
    # fichero no incluye marca temporal intra-trimestral, por lo que la
    # antigüedad relativa de las apariciones no es observable con
    # garantías. Se colapsa cada par (anuncio, periodo) en un único
    # registro tomando la MEDIA de sus apariciones, un criterio
    # independiente del orden del fichero que utiliza toda la
    # información de las revisiones (véase la sección "Registros
    # duplicados" del análisis exploratorio de la memoria). Tras la
    # media, las variables discretas se redondean al entero más próximo
    # y las binarias se resuelven por mayoría (media >= 0,5 -> 1); el
    # precio unitario se recalcula para mantener la coherencia interna.
    DISCRETAS = [
        "ROOMNUMBER", "BATHNUMBER", "FLOORCLEAN",
        "CADCONSTRUCTIONYEAR", "CADMAXBUILDINGFLOOR", "CADDWELLINGCOUNT",
        "CADASTRALQUALITYID", "FLATLOCATIONID", "AMENITYID",
        "ISPARKINGSPACEINCLUDEDINPRICE", "BUILTTYPEID_1", "BUILTTYPEID_2",
        "BUILTTYPEID_3",
    ]
    df = df.groupby(["ASSETID", "PERIOD"], as_index=False).mean()
    for col in DISCRETAS:
        df[col] = np.floor(df[col] + 0.5)          # redondeo al entero
    for col in BINARIAS:
        df[col] = (df[col] >= 0.5).astype(float)    # voto por mayoría
    df["UNITPRICE"] = df["PRICE"] / df["CONSTRUCTEDAREA"]

    # Año de construcción: el análisis exploratorio muestra que la
    # variable declarada en el anuncio tiene un 59,8 % de ausencias tras
    # el colapso y errores manifiestos entre los valores presentes (años
    # del 1 al 2291), mientras que el registro catastral está completo,
    # no contiene valores imposibles (rango 1623-2018) y coincide con el
    # declarado en el 98,8 % de los registros en que ambos existen. Se
    # descarta por tanto la variable declarada y se conserva la catastral
    # (CADCONSTRUCTIONYEAR) como único año de construcción.
    df = df.drop(columns=["CONSTRUCTIONYEAR"])

    # Cribado de colas extremas en precio y superficie.
    for col in [RESPUESTA, "CONSTRUCTEDAREA"]:
        lo, hi = df[col].quantile([P_INF, P_SUP])
        df = df[(df[col] >= lo) & (df[col] <= hi)]

    # Binarias a entero 0/1.
    for col in BINARIAS:
        df[col] = df[col].fillna(0).astype(int)

    # ------------------------------------------------------------------
    # Filtros adicionales motivados por el análisis exploratorio
    # (sección "Análisis exploratorio de los datos" de la memoria).
    # ------------------------------------------------------------------

    # a) Errores de geolocalización: anuncios situados fuera del término
    #    municipal de Madrid (el EDA detectó un anuncio geocodificado en
    #    la provincia de Almería, a 416 km del centro, que distorsionaba
    #    las tres variables de distancia).
    dentro = df["LATITUDE"].between(40.30, 40.66) & df["LONGITUDE"].between(
        -3.90, -3.50
    )
    df = df[dentro]

    # b) Incoherencias físicas: viviendas sin cuarto de baño (no
    #    habitables según el art. 7.3.4 del PGOUM), número de
    #    habitaciones imposible (>15, p. ej. 93 habitaciones en 119 m2)
    #    o menos de 8 m2 construidos por habitación.
    incoherente = (
        (df["BATHNUMBER"] == 0)
        | (df["ROOMNUMBER"] > 15)
        | (
            (df["ROOMNUMBER"] > 0)
            & (df["CONSTRUCTEDAREA"] / df["ROOMNUMBER"] < 8)
        )
    )
    df = df[~incoherente]

    # c) Nulos residuales en dos predictores (planta y calidad
    #    catastral): eliminación por lista. Este paso se realizaba antes
    #    de forma implícita en la fase de modelización; se traslada aquí
    #    para que el conjunto depurado sea directamente el de trabajo.
    df = df.dropna(subset=["FLOORCLEAN", "CADASTRALQUALITYID"])

    df = df.reset_index(drop=True)
    print(f"Transversal: {n0} -> {len(df)} registros tras depuración")
    return df


def cargar_panel(ruta: Path = XLSX_PANEL) -> pd.DataFrame:
    """Construye el panel distrito-mes de precio medio (eur/m2).

    Cada hoja del libro corresponde a un distrito. Los precios vienen
    como texto con formato español ("6.389 €/m2") dentro de fórmulas
    cacheadas de Google Sheets; pandas recupera el valor cacheado y
    aquí se convierte a numérico. Los meses sin dato ("n.d.") se
    eliminan, de modo que cada distrito conserva su propia ventana.
    """
    xl = pd.ExcelFile(ruta)
    series = []
    for distrito in xl.sheet_names:
        d = pd.read_excel(xl, distrito, header=0)
        d = d.rename(columns={"Mes": "fecha", "Precio m2": "texto"})
        precio = (
            d["texto"].astype(str)
            .str.replace(".", "", regex=False)   # separador de millares
            .str.extract(r"(\d+)")[0]
        )
        d["precio_m2"] = pd.to_numeric(precio, errors="coerce")
        d["fecha"] = pd.to_datetime(d["fecha"])
        d["distrito"] = distrito
        d = d.dropna(subset=["precio_m2"])
        series.append(d[["distrito", "fecha", "precio_m2"]])
    panel = (
        pd.concat(series)
        .sort_values(["distrito", "fecha"])
        .reset_index(drop=True)
    )
    resumen = panel.groupby("distrito")["fecha"].agg(["min", "max", "count"])
    print("Panel temporal: 21 distritos" if resumen.shape[0] == 21
          else f"Panel temporal: {resumen.shape[0]} distritos (¡revisar!)")
    print(f"  meses por distrito: min={resumen['count'].min()}, "
          f"max={resumen['count'].max()}, "
          f"media={resumen['count'].mean():.0f}")
    return panel


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    df = depurar_transversal(cargar_transversal())
    df.to_parquet(PROCESSED / "transversal.parquet", index=False)

    panel = cargar_panel()
    panel.to_csv(PROCESSED / "panel_distritos.csv", index=False)

    print(f"\nFicheros generados en {PROCESSED}")


if __name__ == "__main__":
    main()
