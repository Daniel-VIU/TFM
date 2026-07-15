"""
Figuras del análisis exploratorio de datos (sección "Análisis
exploratorio de los datos" de la memoria).

Genera las seis figuras de la sección en formato PDF vectorial:

    fig_eda_precio.pdf       distribución del precio y de su logaritmo
    fig_eda_boxplots.pdf     precio por nº de habitaciones y superficie
    fig_eda_correlacion.pdf  matriz de correlaciones de Pearson
    fig_eda_dispersion.pdf   log(precio) frente a cuatro predictores
    fig_eda_mapa.pdf         mapa de anuncios por precio unitario
    fig_eda_panel.pdf        series de distritos y variación intermensual
    fig_mape_horizonte.pdf   MAPE medio por modelo y horizonte (cap. 5)

Las cinco figuras transversales se calculan sobre el conjunto de 74.332
anuncios resultante del colapso por media de duplicados, el descarte
del año de construcción declarado y el recorte de colas, es decir, ANTES de los
filtros adicionales motivados por el propio EDA (que son lo que estas
figuras documentan). Como el pipeline solo persiste el conjunto
definitivo, este módulo reconstruye ese conjunto intermedio desde el
CSV bruto replicando los primeros pasos de
``preparacion_datos.depurar_transversal``. La figura del panel se lee
del CSV procesado.

Uso:
    python -m src.graficas_eda
Genera los PDF en la carpeta FIGURAS (ajustar a la ruta de imágenes de
la memoria, p. ej. ``memoria/Images``).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from .preparacion_datos import (BINARIAS, P_INF, P_SUP, PROCESSED,
                                cargar_transversal)

# Carpeta de salida: apuntar a la carpeta Images del proyecto LaTeX.
FIGURAS = Path(__file__).resolve().parents[1] / "memoria" / "Images"

# Carpeta de resultados generada por modelos_temporal.py.
RESULTS = Path(__file__).resolve().parents[1] / "results"

# ---------------------------------------------------------------------
# Estilo común: sobrio, tipografía con serifa (coherente con la memoria)
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
AZUL, NARANJA, GRIS = "#2b5d8c", "#c96a2b", "#9aa0a6"

# Separador de millares español en los ejes (40.000 en lugar de 40,000).
MILES = FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", "."))

CAJA = dict(
    flierprops=dict(marker=".", ms=2, mec=GRIS, alpha=0.35),
    medianprops=dict(color=NARANJA),
    patch_artist=True,
    boxprops=dict(facecolor="#dbe6f0", color=AZUL),
    whiskerprops=dict(color=AZUL),
    capprops=dict(color=AZUL),
)


def conjunto_basico() -> pd.DataFrame:
    """Reconstruye el conjunto intermedio del EDA (74.332 anuncios).

    Replica los pasos 2 y 3 de la tabla de depuración de la memoria
    (colapso por media de los pares repetidos, descarte del año de
    construcción declarado y recorte de colas), sin los filtros posteriores
    motivados por el EDA, que son precisamente lo que estas figuras
    documentan.
    """
    df = cargar_transversal()

    # Colapso por media de los pares (anuncio, periodo) repetidos.
    discretas = [
        "ROOMNUMBER", "BATHNUMBER", "FLOORCLEAN",
        "CADCONSTRUCTIONYEAR", "CADMAXBUILDINGFLOOR", "CADDWELLINGCOUNT",
        "CADASTRALQUALITYID", "FLATLOCATIONID", "AMENITYID",
        "ISPARKINGSPACEINCLUDEDINPRICE", "BUILTTYPEID_1", "BUILTTYPEID_2",
        "BUILTTYPEID_3",
    ]
    df = df.groupby(["ASSETID", "PERIOD"], as_index=False).mean()
    for col in discretas:
        df[col] = np.floor(df[col] + 0.5)
    for col in BINARIAS:
        df[col] = (df[col] >= 0.5).astype(float)
    df["UNITPRICE"] = df["PRICE"] / df["CONSTRUCTEDAREA"]

    # Año de construcción: se descarta la variable declarada (redundante
    # con la catastral, que es completa y sin valores imposibles).
    df = df.drop(columns=["CONSTRUCTIONYEAR"])

    # Recorte de colas extremas de precio y superficie.
    for col in ["PRICE", "CONSTRUCTEDAREA"]:
        lo, hi = df[col].quantile([P_INF, P_SUP])
        df = df[(df[col] >= lo) & (df[col] <= hi)]

    for col in BINARIAS:
        df[col] = df[col].fillna(0).astype(int)
    return df.reset_index(drop=True)


def fig_precio(bas: pd.DataFrame) -> None:
    """Histograma del precio de oferta y de su logaritmo."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))

    ax = axes[0]
    ax.hist(bas["PRICE"] / 1000, bins=80, color=AZUL,
            edgecolor="white", lw=0.3)
    mediana, media = bas["PRICE"].median() / 1000, bas["PRICE"].mean() / 1000
    ax.axvline(mediana, color=NARANJA, lw=1.4,
               label=f"Mediana: {mediana:,.0f} mil €".replace(",", "."))
    ax.axvline(media, color=NARANJA, lw=1.4, ls="--",
               label=f"Media: {media:,.0f} mil €".replace(",", "."))
    ax.set_xlabel("Precio (miles de €)")
    ax.set_ylabel("Número de anuncios")
    ax.set_title("(a) Precio de oferta")
    ax.xaxis.set_major_formatter(MILES)
    ax.yaxis.set_major_formatter(MILES)
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]
    ax.hist(np.log(bas["PRICE"]), bins=80, color=AZUL,
            edgecolor="white", lw=0.3)
    ax.set_xlabel("log(precio)")
    ax.set_ylabel("Número de anuncios")
    ax.set_title("(b) Logaritmo del precio")
    ax.yaxis.set_major_formatter(MILES)

    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_eda_precio.pdf", bbox_inches="tight")
    plt.close()


def fig_boxplots(bas: pd.DataFrame) -> None:
    """Precio por número de habitaciones y superficie construida."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))

    ax = axes[0]
    hab = bas["ROOMNUMBER"].clip(upper=6)
    grupos = [bas["PRICE"][hab == k] / 1000 for k in range(0, 7)]
    ax.boxplot(grupos, tick_labels=["0", "1", "2", "3", "4", "5", "6+"],
               **CAJA)
    ax.set_xlabel("Número de habitaciones")
    ax.set_ylabel("Precio (miles de €)")
    ax.set_title("(a) Precio por número de habitaciones")
    ax.yaxis.set_major_formatter(MILES)

    ax = axes[1]
    ax.boxplot([bas["CONSTRUCTEDAREA"]], vert=False, widths=0.5,
               tick_labels=[""], **CAJA)
    ax.set_xlabel("Superficie construida (m$^2$)")
    ax.set_title("(b) Superficie construida")

    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_eda_boxplots.pdf", bbox_inches="tight")
    plt.close()


NOMBRES = {
    "PRICE": "Precio", "CONSTRUCTEDAREA": "Superficie",
    "ROOMNUMBER": "Habitaciones", "BATHNUMBER": "Baños",
    "CADCONSTRUCTIONYEAR": "Año constr.", "FLOORCLEAN": "Planta",
    "CADMAXBUILDINGFLOOR": "Plantas edif.",
    "CADDWELLINGCOUNT": "Viviendas edif.",
    "CADASTRALQUALITYID": "Calidad catastral",
    "DISTANCE_TO_CITY_CENTER": "Dist. centro",
    "DISTANCE_TO_METRO": "Dist. metro",
    "DISTANCE_TO_CASTELLANA": "Dist. Castellana",
    "LONGITUDE": "Longitud", "LATITUDE": "Latitud",
}


def fig_correlacion(bas: pd.DataFrame) -> None:
    """Matriz de correlaciones de Pearson con anotaciones."""
    cm = bas[list(NOMBRES)].corr().rename(index=NOMBRES, columns=NOMBRES)

    fig, ax = plt.subplots(figsize=(8.4, 7))
    im = ax.imshow(cm, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(cm)))
    ax.set_yticks(range(len(cm)))
    ax.set_xticklabels(cm.columns, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(cm.index, fontsize=8.5)
    for i in range(len(cm)):
        for j in range(len(cm)):
            v = cm.iloc[i, j]
            ax.text(j, i, f"{v:.2f}".replace(".", ","), ha="center",
                    va="center", fontsize=6.8,
                    color="white" if abs(v) > 0.55 else "black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8,
                 label="Coeficiente de correlación de Pearson")
    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_eda_correlacion.pdf", bbox_inches="tight")
    plt.close()


def fig_dispersion(bas: pd.DataFrame, n: int = 15_000, semilla: int = 1
                   ) -> None:
    """log(precio) frente a cuatro predictores (muestra aleatoria).

    Los puntos se rasterizan dentro del PDF para que el fichero no pese
    decenas de megabytes; ejes y texto siguen siendo vectoriales.
    """
    m = bas.sample(n, random_state=semilla)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6))

    def scat(ax, x, xlab):
        ax.scatter(x, np.log(m["PRICE"]), s=2, alpha=0.15, color=AZUL,
                   rasterized=True)
        ax.set_xlabel(xlab)
        ax.set_ylabel("log(precio)")

    scat(axes[0, 0], m["CONSTRUCTEDAREA"], "Superficie construida (m$^2$)")
    axes[0, 0].set_title("(a) Superficie")
    scat(axes[0, 1], m["DISTANCE_TO_CASTELLANA"],
         "Distancia a la Castellana (km)")
    axes[0, 1].set_title("(b) Distancia a la Castellana")
    scat(axes[1, 0], m["CADCONSTRUCTIONYEAR"], "Año de construcción")
    axes[1, 0].set_title("(c) Año de construcción")
    axes[1, 0].set_xlim(1850, 2020)
    scat(axes[1, 1], m["CADASTRALQUALITYID"],
         "Calidad catastral (1 = máxima, 9 = mínima)")
    axes[1, 1].set_title("(d) Calidad catastral")

    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_eda_dispersion.pdf", bbox_inches="tight")
    plt.close()


def fig_mapa(bas: pd.DataFrame) -> None:
    """Anuncios sobre el plano, coloreados por precio unitario.

    Se excluye el anuncio geocodificado fuera del municipio (el error de
    Almería documentado en la memoria) y la escala de color se trunca en
    los percentiles 2 y 98 para que el gradiente no quede aplastado por
    los extremos.
    """
    d = bas[bas["LATITUDE"].between(40.30, 40.66)
            & bas["LONGITUDE"].between(-3.90, -3.50)]

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    sc = ax.scatter(d["LONGITUDE"], d["LATITUDE"], c=d["UNITPRICE"],
                    s=2.5, cmap="viridis",
                    vmin=d["UNITPRICE"].quantile(0.02),
                    vmax=d["UNITPRICE"].quantile(0.98),
                    alpha=0.6, rasterized=True)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    # Relación de aspecto correcta para la latitud de Madrid, de modo que
    # el plano no aparezca achatado.
    ax.set_aspect(1 / np.cos(np.deg2rad(40.42)))
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("Precio unitario (€/m$^2$)")
    cb.ax.yaxis.set_major_formatter(MILES)
    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_eda_mapa.pdf", bbox_inches="tight")
    plt.close()


def fig_panel() -> None:
    """Series de los 21 distritos y variación intermensual del precio."""
    panel = pd.read_csv(PROCESSED / "panel_distritos.csv",
                        parse_dates=["fecha"])
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6))

    # (a) series: todas en gris, mediana en negro, extremos destacados.
    ax = axes[0]
    ultimo = (panel.sort_values("fecha")
              .groupby("distrito")["precio_m2"].last().sort_values())
    d_min, d_max = ultimo.index[0], ultimo.index[-1]
    for distrito, g in panel.groupby("distrito"):
        if distrito in (d_min, d_max):
            continue
        ax.plot(g["fecha"], g["precio_m2"], color=GRIS, lw=0.6, alpha=0.5)
    mediana = panel.groupby("fecha")["precio_m2"].median()
    ax.plot(mediana.index, mediana.values, color="black", lw=1.6,
            label="Mediana de distritos")
    g = panel[panel["distrito"] == d_max]
    ax.plot(g["fecha"], g["precio_m2"], color=NARANJA, lw=1.3, label=d_max)
    g = panel[panel["distrito"] == d_min]
    ax.plot(g["fecha"], g["precio_m2"], color=AZUL, lw=1.3, label=d_min)
    ax.set_ylabel("Precio medio (€/m$^2$)")
    ax.set_title("(a) Series de los 21 distritos")
    ax.yaxis.set_major_formatter(MILES)
    ax.legend(frameon=False, fontsize=8)

    # (b) variación intermensual.
    ax = axes[1]
    var = (panel.sort_values(["distrito", "fecha"])
           .groupby("distrito")["precio_m2"].pct_change().dropna() * 100)
    ax.hist(var, bins=90, range=(-6, 6), color=AZUL,
            edgecolor="white", lw=0.3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Variación intermensual (%)")
    ax.set_ylabel("Número de observaciones")
    ax.set_title("(b) Variación intermensual del precio")
    ax.yaxis.set_major_formatter(MILES)

    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_eda_panel.pdf", bbox_inches="tight")
    plt.close()


# Valores de la tabla de resultados de la memoria (tab:res-temporal),
# empleados como respaldo si aún no existe results/temporal_metricas.csv.
MAPE_HORIZONTE = {
    "ARIMA":      [1.16, 2.58, 4.76, 10.17],
    "ARIMA + RF": [1.19, 2.56, 4.71, 10.09],
    "LSTM":       [3.04, 3.85, 5.44, 9.15],
}
HORIZONTES = [1, 3, 6, 12]


def fig_mape_horizonte() -> None:
    """Evolución del MAPE medio con el horizonte de predicción (fig. 5.2).

    Lee las métricas de results/temporal_metricas.csv (generado por
    ``modelos_temporal.py``); si el fichero no existe, emplea los valores
    de la tabla de resultados de la memoria.
    """
    csv = RESULTS / "temporal_metricas.csv"
    if csv.exists():
        tabla = pd.read_csv(csv)
        mape = {m: g.sort_values("Horizonte")["MAPE"].tolist()
                for m, g in tabla.groupby("Modelo")}
    else:
        mape = MAPE_HORIZONTE

    estilo = {
        "ARIMA":      dict(color=AZUL, marker="o", ls="-", zorder=3),
        "ARIMA + RF": dict(color="#8fb8d9", marker="s", ls="--", zorder=4),
        "LSTM":       dict(color=NARANJA, marker="^", ls="-", zorder=3),
    }

    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    for nombre in ("ARIMA", "ARIMA + RF", "LSTM"):
        ax.plot(HORIZONTES, mape[nombre], lw=1.8, ms=7, label=nombre,
                **estilo[nombre])

    ax.set_xticks(HORIZONTES)
    ax.set_xlabel("Horizonte de predicción (meses)")
    ax.set_ylabel("MAPE (%)")
    ax.set_ylim(0, 11)
    # Coma decimal en el eje de ordenadas, coherente con la memoria.
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    ax.legend(frameon=False, loc="upper left")

    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_mape_horizonte.pdf", bbox_inches="tight")
    plt.close()


def main() -> None:
    FIGURAS.mkdir(parents=True, exist_ok=True)

    bas = conjunto_basico()
    print(f"Conjunto básico reconstruido: {len(bas)} anuncios")

    fig_precio(bas)
    fig_boxplots(bas)
    fig_correlacion(bas)
    fig_dispersion(bas)
    fig_mapa(bas)
    fig_panel()
    fig_mape_horizonte()

    print(f"Figuras generadas en {FIGURAS}")


if __name__ == "__main__":
    main()
