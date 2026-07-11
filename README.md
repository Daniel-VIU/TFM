# Predicción del Mercado Inmobiliario con Inteligencia Artificial

Código del Trabajo Fin de Máster *"Predicción del Mercado Inmobiliario con
Inteligencia Artificial: Integración de Machine Learning y Modelos de
Series Temporales"*.

## Estructura

```
├── data/
│   ├── raw/          # datos originales (no incluidos, ver abajo)
│   └── processed/    # datos generados por el pipeline
├── results/          # tablas de métricas generadas
└── src/
    ├── preparacion_datos.py    # limpieza idealista18 y panel de distritos
    ├── metricas.py             # RMSE, MAE, MAPE, R2
    ├── modelos_transversal.py  # regresión, random forest, boosting
    └── modelos_temporal.py     # ARIMA, LSTM, CNN-LSTM, híbrido ARIMA+RF
```

## Datos

En `data/raw/` deben situarse dos ficheros:

1. `madrid_sale.csv` — muestra de Madrid del conjunto abierto
   [idealista18](https://paezha.github.io/idealista18/) (licencia ODbL).
2. `Evolución_Precio_Idealista_Madrid.xlsx` — serie mensual de precio
   (€/m²) por distrito, recopilada de los informes públicos de idealista.

## Ejecución

```bash
pip install -r requirements.txt

# 1. Preparación de los datos
python -m src.preparacion_datos

# 2. Formulación transversal
python -m src.modelos_transversal

# 3. Formulación temporal (horizontes 1, 3, 6, 12 y 24 meses)
python -m src.modelos_temporal
```

Añadir `--rapido` a los módulos de modelos para una comprobación sobre
una submuestra. Las tablas de métricas quedan en `results/`. Las semillas
aleatorias están fijadas (`SEED = 2025`) para reproducibilidad.

Los modelos temporales se evalúan en cinco horizontes (1, 3, 6, 12 y
24 meses) con partición cronológica 60/20/20, de modo que el tramo de
prueba sea suficiente para evaluar el horizonte de 24 meses.
