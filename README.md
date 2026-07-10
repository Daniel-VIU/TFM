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
    ├── preparacion_datos.py    # limpieza y construcción del panel
    ├── metricas.py             # RMSE, MAE, MAPE, R2
    ├── modelos_transversal.py  # regresión, random forest, boosting
    └── modelos_temporal.py     # ARIMA, LSTM, CNN-LSTM, híbrido
```

## Datos

En `data/raw/` deben situarse dos ficheros:

1. `madrid_sale.csv` — muestra de Madrid del conjunto abierto
   [idealista18](https://paezha.github.io/idealista18/) (licencia ODbL),
   exportada a CSV desde el paquete de R.
2. `Evolución_Precio_Idealista_Madrid.xlsx` — libro con una hoja por
   distrito, con la serie mensual de precio (€/m²) recopilada de los
   informes públicos de precios de idealista.

## Ejecución

```bash
pip install -r requirements.txt

# 1. Preparación de los datos
python -m src.preparacion_datos

# 2. Formulación transversal (añadir --rapido para una comprobación)
python -m src.modelos_transversal

# 3. Formulación temporal (añadir --rapido para una comprobación)
python -m src.modelos_temporal
```

Las tablas de métricas quedan en `results/`. Las semillas aleatorias
están fijadas (`SEED = 2025`) para que los resultados sean reproducibles.
