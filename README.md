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
    ├── preparacion_datos.py          # limpieza y construcción del panel
    ├── construir_espaciotemporal.py  # fusión inmueble + contexto distrito
    ├── metricas.py                    # RMSE, MAE, MAPE, R2
    ├── modelos_transversal.py         # regresión, random forest, boosting
    ├── modelos_temporal.py            # ARIMA, LSTM, CNN-LSTM, híbrido
    └── modelos_espaciotemporal.py     # inmueble + contexto temporal
```

## Datos

En `data/raw/` deben situarse tres ficheros:

1. `madrid_sale.csv` — muestra de Madrid del conjunto abierto
   [idealista18](https://paezha.github.io/idealista18/) (licencia ODbL).
2. `Evolución_Precio_Idealista_Madrid.xlsx` — libro con una hoja por
   distrito, con la serie mensual de precio (€/m²) de los informes
   públicos de idealista.
3. `distritos_madrid.geojson` — polígonos de los 21 distritos de Madrid,
   usados para asignar cada inmueble a su distrito por cruce espacial.

## Ejecución

```bash
pip install -r requirements.txt

# 1. Preparación de los datos
python -m src.preparacion_datos

# 2. Formulación transversal
python -m src.modelos_transversal

# 3. Formulación temporal (multi-horizonte 1/3/6 meses)
python -m src.modelos_temporal

# 4. Conjunto espacio-temporal y su evaluación
python -m src.construir_espaciotemporal
python -m src.modelos_espaciotemporal
```

Añadir `--rapido` a cualquier módulo de modelos para una comprobación
sobre una submuestra. Las tablas de métricas quedan en `results/`. Las
semillas aleatorias están fijadas (`SEED = 2025`) para reproducibilidad.
