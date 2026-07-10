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
    ├── preparacion_datos.py          # limpieza idealista18 y panel Madrid
    ├── preparacion_case_shiller.py   # panel Case-Shiller (EE. UU.)
    ├── construir_espaciotemporal.py  # fusión inmueble + contexto distrito
    ├── metricas.py                    # RMSE, MAE, MAPE, R2
    ├── modelos_transversal.py         # regresión, random forest, boosting
    ├── modelos_temporal.py            # ARIMA, LSTM, CNN-LSTM, híbrido
    └── modelos_espaciotemporal.py     # inmueble + contexto temporal
```

## Datos

En `data/raw/` deben situarse los ficheros siguientes:

1. `madrid_sale.csv` — muestra de Madrid del conjunto abierto
   [idealista18](https://paezha.github.io/idealista18/) (licencia ODbL).
2. `Evolución_Precio_Idealista_Madrid.xlsx` — serie mensual de precio
   (€/m²) por distrito de los informes públicos de idealista.
3. `distritos_madrid.geojson` — polígonos de los 21 distritos de Madrid.
4. `cities-month-SA.csv` — índice Case-Shiller mensual (desestacionalizado)
   de las 20 áreas metropolitanas de EE. UU., descargable de
   [datahub.io/core/house-prices-us](https://datahub.io/core/house-prices-us)
   (licencia PDDL, dominio público).

## Ejecución

```bash
pip install -r requirements.txt

# 1. Preparación de los datos
python -m src.preparacion_datos
python -m src.preparacion_case_shiller

# 2. Formulación transversal (idealista18)
python -m src.modelos_transversal

# 3. Formulación temporal — caso Madrid (mercado estable)
python -m src.modelos_temporal --dataset madrid

# 4. Formulación temporal — caso Case-Shiller (mercado volátil)
python -m src.modelos_temporal --dataset case_shiller

# 5. Conjunto espacio-temporal
python -m src.construir_espaciotemporal
python -m src.modelos_espaciotemporal
```

Añadir `--rapido` a cualquier módulo de modelos para una comprobación
sobre una submuestra. Las tablas de métricas quedan en `results/` (los
ficheros del caso Case-Shiller llevan el sufijo `_case_shiller`). Las
semillas aleatorias están fijadas (`SEED = 2025`) para reproducibilidad.

## Los dos casos de estudio temporal

El trabajo compara los mismos modelos en dos regímenes de mercado
distintos: el panel de distritos de Madrid (mercado maduro, baja
volatilidad mensual) y el índice Case-Shiller (mercado que atraviesa el
ciclo de burbuja y crisis de 2006-2012, con caídas superiores al 50 %).
El contraste permite estudiar en qué condiciones los modelos de
aprendizaje profundo superan al ARIMA.
