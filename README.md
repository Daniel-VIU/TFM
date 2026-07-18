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
    ├── graficas_eda.py         # figuras del análisis exploratorio
    ├── metricas.py             # RMSE, MAE, MAPE, R2
    ├── modelos_transversal.py  # regresión, random forest, boosting
    └── modelos_temporal.py     # ARIMA, LSTM, híbrido ARIMA+RF
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

# 3. Formulación temporal (horizontes 1, 3, 6 y 12 meses)
python -m src.modelos_temporal
```

Añadir `--rapido` a los módulos de modelos para una comprobación sobre
una submuestra. Las tablas de métricas quedan en `results/`. Las semillas
aleatorias están fijadas (`SEED = 2025`) para reproducibilidad.

Los modelos temporales se evalúan en cuatro horizontes (1, 3, 6 y 12
meses) con partición cronológica 70/20/10.

## Notas sobre la formulación temporal

- **Híbrido ARIMA + random forest.** El corrector de residuos se
  construye estrictamente con la información disponible en el origen de
  cada predicción: la cola de residuos contiene solo los realizados
  hasta ese origen y, para el objetivo situado `h` meses por delante, el
  bosque predice recursivamente los `h` residuos siguientes
  realimentando sus propias estimaciones. La corrección aplicada al
  nivel del ARIMA es la exponencial de su suma, coherente con el
  encadenamiento de log-retornos.

- **Asimetría de información ARIMA / LSTM.** El ARIMA se reajusta de
  forma rodante con toda la historia disponible en cada origen de
  prueba, mientras que la LSTM se entrena solo con el 70 % inicial de
  cada serie (la validación interviene únicamente en la parada
  temprana). Es una decisión de diseño deliberada, documentada en la
  memoria, que sitúa al modelo clásico como línea base exigente. La
  opción

  ```bash
  python -m src.modelos_temporal --incluir-validacion
  ```

  reentrena la LSTM con el tramo de validación incluido (durante el
  número de épocas fijado por la parada temprana) y permite cuantificar
  el efecto de esa asimetría como experimento de sensibilidad; la
  configuración por defecto es la reportada en la memoria.
