# Inteligencia Aplicada a la Ciencia de Datos

Proyecto Integrador de análisis de sentimientos sobre tweets, en dos etapas.

**Etapa 1** (`sentiment_pipeline.py`): canalización de análisis de sentimientos sobre **sentimientos.txt** (tweets etiquetados como positivos o negativos). Construye una canalización (Pipeline) de machine learning y compara tres clasificadores (SVM lineal, árbol de decisión y Naive Bayes), equivalentes a los algoritmos SMO, J48 y NaiveBayes de Weka.

**Etapa 2** (`shakira_pipeline.py`): reutiliza los tres clasificadores ya entrenados para clasificar los tweets del archivo **SHAKIRA**, exporta un archivo de resultados por clasificador, contrasta las distribuciones en una gráfica de barras y mide cada clasificador contra una muestra etiquetada a mano.

## Estructura

```
artificial_inteligence/
├── sentiment_pipeline.py   # etapa 1: carga → validación → limpieza → muestreo → entrenamiento
├── shakira_pipeline.py     # etapa 2: reevaluación → clasificación → exportación → gráfica → concordancia
├── data/                   # NO versionado (ver .gitignore)
│   └── sentimientos.txt    # dataset crudo (~228 MB)
├── models/                 # clasificadores entrenados (.joblib), compartidos por ambas etapas
├── shakira/
│   ├── SHAKIRA.csv             # tweets a clasificar en la etapa 2
│   ├── referencia_manual.csv   # 100 tweets etiquetados a mano como muestra de referencia
│   ├── output/                 # SHAKIRA1.csv, SHAKIRA2.csv, SHAKIRA3.csv
│   └── plots/                  # gráfica comparativa de los tres clasificadores
└── README.md
```

El dataset de la etapa 1 no se versiona por su tamaño. Los tres clasificadores entrenados se guardan en
`models/` con `joblib`, que es lo que permite que la etapa 2 los reutilice sin reentrenar.

## Ejecución

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python artificial_inteligence/sentiment_pipeline.py   # etapa 1
python artificial_inteligence/shakira_pipeline.py     # etapa 2
```

La etapa 1 imprime, por etapa, los conteos de validación, el resultado del muestreo y el reporte de clasificación de cada modelo con su matriz de confusión. La etapa 2 reconstruye el conjunto de prueba de la etapa 1 para reevaluar los modelos guardados, informa la cobertura del vocabulario sobre los tweets nuevos, exporta los tres archivos de resultados y compara las predicciones con las etiquetas manuales.
