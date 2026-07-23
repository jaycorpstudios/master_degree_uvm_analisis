# Inteligencia Aplicada a la Ciencia de Datos

Canalización de análisis de sentimientos sobre **sentimientos.txt** (tweets
etiquetados como positivos o negativos). Corresponde a la Etapa 1 del Proyecto Integrador: construir una canalización (Pipeline) de machine learning y comparar tres clasificadores (SVM lineal, árbol de decisión y Naive Bayes), equivalentes a los algoritmos SMO, J48 y NaiveBayes de Weka.

## Estructura

```
artificial_inteligence/
├── sentiment_pipeline.py   # pipeline completo: carga → validación → limpieza → muestreo → entrenamiento
├── data/                   # NO versionado (ver .gitignore)
│   └── sentimientos.txt    # dataset crudo (~228 MB)
├── models/                 # clasificadores entrenados (.joblib) para reusar en etapas posteriores
└── README.md
```

El dataset crudo no se versiona por su tamaño. Los tres clasificadores entrenados se guardan en
`models/` con `joblib` para reutilizarse en una etapa posterior del proyecto sin reentrenar.

## Ejecución

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python artificial_inteligence/sentiment_pipeline.py
```

El script imprime, por etapa, los conteos de validación, el resultado del muestreo y el reporte de clasificación de cada modelo con su matriz de confusión.
