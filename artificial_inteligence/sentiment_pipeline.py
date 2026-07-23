"""Sentiment analysis pipeline for the Sentiment140 corpus (Project, Stage 1).

Loads the raw tweets, validates and cleans them, draws a balanced sample, and
trains and evaluates three classifiers (a linear SVM, a decision tree and a
multinomial Naive Bayes) that mirror Weka's SMO, J48 and NaiveBayes algorithms.
Each fitted pipeline is persisted to disk for reuse in a later project stage.
"""

import re
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# --- Configuration (tune the run from here) ---
DATA_PATH = Path(__file__).parent / "data" / "sentimientos.txt"
MODELS_DIR = Path(__file__).parent / "models"
COLUMN_NAMES = ["target", "ids", "date", "flag", "user", "text"]
VALID_LABELS = {"0", "2", "4"}
LABEL_NAMES = {"0": "negativo", "4": "positivo"}
POSITIVE_LABEL = "positivo"

SAMPLE_PER_CLASS = 10_000  # 10k por clase = 20k tweets en total
TEST_SIZE = 0.25
RANDOM_STATE = 42

NGRAM_RANGE = (1, 2)
MIN_DF = 5
MAX_FEATURES = 50_000

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")


# === Data utilities ===

def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_dataset():
    section("ETAPA 1 - RECOLECCION / CARGA")
    started = time.time()
    df = pd.read_csv(
        DATA_PATH,
        encoding="latin-1",
        header=None,
        names=COLUMN_NAMES,
        dtype=str,
        keep_default_na=False,
    )
    print(f"Filas leidas: {len(df):,}")
    print(f"Columnas: {list(df.columns)}")
    print(f"Tiempo de carga: {time.time() - started:.1f} s\n")
    print("Primeras filas (columnas target y text):")
    print(df[["target", "text"]].head(3).to_string(index=False))
    return df


def report_quality(df):
    section("ETAPA 2 - VALIDACION DE CALIDAD")
    print("Valores encontrados en el atributo objetivo (target):")
    print(df["target"].value_counts(dropna=False).to_string())

    invalid = df[~df["target"].isin(VALID_LABELS)]
    print(f"\nRegistros con target invalido (lineas corruptas): {len(invalid):,}")
    print("Clase neutral (2) presente:", "2" in set(df["target"].unique()))
    print(f"Textos vacios: {(df['text'].str.strip() == '').sum():,}")
    print(f"Textos duplicados: {df.duplicated(subset=['text']).sum():,}")


def clean_text(text):
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def clean_and_consolidate(df):
    section("ETAPA 3 - LIMPIEZA Y CONSOLIDACION")
    initial = len(df)

    df = df[df["target"].isin(VALID_LABELS)].copy()
    print(f"Tras descartar lineas corruptas: {len(df):,} (se eliminaron {initial - len(df):,})")

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["text"]).copy()
    print(f"Tras eliminar textos duplicados: {len(df):,} (se eliminaron {before_dedup - len(df):,})")

    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.len() > 0].copy()
    print(f"Tras eliminar textos vacios post-limpieza: {len(df):,}")

    df["sentiment"] = df["target"].map(LABEL_NAMES)
    print("\nDistribucion final por clase:")
    print(df["sentiment"].value_counts().to_string())

    sample_row = df.iloc[0]
    print("\nEjemplo de limpieza de texto:")
    print(f"  original: {sample_row['text'][:90]}")
    print(f"  limpio  : {sample_row['clean_text'][:90]}")
    return df


def stratified_sample(df):
    section("ETAPA 4 - MUESTREO ESTRATIFICADO")
    parts = [
        group.sample(n=min(SAMPLE_PER_CLASS, len(group)), random_state=RANDOM_STATE)
        for _, group in df.groupby("sentiment")
    ]
    sample = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Muestra balanceada: {len(sample):,} registros")
    print(sample["sentiment"].value_counts().to_string())
    return sample


# === Model training & evaluation (core) ===

def build_pipeline(classifier):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_features=MAX_FEATURES,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def positive_scores(pipeline, X):
    # LinearSVC no expone predict_proba; su decision_function sirve igual para ROC-AUC.
    classifier = pipeline.named_steps["clf"]
    if hasattr(classifier, "predict_proba"):
        classes = list(classifier.classes_)
        return pipeline.predict_proba(X)[:, classes.index(POSITIVE_LABEL)]
    return pipeline.decision_function(X)


def save_model(filename, pipeline):
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(pipeline, path)
    print(f"Modelo guardado en: {path.relative_to(Path(__file__).parent)}")


def evaluate(name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    scores = positive_scores(pipeline, X_test)

    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    roc_auc = roc_auc_score((y_test == POSITIVE_LABEL).astype(int), scores)
    f1 = f1_score(y_test, y_pred, pos_label=POSITIVE_LABEL)

    print(f"Exactitud (accuracy): {accuracy:.4f}")
    print(f"Kappa de Cohen: {kappa:.4f}")
    print(f"Area bajo la curva ROC (AUC): {roc_auc:.4f}")
    print(f"F1 (clase positivo): {f1:.4f}\n")
    print("Reporte de clasificacion:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Matriz de confusion (filas=real, columnas=prediccion) [negativo, positivo]:")
    print(confusion_matrix(y_test, y_pred, labels=["negativo", "positivo"]))

    return {"clasificador": name, "exactitud": round(accuracy, 4),
            "kappa": round(kappa, 4), "roc_auc": round(roc_auc, 4), "f1_positivo": round(f1, 4)}


def run_experiments(df):
    section("ETAPA 5/6 - PIPELINE, ENTRENAMIENTO Y EVALUACION")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["sentiment"],
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["sentiment"],
    )
    print(f"Entrenamiento: {len(X_train):,} | Prueba: {len(X_test):,}")

    classifiers = {
        "SMO (SVM lineal)": ("svm_smo.joblib", LinearSVC(C=1.0, random_state=RANDOM_STATE)),
        "J48 (arbol de decision)": ("decision_tree_j48.joblib",
                                     DecisionTreeClassifier(max_depth=40, random_state=RANDOM_STATE)),
        "Naive Bayes (multinomial)": ("naive_bayes.joblib", MultinomialNB()),
    }

    results = []
    for name, (filename, classifier) in classifiers.items():
        section(f"CLASIFICADOR: {name}")
        pipeline = build_pipeline(classifier)
        started = time.time()
        pipeline.fit(X_train, y_train)
        print(f"Tiempo de entrenamiento: {time.time() - started:.1f} s")
        results.append(evaluate(name, pipeline, X_test, y_test))
        save_model(filename, pipeline)

    section("RESUMEN COMPARATIVO")
    print(pd.DataFrame(results).to_string(index=False))


def main():
    df = load_dataset()
    report_quality(df)
    df = clean_and_consolidate(df)
    sample = stratified_sample(df)
    run_experiments(sample)


if __name__ == "__main__":
    main()
