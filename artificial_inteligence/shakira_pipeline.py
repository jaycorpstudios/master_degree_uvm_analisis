"""Stage 2 of the project: classify the SHAKIRA tweets.

Reloads the three classifiers persisted in stage 1 (a linear SVM, a decision
tree and a multinomial Naive Bayes), re-runs their stage 1 evaluation, applies
them to the SHAKIRA file, exports one result file per classifier, contrasts the
predicted distributions in a bar chart and measures each classifier against a
hand-labelled reference sample.
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split

import sentiment_pipeline as etapa1
from sentiment_pipeline import clean_text, section

# --- Configuration (tune the run from here) ---
BASE_DIR = Path(__file__).parent / "shakira"
SHAKIRA_PATH = BASE_DIR / "SHAKIRA.csv"
REFERENCE_PATH = BASE_DIR / "referencia_manual.csv"
OUTPUT_DIR = BASE_DIR / "output"
PLOTS_DIR = BASE_DIR / "plots"

CLASSIFIERS = {
    "Clasificador 1 - SMO (SVM lineal)": ("svm_smo.joblib", "SHAKIRA1.csv"),
    "Clasificador 2 - J48 (árbol de decisión)": ("decision_tree_j48.joblib", "SHAKIRA2.csv"),
    "Clasificador 3 - Naive Bayes (multinomial)": ("naive_bayes.joblib", "SHAKIRA3.csv"),
}

TARGET_CODES = {"negativo": "0", "positivo": "4"}
OUTPUT_COLUMNS = ["target", "ids", "date", "user", "text"]
CLASS_ORDER = ["negativo", "positivo"]

DATE_FORMAT = "%d/%m/%Y %H:%M"
REFERENCE_SAMPLE_SIZE = 100
RANDOM_STATE = 42


# === Inciso a: resumen de la evaluacion de la etapa 1 ===

def rebuild_stage1_split():
    """Rebuild stage 1's train/test split so the saved models can be re-scored.

    Every step downstream of the raw file is deterministic (fixed seeds), so
    replaying it yields byte-for-byte the same split the models were fitted on.
    """
    df = pd.read_csv(
        etapa1.DATA_PATH,
        encoding="latin-1",
        header=None,
        names=etapa1.COLUMN_NAMES,
        dtype=str,
        keep_default_na=False,
    )
    df = df[df["target"].isin(etapa1.VALID_LABELS)].copy()
    df = df.drop_duplicates(subset=["text"]).copy()
    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.len() > 0].copy()
    df["sentiment"] = df["target"].map(etapa1.LABEL_NAMES)

    parts = [
        group.sample(n=min(etapa1.SAMPLE_PER_CLASS, len(group)), random_state=etapa1.RANDOM_STATE)
        for _, group in df.groupby("sentiment")
    ]
    sample = (
        pd.concat(parts)
        .sample(frac=1, random_state=etapa1.RANDOM_STATE)
        .reset_index(drop=True)
    )

    X_train, X_test, _, y_test = train_test_split(
        sample["clean_text"],
        sample["sentiment"],
        test_size=etapa1.TEST_SIZE,
        random_state=etapa1.RANDOM_STATE,
        stratify=sample["sentiment"],
    )
    return X_train, X_test, y_test


def resumen_etapa1(pipelines, X_test, y_test):
    section("INCISO A - RESUMEN DE LA EVALUACION DE LA ETAPA 1")
    print(f"Conjunto de prueba reconstruido: {len(X_test):,} tweets\n")

    results = []
    for name, pipeline in pipelines.items():
        section(f"{name} (evaluacion etapa 1)")
        results.append(etapa1.evaluate(name, pipeline, X_test, y_test))

    section("RESUMEN COMPARATIVO DE LA ETAPA 1")
    print(pd.DataFrame(results).to_string(index=False))
    return results


# === Carga, validacion y preparacion del archivo SHAKIRA ===

def load_models():
    return {
        name: joblib.load(etapa1.MODELS_DIR / filename)
        for name, (filename, _) in CLASSIFIERS.items()
    }


def load_shakira():
    section("CARGA DEL ARCHIVO SHAKIRA")
    df = pd.read_csv(SHAKIRA_PATH, dtype=str, keep_default_na=False)
    print(f"Filas leidas: {len(df):,}")
    print(f"Columnas: {list(df.columns)}\n")
    print("Primeras filas (columnas author, content y date_time):")
    print(df[["author", "content", "date_time"]].head(3).to_string(index=False, max_colwidth=70))
    return df


def report_quality(df):
    section("VALIDACION DE CALIDAD DEL ARCHIVO SHAKIRA")
    print("Idiomas presentes:")
    print(df["language"].value_counts().to_string())
    print("\nAutores presentes:")
    print(df["author"].value_counts().to_string())

    scientific = df["id"].str.contains("E+", regex=False).sum()
    print(f"\nIdentificadores en notacion cientifica: {scientific:,} de {len(df):,}")
    print(f"Identificadores unicos: {df['id'].nunique():,} (deberian ser {len(df):,})")
    print(f"Ejemplo de identificador: {df['id'].iloc[0]}")

    dates = pd.to_datetime(df["date_time"], format=DATE_FORMAT, errors="coerce")
    print(f"\nFechas no interpretables: {dates.isna().sum():,}")
    print(f"Rango de fechas: {dates.min():%Y-%m-%d} a {dates.max():%Y-%m-%d}")

    print(f"\nTextos vacios: {(df['content'].str.strip() == '').sum():,}")
    print(f"Textos duplicados: {df.duplicated(subset=['content']).sum():,}")


def prepare(df):
    section("PREPARACION Y LIMPIEZA")
    prepared = pd.DataFrame(
        {
            "ids": df["id"],
            "date": pd.to_datetime(df["date_time"], format=DATE_FORMAT).dt.strftime("%Y-%m-%d %H:%M"),
            "user": df["author"],
            "text": df["content"],
            "clean_text": df["content"].map(clean_text),
        }
    )
    prepared = prepared[prepared["clean_text"].str.len() > 0].copy().reset_index(drop=True)
    print(f"Tweets tras la limpieza: {len(prepared):,}")

    print("\nEjemplo de limpieza de texto:")
    print(f"  original: {prepared['text'].iloc[0][:95]}")
    print(f"  limpio  : {prepared['clean_text'].iloc[0][:95]}")
    return prepared


# === Diagnostico de cobertura del vocabulario ===

def vocabulary_coverage(pipelines, df):
    section("COBERTURA DEL VOCABULARIO APRENDIDO EN LA ETAPA 1")
    vectorizer = next(iter(pipelines.values())).named_steps["tfidf"]
    vocabulary = vectorizer.vocabulary_
    print(f"Terminos del vocabulario aprendido en la etapa 1: {len(vocabulary):,}")

    matrix = vectorizer.transform(df["clean_text"])
    features_per_tweet = matrix.getnnz(axis=1)
    print(f"Caracteristicas reconocidas por tweet: media {features_per_tweet.mean():.2f}"
          f" | mediana {np.median(features_per_tweet):.0f}"
          f" | maximo {features_per_tweet.max()}")

    empty = int((features_per_tweet == 0).sum())
    print(f"Tweets sin ninguna caracteristica reconocida: {empty:,}"
          f" ({empty / len(df):.1%})")

    inverse = {index: term for term, index in vocabulary.items()}
    counts = pd.Series(matrix.nonzero()[1]).value_counts()
    top = [(inverse[index], int(total)) for index, total in counts.head(15).items()]
    print("\nTerminos del vocabulario que mas se activan en estos tweets:")
    print(", ".join(f"{term} ({total})" for term, total in top))

    return features_per_tweet > 0, [term for term, _ in top]


def vocabulary_origin(pipelines, X_train, terms):
    """Show which training tweets put each term in the vocabulary in the first place."""
    section("DE DONDE SALIERON ESOS TERMINOS EN EL ENTRENAMIENTO")
    vectorizer = next(iter(pipelines.values())).named_steps["tfidf"]
    print(f"Filtro de palabras vacias del vectorizador: {vectorizer.stop_words}")
    print(f"Minimo de documentos para entrar al vocabulario (min_df): {vectorizer.min_df}\n")

    analyzer = vectorizer.build_analyzer()
    tokenized = [set(analyzer(text)) for text in X_train]
    for term in terms:
        hits = [text for text, tokens in zip(X_train, tokenized) if term in tokens]
        print(f"{term}: aparece en {len(hits)} tweets de entrenamiento")
        for example in hits[:2]:
            print(f"    {example[:88]}")


# === Incisos b a g: clasificacion y exportacion ===

def classify_and_export(pipelines, df, classifiable):
    section("INCISOS B a G - CLASIFICACION Y EXPORTACION")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    omitted = int((~classifiable).sum())
    print(f"Tweets omitidos por no activar ninguna caracteristica: {omitted:,}")
    print(f"Tweets que se exportan: {int(classifiable.sum()):,}\n")

    predictions = {}
    for name, (_, output_file) in CLASSIFIERS.items():
        section(name)
        predicted = pd.Series(pipelines[name].predict(df["clean_text"]), index=df.index)
        predictions[name] = predicted

        print("Distribucion de las predicciones (sobre los tweets exportados):")
        distribution = predicted[classifiable].value_counts().reindex(CLASS_ORDER, fill_value=0)
        for label, total in distribution.items():
            print(f"  {label}: {total:,} ({total / classifiable.sum():.1%})")

        classifier = pipelines[name].named_steps["clf"]
        if hasattr(classifier, "predict_proba"):
            confidence = pd.Series(pipelines[name].predict_proba(df["clean_text"]).max(axis=1).round(3))
            repeated = confidence.value_counts()
            print(f"  confianza mas repetida: {repeated.index[0]}"
                  f" en el {repeated.iloc[0] / len(df):.1%} de los tweets")

        exported = df[classifiable].copy()
        exported.insert(0, "target", predicted[classifiable].map(TARGET_CODES))
        exported = exported[OUTPUT_COLUMNS]
        exported.to_csv(OUTPUT_DIR / output_file, index=False)
        print(f"\nArchivo generado: {output_file} ({len(exported):,} registros)")
        print(exported.head(3).to_string(index=False, max_colwidth=45))

    return pd.DataFrame(predictions)


# === Inciso h: grafica comparativa ===

def plot_comparison(predictions, classifiable):
    section("INCISO H - CONTRASTE DE LOS TRES CLASIFICADORES")
    counts = pd.DataFrame(
        {
            name: predictions.loc[classifiable, name].value_counts().reindex(CLASS_ORDER, fill_value=0)
            for name in predictions.columns
        }
    ).T
    print(counts.to_string())

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = [name.split(" - ")[1] for name in counts.index]
    positions = np.arange(len(labels))
    width = 0.35

    figure, axes = plt.subplots(figsize=(9, 5))
    for offset, (sentiment, color) in enumerate(zip(CLASS_ORDER, ["#c0392b", "#27ae60"])):
        bars = axes.bar(
            positions + (offset - 0.5) * width,
            counts[sentiment],
            width,
            label=sentiment.capitalize(),
            color=color,
        )
        axes.bar_label(bars, padding=2, fontsize=9)

    axes.set_title("Tweets clasificados por cada clasificador (archivo SHAKIRA)")
    axes.set_ylabel("Número de tweets")
    axes.set_xticks(positions, labels, fontsize=9)
    axes.set_ylim(0, counts.to_numpy().max() * 1.15)
    axes.legend(title="Polaridad predicha")
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()

    path = PLOTS_DIR / "comparativa_clasificadores.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"\nGrafica guardada en: {path.relative_to(Path(__file__).parent)}")
    return counts


# === Inciso i: concordancia con la muestra etiquetada a mano ===

def build_reference_sample(df, classifiable):
    """Draw the tweets to be hand-labelled. Only needed to create the file once."""
    sample = df[classifiable].sample(n=REFERENCE_SAMPLE_SIZE, random_state=RANDOM_STATE)
    return sample.sort_index()[["ids", "date", "text"]]


def manual_agreement(predictions, df):
    section("INCISO I - CONCORDANCIA CON LA MUESTRA ETIQUETADA A MANO")
    reference = pd.read_csv(REFERENCE_PATH, dtype=str, keep_default_na=False)
    reference = reference.set_index(reference["indice"].astype(int))

    print(f"Tweets etiquetados a mano: {len(reference):,}")
    print("\nDistribucion de las etiquetas manuales:")
    distribution = reference["etiqueta_manual"].value_counts()
    for label, total in distribution.items():
        print(f"  {label}: {total} ({total / len(reference):.1%})")

    binary = reference[reference["etiqueta_manual"].isin(CLASS_ORDER)]
    print(f"\nSubconjunto comparable (positivo o negativo): {len(binary)} tweets")
    print("Los tweets neutrales quedan fuera: los modelos no pueden predecir esa clase.\n")

    truth = binary["etiqueta_manual"]
    results = []
    for name in predictions.columns:
        predicted = predictions.loc[binary.index, name]
        results.append(
            {
                "clasificador": name.split(" - ")[1],
                "exactitud": round(accuracy_score(truth, predicted), 4),
                "kappa": round(cohen_kappa_score(truth, predicted, labels=CLASS_ORDER), 4),
            }
        )
        print(f"{name}")
        print("  Matriz de confusion (filas=manual, columnas=prediccion) [negativo, positivo]:")
        print("  " + str(confusion_matrix(truth, predicted, labels=CLASS_ORDER)).replace("\n", "\n  "))

    print("\nResumen de concordancia contra la muestra de referencia:")
    print(pd.DataFrame(results).to_string(index=False))

    print("\nEjemplos (etiqueta manual | prediccion de los clasificadores 1 / 2 / 3):")
    split = [index for index in binary.index if predictions.loc[index].nunique() > 1]
    examples = list(binary.index[binary["etiqueta_manual"] == "negativo"]) + split[:4]
    for index in examples:
        print(f"  {binary.loc[index, 'etiqueta_manual']:<9}"
              f" | {' / '.join(predictions.loc[index])}")
        print(f"    {df.loc[index, 'text'][:105]}")

    return results


def main():
    pipelines = load_models()
    X_train, X_test, y_test = rebuild_stage1_split()
    resumen_etapa1(pipelines, X_test, y_test)

    raw = load_shakira()
    report_quality(raw)
    df = prepare(raw)

    classifiable, top_terms = vocabulary_coverage(pipelines, df)
    vocabulary_origin(pipelines, X_train, top_terms[:9])
    predictions = classify_and_export(pipelines, df, classifiable)
    plot_comparison(predictions, classifiable)
    manual_agreement(predictions, df)


if __name__ == "__main__":
    main()
