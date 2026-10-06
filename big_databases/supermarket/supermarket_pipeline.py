from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.io import arff
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "supermarket.arff"

TARGET = "total"
SEED = 42
TEST_SIZE = 0.30
CLUSTER_SIZES = (3, 5)
APRIORI_SWEEP = [(0.30, 0.90), (0.20, 0.90), (0.15, 0.90)]
APRIORI_RUNS = [
    {"name": "A", "support": 0.15, "confidence": 0.90, "max_len": None, "sort_by": "confidence"},
    {"name": "B", "support": 0.10, "confidence": 0.75, "max_len": 3, "sort_by": "lift"},
]
RULES_PER_RUN = 10

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 120)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_dataset():
    section("PASO 1 - CARGA DEL ARCHIVO ARFF")
    data, meta = arff.loadarff(DATA_PATH)
    df = pd.DataFrame(data).apply(lambda col: col.str.decode("utf-8"))
    types = pd.Series(meta.types()).value_counts()
    print(f"Relacion: {meta.name}")
    print(f"Tickets (filas): {len(df)}")
    print(f"Atributos (columnas): {df.shape[1]}")
    print(f"Tipos de atributo: {types.to_dict()}")
    print(f"\nValores posibles de 'bread and cake': {meta['bread and cake'][1]}")
    print(f"Valores posibles de '{TARGET}': {meta[TARGET][1]}")
    print(f"\nDistribucion de '{TARGET}':")
    print(df[TARGET].value_counts().to_string())
    return df


def preprocess(df):
    section("PASO 2 - PREPROCESAMIENTO")
    departments = df.drop(columns=TARGET)
    print("Valores distintos en las columnas de departamento:",
          sorted(pd.unique(departments.values.ravel())))

    baskets = (departments == "t").astype(int)
    purchases = baskets.sum()
    empty = purchases[purchases == 0].index
    baskets = baskets.drop(columns=empty)
    print(f"Departamentos sin ninguna compra: {len(empty)}")
    print(f"Departamentos que se conservan: {baskets.shape[1]}")

    per_ticket = baskets.sum(axis=1)
    print(f"\nDepartamentos por ticket: minimo {per_ticket.min()}, "
          f"mediana {per_ticket.median():.0f}, maximo {per_ticket.max()}")
    medians = per_ticket.groupby(df[TARGET]).median()
    print(f"Mediana segun '{TARGET}': low {medians['low']:.0f}, high {medians['high']:.0f}")
    print("\nDepartamentos mas frecuentes (tickets que los incluyen):")
    print(purchases.sort_values(ascending=False).head(10).to_string())
    return baskets, df[TARGET]


def classify(baskets, target):
    section("PASO 3 - CLASIFICACION CON NAIVE BAYES")
    X_train, X_test, y_train, y_test = train_test_split(
        baskets, target, test_size=TEST_SIZE, stratify=target, random_state=SEED
    )
    print(f"Entrenamiento: {len(X_train)} tickets | Prueba: {len(X_test)} tickets")

    model = BernoulliNB()
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    labels = ["low", "high"]
    matrix = pd.DataFrame(
        confusion_matrix(y_test, predicted, labels=labels),
        index=[f"real {label}" for label in labels],
        columns=[f"predicho {label}" for label in labels],
    )
    print(f"\nExactitud: {accuracy_score(y_test, predicted):.4f}")
    print("\nMatriz de confusion:")
    print(matrix.to_string())
    print("\nReporte por clase:")
    print(classification_report(y_test, predicted, labels=labels, digits=3))
    return model


def profile_clusters(baskets, target, clusters, top_departments):
    profile = pd.DataFrame({
        "tickets": clusters.value_counts().sort_index(),
        "% high": (target == "high").groupby(clusters).mean().mul(100).round(1),
        "deptos/ticket": baskets.sum(axis=1).groupby(clusters).mean().round(1),
    })
    shares = baskets[top_departments].groupby(clusters).mean().mul(100).round(0)
    return profile.join(shares.astype(int))


def cluster(baskets, target):
    section("PASO 4 - CLUSTERING CON K-MEANS")
    top_departments = baskets.sum().sort_values(ascending=False).head(8).index.tolist()
    print("Columnas de departamento: % de tickets del grupo que lo incluyen")
    for k in CLUSTER_SIZES:
        model = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        clusters = pd.Series(model.fit_predict(baskets), index=baskets.index, name="grupo")
        print(f"\n--- k = {k} | inercia = {model.inertia_:,.0f} ---")
        print(profile_clusters(baskets, target, clusters, top_departments).to_string())


def to_transactions(baskets, target):
    labels = pd.DataFrame({f"{TARGET}={label}": target == label for label in ("low", "high")})
    return pd.concat([baskets.astype(bool), labels], axis=1)


def format_rule(itemset):
    return ", ".join(sorted(itemset))


def mine_rules(items, min_support, min_confidence, max_len=None):
    frequent = apriori(items, min_support=min_support, use_colnames=True, max_len=max_len)
    rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
    return frequent, rules


def print_rules(rules, first_number):
    for number, (_, rule) in enumerate(rules.iterrows(), start=first_number):
        print(f"{number:>2}. {format_rule(rule['antecedents'])} "
              f"==> {format_rule(rule['consequents'])}")
        print(f"    soporte={rule['support']:.3f}  confianza={rule['confidence']:.3f}  "
              f"lift={rule['lift']:.2f}")


def associate(baskets, target):
    section("PASO 5 - REGLAS DE ASOCIACION CON APRIORI")
    items = to_transactions(baskets, target)

    print("Exploracion de umbrales:")
    print("soporte  confianza  conjuntos_frecuentes  reglas")
    for min_support, min_confidence in APRIORI_SWEEP:
        frequent, rules = mine_rules(items, min_support, min_confidence)
        print(f"{min_support:>7.2f}  {min_confidence:>9.2f}  {len(frequent):>20}  {len(rules):>6}")

    number = 1
    for run in APRIORI_RUNS:
        _, rules = mine_rules(items, run["support"], run["confidence"], run["max_len"])
        top = rules.sort_values([run["sort_by"], "support"], ascending=False).head(RULES_PER_RUN)
        max_items = run["max_len"] or "sin limite"
        print(f"\n--- Corrida {run['name']}: soporte >= {run['support']}, "
              f"confianza >= {run['confidence']}, max. elementos por regla: {max_items}, "
              f"orden por {run['sort_by']} ({len(rules)} reglas, se muestran {RULES_PER_RUN}) ---")
        print_rules(top, number)
        number += RULES_PER_RUN


def main():
    df = load_dataset()
    baskets, target = preprocess(df)
    classify(baskets, target)
    cluster(baskets, target)
    associate(baskets, target)


if __name__ == "__main__":
    main()
