"""K-means and EM clustering over the FLOR petal measurements."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "FLOR.arff"
PLOTS_DIR = BASE_DIR / "plots"

SEED = 10
K_MAIN = 3
K_RANGE = range(2, 6)
SEEDS_TO_COMPARE = range(1, 11)
EM_MAX_COMPONENTS = 6
EM_FOLDS = 10
CANOPY_T2_FACTOR = 0.5
CANOPY_T1_FACTOR = 1.25
K_STABILITY = (3, 4)

FEATURES = ["petallength", "petalwidth"]
GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GROUP_MARKERS = ["o", "s", "^", "D", "v"]
INK = "#292929"
GRID = "#e1e0d9"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_flowers():
    section("CARGA DE DATOS Y ELIMINACION DEL ATRIBUTO class")
    records, _ = arff.loadarff(DATA_PATH)
    flowers = pd.DataFrame(records)
    flowers["class"] = flowers["class"].str.decode("utf-8")
    print(f"Filas: {len(flowers)}  Columnas: {list(flowers.columns)}")
    print(flowers.describe().round(2).to_string())

    species = flowers.pop("class")
    print(f"\nColumnas para agrupar: {list(flowers.columns)}")
    return flowers, species


def scale(flowers):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(flowers[FEATURES])
    return scaled, scaler


def centroids_in_cm(model, scaler):
    centers = scaler.inverse_transform(model.cluster_centers_)
    return pd.DataFrame(centers, columns=FEATURES).round(3)


def style_axes(axis, title):
    axis.set_title(title, color=INK, fontsize=11)
    axis.set_xlabel("petallength (cm)", color=INK)
    axis.set_ylabel("petalwidth (cm)", color=INK)
    axis.grid(color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def plot_groups(flowers, labels, centers, title, path):
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for group in sorted(set(labels)):
        members = flowers[labels == group]
        axis.scatter(members["petallength"], members["petalwidth"], s=28,
                     color=GROUP_COLORS[group], marker=GROUP_MARKERS[group],
                     edgecolor="white", linewidth=0.6,
                     label=f"Grupo {group} ({len(members)})")
    if centers is not None:
        axis.scatter(centers["petallength"], centers["petalwidth"], s=180,
                     marker="X", color=INK, edgecolor="white", label="Centroide")
    style_axes(axis, title)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def kmeans_three_groups(flowers, species, scaled, scaler):
    section("K-MEANS CON TRES GRUPOS")
    model = KMeans(n_clusters=K_MAIN, init="random", n_init=1, random_state=SEED)
    labels = model.fit_predict(scaled)
    print(f"Iteraciones: {model.n_iter_}")
    print(f"SSE (datos escalados a [0,1]): {model.inertia_:.4f}")
    print("\nCentroides (cm):")
    print(centroids_in_cm(model, scaler).to_string())
    print("\nFlores por grupo:")
    print(pd.Series(labels).value_counts().sort_index().to_string())

    section("ASIGNACION DE GRUPOS")
    assigned = flowers.assign(grupo=labels)
    print(assigned.sample(8, random_state=SEED).sort_index().to_string())
    plot_groups(flowers, labels, centroids_in_cm(model, scaler),
                "K-means con k = 3", PLOTS_DIR / "kmeans_k3.png")

    section("GRUPOS CONTRA LA ESPECIE ORIGINAL")
    print(pd.crosstab(labels, species, rownames=["grupo"], colnames=["especie"]).to_string())
    return labels


def sse_by_k(scaled):
    section("SSE PARA k = 2, 3, 4 y 5")
    rows = []
    for k in K_RANGE:
        model = KMeans(n_clusters=k, init="random", n_init=1, random_state=SEED).fit(scaled)
        rows.append({"k": k, "SSE": round(model.inertia_, 4),
                     "flores_por_grupo": sorted(np.bincount(model.labels_).tolist(), reverse=True)})
    table = pd.DataFrame(rows)
    table["reduccion_vs_k_anterior"] = (-table["SSE"].pct_change() * 100).round(1)
    print(table.to_string(index=False))

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(table["k"], table["SSE"], color=GROUP_COLORS[0], linewidth=2, marker="o", markersize=8)
    for k, sse in zip(table["k"], table["SSE"]):
        axis.annotate(f"{sse:.2f}", (k, sse), textcoords="offset points", xytext=(8, 6), color=INK)
    axis.set_xticks(list(K_RANGE))
    axis.set_title("SSE según el número de grupos", color=INK, fontsize=11)
    axis.set_xlabel("k", color=INK)
    axis.set_ylabel("SSE", color=INK)
    axis.grid(color=GRID, linewidth=0.6)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "sse_por_k.png", dpi=150)
    plt.close(figure)
    return table


def distances_to(points, center):
    return np.linalg.norm(points - center, axis=1)


def farthest_first_centers(scaled, k, seed):
    rng = np.random.default_rng(seed)
    centers = [scaled[rng.integers(len(scaled))]]
    while len(centers) < k:
        nearest = np.min([distances_to(scaled, c) for c in centers], axis=0)
        centers.append(scaled[np.argmax(nearest)])
    return np.array(centers)


def canopy_centers(scaled, k, seed):
    t2 = CANOPY_T2_FACTOR * scaled.std(axis=0).mean()
    t1 = CANOPY_T1_FACTOR * t2
    rng = np.random.default_rng(seed)
    candidates = list(rng.permutation(len(scaled)))
    canopies = []
    while candidates:
        center = scaled[candidates[0]]
        distance = distances_to(scaled, center)
        canopies.append(scaled[distance < t1])
        candidates = [i for i in candidates if distance[i] >= t2]
    largest = sorted(canopies, key=len, reverse=True)[:k]
    return np.array([canopy.mean(axis=0) for canopy in largest])


INIT_METHODS = {
    "aleatorio": lambda scaled, k, seed: "random",
    "k-means++": lambda scaled, k, seed: "k-means++",
    "Canopy": canopy_centers,
    "Farthest First": farthest_first_centers,
}


def fit_with_init(scaled, method, k, seed):
    init = INIT_METHODS[method](scaled, k, seed)
    model = KMeans(n_clusters=k, init=init, n_init=1, random_state=seed)
    return model.fit(scaled)


def compare_initializations(scaled):
    section(f"METODOS DE INICIALIZACION (k = {K_MAIN}, semilla {SEED})")
    rows = []
    for method in INIT_METHODS:
        model = fit_with_init(scaled, method, K_MAIN, SEED)
        rows.append({"metodo": method, "SSE": round(model.inertia_, 4),
                     "iteraciones": model.n_iter_,
                     "flores_por_grupo": sorted(np.bincount(model.labels_).tolist(), reverse=True)})
    print(pd.DataFrame(rows).to_string(index=False))

    for k in K_STABILITY:
        section(f"ESTABILIDAD CON {len(SEEDS_TO_COMPARE)} SEMILLAS DISTINTAS (k = {k})")
        sse_by_method = {
            method: [fit_with_init(scaled, method, k, seed).inertia_ for seed in SEEDS_TO_COMPARE]
            for method in INIT_METHODS
        }
        best_sse = min(min(sse) for sse in sse_by_method.values())
        rows = [{"metodo": method, "SSE_min": round(min(sse), 4), "SSE_max": round(max(sse), 4),
                 "veces_en_el_mejor_SSE": int(np.isclose(sse, best_sse).sum())}
                for method, sse in sse_by_method.items()]
        print(pd.DataFrame(rows).to_string(index=False))


def cross_validated_loglikelihood(raw, components):
    folds = KFold(n_splits=EM_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    for train, test in folds.split(raw):
        model = GaussianMixture(components, covariance_type="diag", random_state=SEED)
        scores.append(model.fit(raw[train]).score(raw[test]))
    return np.mean(scores)


def expectation_maximization(flowers, species):
    section("EM: ELECCION DEL NUMERO DE GRUPOS POR VALIDACION CRUZADA")
    raw = flowers[FEATURES].to_numpy()
    best, best_score = 1, cross_validated_loglikelihood(raw, 1)
    print(f"  1 grupos -> log-verosimilitud media: {best_score:.4f}")
    while best < EM_MAX_COMPONENTS:
        score = cross_validated_loglikelihood(raw, best + 1)
        print(f"  {best + 1} grupos -> log-verosimilitud media: {score:.4f}")
        if score <= best_score:
            break
        best, best_score = best + 1, score
    print(f"Numero de grupos elegido: {best}")

    section(f"EM CON {best} GRUPOS")
    model = GaussianMixture(best, covariance_type="diag", random_state=SEED).fit(raw)
    labels = model.predict(raw)
    summary = pd.DataFrame({
        "peso": model.weights_,
        "media_petallength": model.means_[:, 0],
        "desv_petallength": np.sqrt(model.covariances_[:, 0]),
        "media_petalwidth": model.means_[:, 1],
        "desv_petalwidth": np.sqrt(model.covariances_[:, 1]),
    }).round(3)
    print(summary.to_string())
    print(f"\nIteraciones: {model.n_iter_}  Log-verosimilitud media: {model.score(raw):.4f}")
    print("\nFlores por grupo:")
    print(pd.Series(labels).value_counts().sort_index().to_string())

    section("GRUPOS DE EM CONTRA LA ESPECIE ORIGINAL")
    print(pd.crosstab(labels, species, rownames=["grupo"], colnames=["especie"]).to_string())
    centers = pd.DataFrame(model.means_, columns=FEATURES)
    plot_groups(flowers, labels, centers, f"EM con {best} grupos", PLOTS_DIR / "em.png")


def main():
    PLOTS_DIR.mkdir(exist_ok=True)
    flowers, species = load_flowers()
    scaled, scaler = scale(flowers)
    kmeans_three_groups(flowers, species, scaled, scaler)
    sse_by_k(scaled)
    compare_initializations(scaled)
    expectation_maximization(flowers, species)


if __name__ == "__main__":
    main()
