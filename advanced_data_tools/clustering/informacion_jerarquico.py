"""Agglomerative clustering of INFORMACION with single, complete and average linkage."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "INFORMACION.csv"
PLOTS_DIR = BASE_DIR / "plots"

LINKAGES = {"single": "Enlace único", "complete": "Enlace completo", "average": "Enlace promedio"}
CUTS = (2, 3)
LINE_COLOR = "#2a78d6"
INK = "#292929"
GRID = "#e1e0d9"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_points():
    section("CARGA DE INFORMACION.csv")
    points = pd.read_csv(DATA_PATH)
    points.index = [f"P{i}" for i in range(1, len(points) + 1)]
    print(points.to_string())
    return points


def merge_table(merges, labels):
    names = list(labels)
    rows = []
    for step, (left, right, height, size) in enumerate(merges, start=1):
        joined = f"{names[int(left)]} + {names[int(right)]}"
        names.append(f"C{step}")
        rows.append({"paso": step, "une": joined, "nuevo": f"C{step}",
                     "distancia": round(height, 3), "puntos": int(size)})
    return pd.DataFrame(rows)


def build_dendrograms(points):
    distances = pdist(points.to_numpy())
    figure, axes = plt.subplots(1, len(LINKAGES), figsize=(13, 4.2), sharey=True)
    summary = []
    tallest_merge = 0
    for axis, (method, title) in zip(axes, LINKAGES.items()):
        section(f"{title.upper()} ({method})")
        merges = linkage(distances, method=method)
        print(merge_table(merges, points.index).to_string(index=False))
        for groups in CUTS:
            labels = fcluster(merges, t=groups, criterion="maxclust")
            members = pd.Series(points.index).groupby(labels).agg(", ".join)
            print(f"Corte en {groups} grupos: " + " | ".join(members))
        tallest_merge = max(tallest_merge, merges[-1, 2])
        correlation, _ = cophenet(merges, distances)
        summary.append({"enlace": title, "altura_final": round(merges[-1, 2], 3),
                        "correlacion_cofenetica": round(correlation, 3)})

        dendrogram(merges, labels=list(points.index), ax=axis,
                   color_threshold=0, above_threshold_color=LINE_COLOR)
        axis.set_title(title, color=INK, fontsize=11)
        axis.grid(axis="y", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].set_ylim(0, tallest_merge * 1.05)
    axes[0].set_ylabel("Distancia de unión", color=INK)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "dendrogramas.png", dpi=150)
    plt.close(figure)

    section("COMPARACION DE LOS TRES ENLACES")
    print(pd.DataFrame(summary).to_string(index=False))


def main():
    PLOTS_DIR.mkdir(exist_ok=True)
    build_dendrograms(load_points())


if __name__ == "__main__":
    main()
