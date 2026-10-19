"""Exploratory linear discriminant analysis of organic nanomaterial descriptors."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from response_models import prepare, read_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/lda"))
    parser.add_argument("--label", default="multiclass_labels")
    parser.add_argument("--group", default="paper_id")
    parser.add_argument("--features", nargs="+")
    parser.add_argument("--drop", nargs="+", default=[])
    parser.add_argument("--color-by", help="Optional numeric column for continuous coloring")
    args = parser.parse_args()
    frame, X = read_data(args.data, args.label, args.group, args.features, args.drop)
    y = frame[args.label]
    counts = y.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("LDA requires at least two classes and at least two rows per class.")
    matrix = prepare(X).fit_transform(X)
    dimension = min(2, len(counts)-1, matrix.shape[1])
    model = LinearDiscriminantAnalysis(n_components=dimension, solver="svd")
    coordinates = model.fit_transform(matrix, y)
    if coordinates.shape[1] == 0:
        raise ValueError("Predictors do not provide an estimable discriminant direction.")
    # A binary endpoint has only one discriminant axis; no artificial LD2 is fitted.
    vertical = coordinates[:, 1] if coordinates.shape[1] > 1 else np.zeros(len(X))
    color = None
    if args.color_by:
        if args.color_by not in frame:
            raise ValueError(f"Missing color column: {args.color_by}")
        color = pd.to_numeric(frame[args.color_by], errors="coerce")
        if color.isna().any() or not np.isfinite(color.to_numpy()).all():
            raise ValueError("The continuous color column must contain finite numeric values.")
    args.out.mkdir(parents=True, exist_ok=False)
    output = pd.DataFrame(coordinates, columns=[f"LD{i+1}" for i in range(coordinates.shape[1])])
    output.insert(0, "row_id", frame.index)
    if args.group in frame:
        output[args.group] = frame[args.group].to_numpy()
    output[args.label] = y.to_numpy()
    output.to_csv(args.out / "discriminant_coordinates.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    if color is not None:
        dots = ax.scatter(coordinates[:, 0], vertical, c=color, cmap="viridis",
                          s=24, alpha=0.65, edgecolors="none")
        fig.colorbar(dots, ax=ax, label=args.color_by, shrink=0.8)
    else:
        for label in counts.index:
            mask = (y == label).to_numpy()
            ax.scatter(coordinates[mask, 0], vertical[mask], label=str(label),
                       s=24, alpha=0.65, edgecolors="none")
        ax.legend(title=args.label, frameon=False)
    ax.set_xlabel("LD1")
    if coordinates.shape[1] > 1:
        ax.set_ylabel("LD2")
    else:
        ax.set_yticks([])
    ax.set_title("Organic materials | exploratory LDA")
    ax.spines[["top", "right"]].set_visible(False)
    for extension in ("png", "svg"):
        fig.savefig(args.out / f"material_projection.{extension}", dpi=250)
    plt.close(fig)
    print(f"Saved {coordinates.shape[1]} LDA axis/axes to {args.out}. "
          "Full-data supervised visualization; not a held-out performance estimate.")


if __name__ == "__main__":
    main()
