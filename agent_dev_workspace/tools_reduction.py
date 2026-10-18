from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelBinarizer
import umap
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: str = "pca",
        n_components: int = 2,
        random_state: Optional[int] = None,
        perplexity: Optional[float] = 30.0,
        n_neighbors: Optional[int] = 15,
    ) -> dict:
        """Reduce dimensionality of the Document-Term Matrix using PCA, t-SNE, or UMAP, and plot the 2D coordinates.

        Args:
            method: Dimensionality reduction technique ('pca', 'tsne', or 'umap').
            n_components: Target dimensions count (default 2).
            random_state: Seed for reproducible output.
            perplexity: Perplexity parameter for t-SNE.
            n_neighbors: Number of neighbors parameter for UMAP.
        """
        if session.feature_matrix is None:
            return {"error": "No Document-Term Matrix found in session. Run build_dtm_tool first."}

        X = session.feature_matrix
        X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

        method_lower = method.lower()
        explained_variance_ratio = None

        if method_lower == "pca":
            model = PCA(n_components=n_components, random_state=random_state)
            coords = model.fit_transform(X_dense)
            explained_variance_ratio = [round(float(v), 4) for v in model.explained_variance_ratio_]
        elif method_lower == "tsne":
            perp = perplexity if perplexity is not None else 30.0
            model = TSNE(n_components=n_components, perplexity=perp, random_state=random_state)
            coords = model.fit_transform(X_dense)
        elif method_lower == "umap":
            nn = n_neighbors if n_neighbors is not None else 15
            model = umap.UMAP(n_components=n_components, n_neighbors=nn, random_state=random_state)
            coords = model.fit_transform(X_dense)
        else:
            return {"error": f"Unknown dimensionality reduction method '{method}'."}

        coords_list = [[round(float(val), 4) for val in row] for row in coords]
        session.artifacts["reduced_coordinates"] = coords_list

        labels = None
        if session.dataframe is not None:
            df = session.dataframe
            cat_col = next((c for c in ["category_name", "category", "label"] if c in df.columns), None)
            if cat_col is not None:
                labels = df[cat_col].astype(str).values

        if labels is None and session.labels is not None:
            labels = np.asarray(session.labels).astype(str)

        fig, ax = plt.subplots(figsize=(7, 5))
        if coords.shape[1] >= 2:
            x_vals = coords[:, 0]
            y_vals = coords[:, 1]
            if labels is not None:
                sns.scatterplot(x=x_vals, y=y_vals, hue=labels, style=labels, s=70, ax=ax)
            else:
                sns.scatterplot(x=x_vals, y=y_vals, s=70, ax=ax)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
        ax.set_title(f"Dimensionality Reduction ({method_lower.upper()})")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("reduce_dimensions")

        summary = {
            "result_id": result_id,
            "method": method_lower,
            "n_components": n_components,
            "coordinates": coords_list,
            "explained_variance_ratio": explained_variance_ratio,
        }

        session.store_result(
            "reduce_dimensions_tool",
            {
                "method": method,
                "n_components": n_components,
                "random_state": random_state,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
            },
            summary,
        )
        return summary

    @tool
    def binarize_labels_tool() -> dict:
        """Binarize category labels into a one-hot or binary matrix format."""
        labels = None
        if session.dataframe is not None:
            df = session.dataframe
            cat_col = next((c for c in ["category_name", "category", "label"] if c in df.columns), None)
            if cat_col is not None:
                labels = df[cat_col].astype(str).values

        if labels is None and session.labels is not None:
            labels = np.asarray(session.labels).astype(str)

        if labels is None:
            return {"error": "No labels found in session dataframe or session labels."}

        lb = LabelBinarizer()
        binarized = lb.fit_transform(labels)
        classes = [str(c) for c in lb.classes_]

        session.artifacts["binarized_labels"] = binarized.tolist()
        session.artifacts["label_classes"] = classes

        result_id = session.next_result_id("binarize_labels")

        summary = {
            "result_id": result_id,
            "classes": classes,
            "shape": list(binarized.shape),
            "binarized_labels": binarized.tolist(),
        }

        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
