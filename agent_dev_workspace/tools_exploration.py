from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def cosine_similarity_tool(doc1_idx: int, doc2_idx: int) -> dict:
        """Compute cosine similarity between two documents in session feature matrix."""
        if session.feature_matrix is None:
            return {"error": "No Document-Term Matrix found in session."}

        X = session.feature_matrix
        X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        n_docs = X_dense.shape[0]

        if doc1_idx < 0 or doc1_idx >= n_docs or doc2_idx < 0 or doc2_idx >= n_docs:
            return {"error": f"Invalid document index. Must be between 0 and {n_docs - 1}."}

        v1 = X_dense[doc1_idx].reshape(1, -1)
        v2 = X_dense[doc2_idx].reshape(1, -1)

        if np.all(v1 == 0) or np.all(v2 == 0):
            sim_val = 0.0
        else:
            sim_val = float(cosine_similarity(v1, v2)[0, 0])

        result_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": result_id,
            "doc1_idx": doc1_idx,
            "doc2_idx": doc2_idx,
            "similarity": round(sim_val, 4),
        }
        session.store_result("cosine_similarity_tool", {"doc1_idx": doc1_idx, "doc2_idx": doc2_idx}, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool(top_n: int = 20) -> dict:
        """Compute and plot feature correlation matrix for top variance terms."""
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No Document-Term Matrix found in session."}

        X = session.feature_matrix
        X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        names = np.array(session.feature_names)

        variances = np.var(X_dense, axis=0)
        top_indices = np.argsort(variances)[::-1][:top_n]

        top_X = X_dense[:, top_indices]
        top_names = names[top_indices].tolist()

        corr_matrix = np.corrcoef(top_X, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, xticklabels=top_names, yticklabels=top_names, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Feature Correlation Matrix (Top Variance Terms)")
        plt.tight_layout()

        session.pending_figure = fig
        result_id = session.next_result_id("feature_correlation")

        corr_list = [[round(float(val), 4) for val in row] for row in corr_matrix]

        summary = {
            "result_id": result_id,
            "terms": top_names,
            "correlation_matrix": corr_list,
        }
        session.store_result("feature_correlation_matrix_tool", {"top_n": top_n}, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
