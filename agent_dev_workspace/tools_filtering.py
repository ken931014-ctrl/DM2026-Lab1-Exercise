from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.0) -> dict:
        """Filter out low-variance terms from the Document-Term Matrix.

        Args:
            threshold: Terms with variance strictly below this threshold are flagged for removal.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No Document-Term Matrix found in session. Run build_dtm_tool first."}

        X = session.feature_matrix
        # Compute population variance (ddof=0) directly on sparse matrix: E[X^2] - (E[X])^2
        if hasattr(X, "power"):
            means = np.asarray(X.mean(axis=0)).ravel()
            means_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
            variances = means_sq - (means ** 2)
        else:
            variances = np.var(X, axis=0, ddof=0)

        term_variances = list(zip(session.feature_names, variances.tolist()))

        # Separate into kept and removed based on threshold
        kept_terms = [term for term, v in term_variances if v >= threshold]
        removed_terms = [term for term, v in term_variances if v < threshold]

        # Sort full report by variance descending for visualize_result_tool
        sorted_term_variances = sorted(term_variances, key=lambda x: x[1], reverse=True)
        df_report = pd.DataFrame(sorted_term_variances, columns=["term", "variance"])

        result_id = session.next_result_id("variance_filter")

        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "kept_count": len(kept_terms),
            "removed_count": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "variances": {term: round(v, 6) for term, v in term_variances},
        }

        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=df_report,
        )
        return summary

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Compute Pearson correlation between each term's count and a target class indicator.

        Args:
            target_class: The target class name to calculate one-vs-rest correlation against.
            threshold: Minimum absolute Pearson correlation required to keep a term.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No Document-Term Matrix found in session. Run build_dtm_tool first."}

        # Resolve target labels from session.labels or session.dataframe
        if session.labels is not None:
            labels = np.asarray(session.labels)
        elif session.dataframe is not None:
            df = session.dataframe
            cat_col = next((c for c in ["category_name", "category", "label"] if c in df.columns), None)
            if cat_col is None:
                return {"error": "No category or label column found in session.dataframe."}
            labels = df[cat_col].values
        else:
            return {"error": "No labels or dataframe found in session."}

        # Binary indicator vector y for target_class
        y = (labels.astype(str) == str(target_class)).astype(float)
        n_docs = len(y)

        if n_docs != session.feature_matrix.shape[0]:
            return {"error": "Mismatch between labels count and feature_matrix rows."}

        y_mean = y.mean()
        y_std = y.std()

        X = session.feature_matrix

        if y_std == 0:
            pearson_rs = [(term, 0.0) for term in session.feature_names]
        else:
            if hasattr(X, "power"):
                x_means = np.asarray(X.mean(axis=0)).ravel()
                x_means_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
                x_vars = np.maximum(0.0, x_means_sq - (x_means ** 2))
                x_stds = np.sqrt(x_vars)

                # Vectorized E[X * y] via sparse matrix-vector dot product X.T @ y / n_docs
                xy_means = np.asarray(X.T.dot(y)).ravel() / n_docs
                covs = xy_means - (x_means * y_mean)

                with np.errstate(divide="ignore", invalid="ignore"):
                    r_vals = np.where((x_stds > 0) & (y_std > 0), covs / (x_stds * y_std), 0.0)
                    r_vals = np.nan_to_num(r_vals, nan=0.0)

                pearson_rs = list(zip(session.feature_names, r_vals.tolist()))
            else:
                X_arr = np.asarray(X)
                x_stds = X_arr.std(axis=0)
                x_means = X_arr.mean(axis=0)
                covs = np.mean(X_arr * y[:, None], axis=0) - (x_means * y_mean)
                with np.errstate(divide="ignore", invalid="ignore"):
                    r_vals = np.where((x_stds > 0) & (y_std > 0), covs / (x_stds * y_std), 0.0)
                    r_vals = np.nan_to_num(r_vals, nan=0.0)
                pearson_rs = list(zip(session.feature_names, r_vals.tolist()))

        # Filter using absolute correlation threshold
        kept_terms = [term for term, r in pearson_rs if abs(r) >= threshold]
        removed_terms = [term for term, r in pearson_rs if abs(r) < threshold]

        # Sort report by pearson_r descending for visualize_result_tool
        sorted_reports = sorted(pearson_rs, key=lambda x: x[1], reverse=True)
        df_report = pd.DataFrame(sorted_reports, columns=["term", "pearson_r"])

        result_id = session.next_result_id("pearson_filter")

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "kept_count": len(kept_terms),
            "removed_count": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "correlations": {term: round(float(r), 4) for term, r in pearson_rs},
        }

        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=df_report,
        )
        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Compute Spearman rank correlation between each term's count and a target class indicator.

        Args:
            target_class: The target class name to calculate one-vs-rest correlation against.
            threshold: Minimum absolute Spearman rank correlation required to keep a term.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No Document-Term Matrix found in session. Run build_dtm_tool first."}

        # Resolve target labels
        if session.labels is not None:
            labels = np.asarray(session.labels)
        elif session.dataframe is not None:
            df = session.dataframe
            cat_col = next((c for c in ["category_name", "category", "label"] if c in df.columns), None)
            if cat_col is None:
                return {"error": "No category or label column found in session.dataframe."}
            labels = df[cat_col].values
        else:
            return {"error": "No labels or dataframe found in session."}

        # Binary indicator vector y for target_class
        y = (labels.astype(str) == str(target_class)).astype(float)
        n_docs = len(y)

        if n_docs != session.feature_matrix.shape[0]:
            return {"error": "Mismatch between labels count and feature_matrix rows."}

        y_rank = rankdata(y)
        y_rank_std = y_rank.std()

        X = session.feature_matrix

        spearman_rs = []
        for j, term in enumerate(session.feature_names):
            if hasattr(X, "getcol"):
                col = np.asarray(X.getcol(j).toarray()).ravel()
            else:
                col = np.asarray(X[:, j]).ravel()

            col_rank = rankdata(col)
            c_std = col_rank.std()

            if c_std == 0 or y_rank_std == 0:
                rho = 0.0
            else:
                rho = float(np.corrcoef(col_rank, y_rank)[0, 1])
                if np.isnan(rho):
                    rho = 0.0

            spearman_rs.append((term, rho))

        # Filter using absolute rank correlation threshold
        kept_terms = [term for term, r in spearman_rs if abs(r) >= threshold]
        removed_terms = [term for term, r in spearman_rs if abs(r) < threshold]

        # Sort report by spearman_r descending for visualize_result_tool
        sorted_reports = sorted(spearman_rs, key=lambda x: x[1], reverse=True)
        df_report = pd.DataFrame(sorted_reports, columns=["term", "spearman_r"])

        result_id = session.next_result_id("spearman_filter")

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "kept_count": len(kept_terms),
            "removed_count": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "correlations": {term: round(float(r), 4) for term, r in spearman_rs},
        }

        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=df_report,
        )
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
