from typing import List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        ngram_range: List[int] = [1, 1],
        stop_words: Optional[str] = None,
    ) -> dict:
        """Build a Document-Term Matrix (DTM) from text in session.dataframe.

        Args:
            max_features: Maximum number of top terms by frequency to include in vocabulary.
            min_df: Minimum document frequency threshold (count or fraction).
            max_df: Maximum document frequency threshold (count or fraction).
            ngram_range: Lower and upper boundary of range of n-values for n-grams, e.g. [1, 1].
            stop_words: Language for stop words filtering, e.g. 'english' or None.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "No dataframe or 'text' column loaded in session."}

        texts = session.dataframe["text"].fillna("").astype(str).tolist()

        ngram_tuple = (ngram_range[0], ngram_range[1]) if len(ngram_range) >= 2 else (1, 1)

        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_tuple,
            stop_words=stop_words,
        )

        dtm = vectorizer.fit_transform(texts)
        feature_names = list(vectorizer.get_feature_names_out())

        session.feature_matrix = dtm
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)
        elif "category" in session.dataframe.columns:
            session.set_labels(session.dataframe["category"].values)

        n_rows, n_cols = dtm.shape
        non_zero = int(dtm.nnz)
        total_elements = int(n_rows * n_cols)
        sparsity_pct = float(100.0 * (1.0 - non_zero / total_elements)) if total_elements > 0 else 0.0

        result_id = session.next_result_id("dtm")

        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "sample_terms": feature_names[:10],
        }

        args = {
            "max_features": max_features,
            "min_df": min_df,
            "max_df": max_df,
            "ngram_range": ngram_range,
            "stop_words": stop_words,
        }

        session.store_result("build_dtm_tool", args, summary)
        return summary

    @tool
    def term_frequency_tool(top_n: Optional[int] = None) -> dict:
        """Calculate aggregate term frequencies across all documents in the DTM.

        Args:
            top_n: Optional limit on the number of top terms to return in the summary.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No Document-Term Matrix found in session. Run build_dtm_tool first."}

        counts = np.asarray(session.feature_matrix.sum(axis=0)).ravel()
        term_freqs = list(zip(session.feature_names, counts.tolist()))
        term_freqs.sort(key=lambda x: x[1], reverse=True)

        df_report = pd.DataFrame(term_freqs, columns=["term", "frequency"])

        result_id = session.next_result_id("term_freq")

        top_slice = term_freqs[:top_n] if top_n is not None else term_freqs

        summary = {
            "result_id": result_id,
            "total_terms": len(session.feature_names),
            "term_frequencies": dict(top_slice),
        }

        session.store_result("term_frequency_tool", {"top_n": top_n}, summary, full_report=df_report)
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> dict:
        """Generate a heatmap plot of a positional slice of the global Document-Term Matrix.

        Args:
            n_terms: Number of terms (columns) to include in the slice.
            n_documents: Number of documents (rows) to include in the slice.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No Document-Term Matrix found in session. Run build_dtm_tool first."}

        n_rows, n_cols = session.feature_matrix.shape
        slice_docs = min(n_documents, n_rows)
        slice_terms = min(n_terms, n_cols)

        sub_matrix_arr = session.feature_matrix[:slice_docs, :slice_terms].toarray()
        sub_terms = session.feature_names[:slice_terms]
        doc_labels = [f"Doc {i}" for i in range(slice_docs)]

        # Generate heatmap plot
        fig, ax = plt.subplots(figsize=(max(6, slice_terms * 0.5), max(4, slice_docs * 0.4)))
        sns.heatmap(
            sub_matrix_arr,
            annot=True,
            fmt="d",
            xticklabels=sub_terms,
            yticklabels=doc_labels,
            cmap="YlGnBu",
            ax=ax,
        )
        ax.set_title(f"Document-Term Matrix Heatmap (Top {slice_docs} Docs x {slice_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")

        summary = {
            "result_id": result_id,
            "n_documents": slice_docs,
            "n_terms": slice_terms,
            "matrix": sub_matrix_arr.tolist(),
            "terms": sub_terms,
            "documents": doc_labels,
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary,
        )
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
