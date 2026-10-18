from typing import Optional
import os
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def tokenize_tool() -> dict:
        """Tokenize each document's text in session.dataframe into lowercased unigram tokens."""
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "No dataframe or 'text' column found in session."}

        unigrams_list = []
        for val in session.dataframe["text"]:
            if val is None or (isinstance(val, float) and val != val):  # Check NaN
                text_str = ""
            else:
                text_str = str(val).strip()

            if not text_str:
                unigrams_list.append([])
            else:
                tokens = nltk.word_tokenize(text_str.lower())
                unigrams_list.append(tokens)

        session.dataframe["unigrams"] = unigrams_list

        total_tokens = sum(len(toks) for toks in unigrams_list)
        result_id = session.next_result_id("tokenize")

        summary = {
            "result_id": result_id,
            "n_documents": len(unigrams_list),
            "total_tokens": total_tokens,
            "sample_unigrams": unigrams_list[0] if unigrams_list else [],
        }

        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = ".") -> dict:
        """List files and directories in the workspace or a specified subdirectory."""
        target_dir = os.path.abspath(subdirectory)
        if not os.path.exists(target_dir):
            return {"error": f"Directory not found: {subdirectory}"}

        entries = os.listdir(target_dir)
        dirs = [e for e in entries if os.path.isdir(os.path.join(target_dir, e))]
        files = [e for e in entries if os.path.isfile(os.path.join(target_dir, e))]

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": sorted(dirs),
            "files": sorted(files),
        }
        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def inspect_data_tool(n: int = 5) -> dict:
        """Peek at the first n rows of the working session.dataframe."""
        if session.dataframe is None:
            return {"error": "No dataframe found in session."}

        df = session.dataframe
        head_records = df.head(n).to_dict(orient="records")

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "head": head_records,
        }
        session.store_result("inspect_data_tool", {"n": n}, summary)
        return summary

    @tool
    def check_missing_tool() -> dict:
        """Check for missing or empty text values in session.dataframe."""
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "No dataframe or 'text' column found in session."}

        missing_mask = session.dataframe["text"].isna() | (session.dataframe["text"].astype(str).str.strip() == "")
        n_missing = int(missing_mask.sum())
        missing_indices = session.dataframe.index[missing_mask].tolist()

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "n_documents": len(session.dataframe),
            "n_missing": n_missing,
            "missing_indices": missing_indices,
        }
        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> dict:
        """Check for and optionally drop duplicate text documents in session.dataframe."""
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "No dataframe or 'text' column found in session."}

        # Identify duplicates
        dup_mask = session.dataframe.duplicated(subset=["text"], keep=False)
        n_duplicates = int(session.dataframe.duplicated(subset=["text"], keep="first").sum())

        if drop and n_duplicates > 0:
            session.dataframe = session.dataframe.drop_duplicates(subset=["text"], keep=False).reset_index(drop=True)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "n_duplicates": n_duplicates,
            "n_documents_remaining": len(session.dataframe),
            "dropped": drop,
        }
        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(n: int = 4, random_state: Optional[int] = None) -> dict:
        """Subsample n rows from session.dataframe with optional random_state."""
        if session.dataframe is None:
            return {"error": "No dataframe found in session."}

        sampled_df = session.dataframe.sample(n=n, random_state=random_state)
        sample_indices = sampled_df.index.tolist()

        # Update active working dataframe in session
        session.dataframe = sampled_df.reset_index(drop=True)

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "sampled_indices": sample_indices,
            "n_samples": len(session.dataframe),
        }
        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    @tool
    def describe_data_tool() -> dict:
        """Compute text character length stats and boxplot grouped by category."""
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "No dataframe or 'text' column found in session."}

        df = session.dataframe.copy()
        df["text_length"] = df["text"].fillna("").astype(str).apply(len)
        session.dataframe["text_length"] = df["text_length"]

        desc_overall = df["text_length"].describe().to_dict()

        desc_by_category = {}
        if "category_name" in df.columns:
            for cat_name, group in df.groupby("category_name"):
                desc_by_category[str(cat_name)] = group["text_length"].describe().to_dict()

        # Build boxplot
        fig, ax = plt.subplots(figsize=(8, 5))
        if "category_name" in df.columns:
            sns.boxplot(data=df, x="category_name", y="text_length", ax=ax)
            ax.set_title("Text Length by Category")
        else:
            sns.boxplot(data=df, y="text_length", ax=ax)
            ax.set_title("Text Length Overall")

        plt.tight_layout()
        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall_stats": {k: float(v) for k, v in desc_overall.items()},
            "category_stats": {
                cat: {k: float(v) for k, v in stats.items()}
                for cat, stats in desc_by_category.items()
            },
        }
        session.store_result("describe_data_tool", {}, summary)
        return summary

    return [
        tokenize_tool,
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
    ]
