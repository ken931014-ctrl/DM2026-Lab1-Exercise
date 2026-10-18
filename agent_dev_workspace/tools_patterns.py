from typing import Optional, List, Dict, Any
import os
import tempfile
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

import PAMI.frequentPattern.basic.FPGrowth as FPGrowth
import PAMI.frequentPattern.topk.FAE as TopK
import PAMI.frequentPattern.maximal.MaxFPGrowth as MaxFPGrowth


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str = "tfidf",
        algorithm: str = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> dict:
        """Mine frequent patterns for a specific category using PAMI algorithms.

        Args:
            category_name: The target category to mine patterns for.
            filtering_method: Feature selection method ('variance', 'tfidf', or 'term_frequency').
            algorithm: Mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth').
            min_sup: Minimum support threshold (required for 'fpgrowth' and 'maxfpgrowth').
            k: Top-k patterns limit (required for 'topk').
        """
        if session.dataframe is None:
            return {"error": "No dataframe found in session."}

        df = session.dataframe
        cat_col = next((c for c in ["category_name", "category", "label"] if c in df.columns), None)
        if cat_col is None:
            return {"error": "No category or label column found in session.dataframe."}

        # Filter dataframe by category_name
        cat_df = df[df[cat_col].astype(str) == str(category_name)]
        if cat_df.empty:
            return {"error": f"No documents found for category '{category_name}'."}

        texts = cat_df["text"].fillna("").astype(str).tolist()

        # Step 1: Fresh category-local CountVectorizer with English stopwords
        vec = CountVectorizer(stop_words="english")
        try:
            dtm = vec.fit_transform(texts)
        except ValueError:
            dtm = None

        if dtm is None or dtm.shape[1] == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "n_patterns": 0,
                "note": "no terms survived filtering",
            }
            artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        feature_names = np.array(vec.get_feature_names_out())
        n_terms = len(feature_names)

        # Step 2: Apply chosen filtering_method using standard percentile calculation
        if filtering_method == "variance":
            scores = np.asarray(dtm.power(2).mean(axis=0) - np.square(dtm.mean(axis=0))).ravel()
            p5 = np.percentile(scores, 5)
            p95 = np.percentile(scores, 95)
            kept_indices = np.where((scores >= p5) & (scores <= p95))[0].tolist()
        elif filtering_method == "term_frequency":
            scores = np.asarray(dtm.sum(axis=0)).ravel()
            p5 = np.percentile(scores, 5)
            p95 = np.percentile(scores, 95)
            kept_indices = np.where((scores >= p5) & (scores <= p95))[0].tolist()
        elif filtering_method == "tfidf":
            tfidf_mat = TfidfTransformer().fit_transform(dtm)
            scores = np.asarray(tfidf_mat.mean(axis=0)).ravel()
            p20 = np.percentile(scores, 20)
            kept_indices = np.where(scores >= p20)[0].tolist()
        else:
            return {"error": f"Unknown filtering_method '{filtering_method}'."}

        # Step 3: Handle zero terms surviving filtering cleanly
        if len(kept_indices) == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "n_patterns": 0,
                "note": "no terms survived filtering",
            }
            artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        kept_terms = feature_names[kept_indices]
        sub_dtm = dtm[:, kept_indices].toarray()

        # Step 4: Convert directly to transactional database file (\t separated)
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as tmp:
            tmp_path = tmp.name
            for row in sub_dtm:
                items = [kept_terms[j] for j in range(len(kept_terms)) if row[j] >= 1]
                if items:
                    tmp.write("\t".join(items) + "\n")

        try:
            # Step 5: Run PAMI algorithm
            alg_lower = algorithm.lower()

            if alg_lower == "fpgrowth":
                if min_sup is None:
                    return {"error": "min_sup is required for algorithm 'fpgrowth'."}
                miner = FPGrowth.FPGrowth(iFile=tmp_path, minSup=min_sup, sep="\t")
            elif alg_lower == "topk":
                if k is None:
                    return {"error": "k is required for algorithm 'topk'."}
                miner = TopK.FAE(iFile=tmp_path, k=k, sep="\t")
            elif alg_lower == "maxfpgrowth":
                if min_sup is None:
                    return {"error": "min_sup is required for algorithm 'maxfpgrowth'."}
                miner = MaxFPGrowth.MaxFPGrowth(iFile=tmp_path, minSup=min_sup, sep="\t")
            else:
                return {"error": f"Unknown algorithm '{algorithm}'."}

            miner.mine()
            df_res = miner.getPatternsAsDataFrame()

            pami_patterns = []
            if df_res is not None and not df_res.empty:
                # Use positional column access (column 0: patterns, column 1: support)
                for _, row in df_res.iterrows():
                    pat_val = str(row.iloc[0])
                    sup_val = int(row.iloc[1])
                    pat_list = pat_val.split("\t") if "\t" in pat_val else pat_val.split()
                    pami_patterns.append({
                        "pattern": pat_list,
                        "support": sup_val,
                    })

        except Exception as e:
            return {"error": f"PAMI pattern mining execution failed: {str(e)}"}
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        result_id = session.next_result_id("patterns")
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"

        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "patterns": pami_patterns,
            "n_patterns": len(pami_patterns),
        }

        session.artifacts[artifact_key] = pami_patterns
        session.store_result(
            "mine_patterns_tool",
            {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k,
            },
            summary,
        )
        return summary

    return [mine_patterns_tool]
