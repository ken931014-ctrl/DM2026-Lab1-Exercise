import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def test_exploration_tools_known_answers():
    session = SessionState()

    # Load dataset using premade tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Build DTM using student tool to populate session.feature_matrix and session.feature_names
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Initialize exploration tools
    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(
        t for t in exploration_tools if t.name == "cosine_similarity_tool"
    )
    feature_correlation_matrix_tool = next(
        t for t in exploration_tools if t.name == "feature_correlation_matrix_tool"
    )

    # 1. Test Cosine Similarity Tool (Known Answer #16)
    # doc 0 vs 1: 0.9847
    res_0_1 = cosine_similarity_tool.invoke({"doc1_idx": 0, "doc2_idx": 1})
    assert np.isclose(res_0_1["similarity"], 0.9847, atol=1e-3)

    # doc 0 vs 3: 0.0909
    res_0_3 = cosine_similarity_tool.invoke({"doc1_idx": 0, "doc2_idx": 3})
    assert np.isclose(res_0_3["similarity"], 0.0909, atol=1e-3)

    # doc 0 vs 7: 0.0
    res_0_7 = cosine_similarity_tool.invoke({"doc1_idx": 0, "doc2_idx": 7})
    assert np.isclose(res_0_7["similarity"], 0.0, atol=1e-3)

    # 2. Test Feature Correlation Matrix Tool (Known Answer #19)
    res_corr = feature_correlation_matrix_tool.invoke({"top_n": 20})
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res_corr["terms"] == expected_terms

    # Check alpha vs beta correlation (alpha is idx 0, beta is idx 3 in top_names)
    alpha_idx = res_corr["terms"].index("alpha")
    beta_idx = res_corr["terms"].index("beta")
    alpha_beta_corr = res_corr["correlation_matrix"][alpha_idx][beta_idx]
    assert np.isclose(alpha_beta_corr, 0.8885, atol=1e-3)
