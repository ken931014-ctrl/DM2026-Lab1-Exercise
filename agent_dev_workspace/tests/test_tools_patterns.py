import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools


def test_mine_patterns_fpgrowth_and_zero_terms():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # Known Answer 17: filtering_method="variance" keeps zero terms for both categories
    res_var = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })
    assert res_var["n_patterns"] == 0
    assert res_var["note"] == "no terms survived filtering"

    # Known Answer 17: filtering_method="term_frequency" keeps zero terms for catA
    res_tf = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "term_frequency",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })
    assert res_tf["n_patterns"] == 0
    assert res_tf["note"] == "no terms survived filtering"

    # Known Answer 17: filtering_method="tfidf" keeps alpha for catA at min_sup=1
    res_tfidf_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })
    assert res_tfidf_catA["n_patterns"] == 1
    pats_catA = res_tfidf_catA["patterns"]
    assert pats_catA[0]["pattern"] == ["alpha"]
    assert pats_catA[0]["support"] == 3

    # Known Answer 17: catA + tfidf + fpgrowth + min_sup=4 returns 0 patterns (alpha support is 3)
    res_tfidf_catA_sup4 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4,
    })
    assert res_tfidf_catA_sup4["n_patterns"] == 0

    # Known Answer 17: filtering_method="tfidf" keeps gamma for catB
    res_tfidf_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })
    assert res_tfidf_catB["n_patterns"] == 1
    pats_catB = res_tfidf_catB["patterns"]
    assert pats_catB[0]["pattern"] == ["gamma"]
    assert pats_catB[0]["support"] == 4


def test_mine_patterns_topk_and_maxfpgrowth():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # Known Answer 20: topk at k=1 returns {alpha} support 3 for catA
    res_topk = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1,
    })
    assert res_topk["n_patterns"] == 1
    pats_topk = res_topk["patterns"]
    assert pats_topk[0]["pattern"] == ["alpha"]
    assert pats_topk[0]["support"] == 3

    # Known Answer 20: maxfpgrowth returns 0 patterns for catA at min_sup=1
    res_maxfp = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1,
    })
    assert res_maxfp["n_patterns"] == 0
