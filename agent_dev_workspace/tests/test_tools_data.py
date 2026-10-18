import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools as make_data_tools


def test_tokenize_tool():
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

    data_tools = make_data_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    result = tokenize_tool.invoke({})

    # Known answer 10 in TEST_FIXTURE.md:
    expected_unigrams = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],
    ]

    assert result["n_documents"] == 8
    assert session.dataframe["unigrams"].tolist() == expected_unigrams


def test_tools_data_remaining_known_answers():
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

    tools = {t.name: t for t in make_data_tools(session)}

    # Known Answer #11: list_files_tool
    res_list = tools["list_files_tool"].invoke({"subdirectory": "newdataset"})
    assert res_list["directories"] == []
    assert "Reddit-stock-sentiment.csv" in res_list["files"]

    # inspect_data_tool (Custom design check)
    res_inspect = tools["inspect_data_tool"].invoke({"n": 3})
    assert res_inspect["n_rows"] == 8
    assert len(res_inspect["head"]) == 3

    # Known Answer #2: check_missing_tool
    res_missing = tools["check_missing_tool"].invoke({})
    assert res_missing["n_missing"] == 1
    assert res_missing["missing_indices"] == [7]

    # Known Answer #3: check_duplicates_tool (without drop and with drop)
    res_dup_check = tools["check_duplicates_tool"].invoke({"drop": False})
    assert res_dup_check["n_duplicates"] == 1
    assert res_dup_check["n_documents_remaining"] == 8

    res_dup_drop = tools["check_duplicates_tool"].invoke({"drop": True})
    assert res_dup_drop["n_duplicates"] == 1
    assert res_dup_drop["n_documents_remaining"] == 6

    # Reload session for sample_data_tool
    session = SessionState()
    load_dataset_tool = next(t for t in load_workspace_tools("premade_tools", session)[0] if t.name == "load_dataset_tool")
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )
    tools = {t.name: t for t in make_data_tools(session)}

    # Known Answer #13: sample_data_tool
    res_sample = tools["sample_data_tool"].invoke({"n": 4, "random_state": 42})
    assert res_sample["sampled_indices"] == [1, 5, 0, 7]
    assert len(session.dataframe) == 4

    # Reload session for describe_data_tool on original 8-row fixture
    session = SessionState()
    load_dataset_tool = next(t for t in load_workspace_tools("premade_tools", session)[0] if t.name == "load_dataset_tool")
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )
    tools = {t.name: t for t in make_data_tools(session)}

    # Known Answer #18: describe_data_tool
    res_desc = tools["describe_data_tool"].invoke({})
    assert session.pending_figure is not None
    assert np.isclose(res_desc["overall_stats"]["mean"], 19.875, atol=1e-3)
    assert np.isclose(res_desc["overall_stats"]["std"], 9.4330, atol=1e-3)
    assert np.isclose(res_desc["category_stats"]["catA"]["mean"], 17.25, atol=1e-3)
    assert np.isclose(res_desc["category_stats"]["catB"]["mean"], 22.5, atol=1e-3)
