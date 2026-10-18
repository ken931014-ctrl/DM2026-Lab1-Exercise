import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool():
    session = SessionState()

    # Load premade dataset tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Instantiate variance_filter_tool
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Call with threshold 0.15
    result = variance_filter_tool.invoke({"threshold": 0.15})

    # Known answer 5 in TEST_FIXTURE.md:
    # alpha: 1.1875
    # gamma: 1.109375
    # delta: 0.25
    # beta: 0.234375
    # always: 0.109375
    variances = result["variances"]
    assert variances["alpha"] == pytest.approx(1.1875, abs=1e-3)
    assert variances["gamma"] == pytest.approx(1.109375, abs=1e-3)
    assert variances["delta"] == pytest.approx(0.25, abs=1e-3)
    assert variances["beta"] == pytest.approx(0.234375, abs=1e-3)
    assert variances["always"] == pytest.approx(0.109375, abs=1e-3)

    # Threshold 0.15 keeps alpha, beta, delta, gamma; removes always
    assert result["kept_count"] == 4
    assert result["removed_count"] == 1
    assert set(result["kept_terms"]) == {"alpha", "beta", "delta", "gamma"}
    assert result["removed_terms"] == ["always"]


def test_pearson_filter_tool():
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

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    result = pearson_filter_tool.invoke({"target_class": "catB"})

    # Known answer 6 in TEST_FIXTURE.md for target_class="catB":
    # delta: 1.0000
    # gamma: 0.8307
    # always: 0.3780
    # alpha: -0.6882
    # beta: -0.7746
    corrs = result["correlations"]
    assert corrs["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert corrs["gamma"] == pytest.approx(0.8307, abs=1e-3)
    assert corrs["always"] == pytest.approx(0.3780, abs=1e-3)
    assert corrs["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert corrs["beta"] == pytest.approx(-0.7746, abs=1e-3)


def test_spearman_filter_tool():
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

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    result = spearman_filter_tool.invoke({"target_class": "catB"})

    # Known answer 6 in TEST_FIXTURE.md for Spearman correlation (target_class="catB"):
    # delta: 1.0000
    # gamma: 0.9363
    # always: 0.3780
    # alpha: -0.7500
    # beta: -0.7746
    corrs = result["correlations"]
    assert corrs["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert corrs["gamma"] == pytest.approx(0.9363, abs=1e-3)
    assert corrs["always"] == pytest.approx(0.3780, abs=1e-3)
    assert corrs["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert corrs["beta"] == pytest.approx(-0.7746, abs=1e-3)
