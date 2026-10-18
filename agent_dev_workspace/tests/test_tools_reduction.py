import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def test_reduce_dimensions_pca():
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

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    result = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})

    # Known Answer 7:
    # EVR: [0.7532, 0.1965]
    # Doc 0: [2.3669, 0.8871]
    assert result["method"] == "pca"
    assert result["explained_variance_ratio"] == pytest.approx([0.7532, 0.1965], abs=1e-3)
    assert session.pending_figure is not None

    coords = np.array(result["coordinates"])
    assert coords[0] == pytest.approx([2.3669, 0.8871], abs=1e-2)


def test_reduce_dimensions_tsne():
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

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    result = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "perplexity": 2,
        "random_state": 42,
    })

    # Known Answer 8: Doc 0: [1330.9991, 32.1039]
    assert result["method"] == "tsne"
    assert session.pending_figure is not None

    coords = np.array(result["coordinates"])
    assert coords[0] == pytest.approx([1330.9991, 32.1039], abs=1e-1)


def test_reduce_dimensions_umap():
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

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    result = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42,
    })

    # Known Answer 9: Doc 0: [6.6811, -6.8218]
    assert result["method"] == "umap"
    assert session.pending_figure is not None

    coords = np.array(result["coordinates"])
    assert coords[0] == pytest.approx([6.6811, -6.8218], abs=1e-1)


def test_binarize_labels_tool():
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

    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    result = binarize_labels_tool.invoke({})

    assert "classes" in result
    assert sorted(result["classes"]) == ["catA", "catB"]
    assert result["shape"][0] == 8
    assert "binarized_labels" in session.artifacts
