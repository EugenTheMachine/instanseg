from types import SimpleNamespace

import numpy as np
import pytest

from instanseg.utils.data_loader import _format_labels, _keep_images


def test_label_formatting_always_uses_cell_masks():
    cell_masks = np.array([[0, 1], [1, 1]], dtype=np.int32)
    item = {
        "image": np.zeros((3, 2, 2), dtype=np.float32),
        "cell_masks": cell_masks,
        "nucleus_masks": np.full((2, 2), 2, dtype=np.int32),
    }

    labels = _format_labels(item)

    np.testing.assert_array_equal(labels, cell_masks)


def test_dataset_filter_rejects_nucleus_only_items():
    args = SimpleNamespace(source_dataset=["all"])
    item = {
        "parent_dataset": "example",
        "nucleus_masks": np.ones((2, 2), dtype=np.int32),
    }

    assert not _keep_images(item, args)


@pytest.mark.parametrize("modality", ["Brightfield", "phase-contrast"])
def test_dataset_filter_accepts_supported_cell_modalities(modality):
    args = SimpleNamespace(source_dataset=["all"])
    item = {
        "parent_dataset": "example",
        "cell_masks": np.ones((2, 2), dtype=np.int32),
        "image_modality": modality,
    }

    assert _keep_images(item, args)


def test_dataset_filter_rejects_fluorescence_even_with_cell_masks():
    args = SimpleNamespace(source_dataset=["all"])
    item = {
        "parent_dataset": "example",
        "cell_masks": np.ones((2, 2), dtype=np.int32),
        "image_modality": "Fluorescence",
    }

    assert not _keep_images(item, args)


def test_dataset_filter_rejects_legacy_mode_argument():
    args = SimpleNamespace(source_dataset=["all"], target_segmentation="C")

    with pytest.raises(ValueError, match="Unsupported segmentation mode arguments"):
        _keep_images({}, args)


def test_dataset_filter_rejects_cells_and_nuclei_argument():
    args = SimpleNamespace(source_dataset=["all"], cells_and_nuclei=True)

    with pytest.raises(ValueError, match="Unsupported segmentation mode arguments"):
        _keep_images({}, args)


def test_format_labels_rejects_target_segmentation_argument():
    item = {"cell_masks": np.zeros((2, 2), dtype=np.int32)}
    with pytest.raises(TypeError):
        _format_labels(item, **{"target_segmentation": "N"})  # pylint: disable=unexpected-keyword-arg