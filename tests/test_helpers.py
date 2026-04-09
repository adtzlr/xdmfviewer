from __future__ import annotations

import numpy as np

from xdmfviewer.app import ScalarExtractor, StatusFormatter


def test_status_formatter_builds_multiline_text() -> None:
    text = StatusFormatter.build_status_text(
        "sample.xdmf",
        1,
        3,
        "Min=1.000e+00, Max=2.000e+00",
        "Min=0.000e+00, Max=4.000e+00",
    )

    assert text == (
        "sample.xdmf\n"
        "Step 2/3\n"
        "Min=1.000e+00, Max=2.000e+00\n"
        "Min=0.000e+00, Max=4.000e+00 (All Steps)"
    )


def test_status_formatter_handles_empty_or_non_finite_values() -> None:
    assert StatusFormatter.format_min_max(None) == "n/a"
    assert StatusFormatter.format_min_max(np.array([])) == "n/a"
    assert StatusFormatter.format_min_max(np.array([np.nan, np.inf])) == "n/a"
    assert StatusFormatter.format_min_max(np.array([1.0, 2.0])) == "Min=1.000e+00, Max=2.000e+00"


def test_scalar_extractor_supports_components_and_magnitude() -> None:
    values = np.array([[3.0, 4.0], [1.0, 2.0]])

    np.testing.assert_allclose(ScalarExtractor.extract_component(values, 0), [3.0, 1.0])
    np.testing.assert_allclose(ScalarExtractor.extract_component(values, 1), [4.0, 2.0])
    np.testing.assert_allclose(ScalarExtractor.extract_component(values, -2), [5.0, np.sqrt(5.0)])
    np.testing.assert_allclose(ScalarExtractor.extract_component(values, 99), [3.0, 1.0])