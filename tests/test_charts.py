from __future__ import annotations

from dataxid_syntheval._report._charts import diff_heatmap, grouped_bar, overlay_histogram


class TestOverlayHistogram:
    def test_returns_html_string(self):
        bins = [
            {"breakpoint": 10.0, "original": 5, "synthetic": 3},
            {"breakpoint": 20.0, "original": 8, "synthetic": 10},
        ]
        html = overlay_histogram("hist-1", bins, title="Age")
        assert isinstance(html, str)
        assert "hist-1" in html

    def test_has_lazy_data_option(self):
        bins = [{"breakpoint": 1.0, "original": 1, "synthetic": 2}]
        html = overlay_histogram("h1", bins)
        assert "echart-lazy" in html
        assert "data-option" in html

    def test_contains_both_series(self):
        bins = [{"breakpoint": 1.0, "original": 1, "synthetic": 2}]
        html = overlay_histogram("h2", bins)
        assert "Original" in html
        assert "Synthetic" in html

    def test_custom_label_key(self):
        bins = [{"bucket": "A", "original": 1, "synthetic": 2}]
        html = overlay_histogram("h3", bins, label_key="bucket")
        assert "A" in html


class TestGroupedBar:
    def test_returns_html_string(self):
        items = [
            {"value": "Istanbul", "original": 20, "synthetic": 18},
            {"value": "Ankara", "original": 15, "synthetic": 16},
        ]
        html = grouped_bar("bar-1", items, title="City")
        assert isinstance(html, str)
        assert "bar-1" in html

    def test_contains_both_series(self):
        items = [{"value": "X", "original": 1, "synthetic": 2}]
        html = grouped_bar("b2", items)
        assert "Original" in html
        assert "Synthetic" in html


class TestDiffHeatmap:
    def test_returns_html_string(self):
        html = diff_heatmap(
            "hm-1",
            x_labels=["a", "b"],
            y_labels=["a", "b"],
            data=[[0.0, 0.1], [-0.1, 0.0]],
            title="Pearson Diff",
        )
        assert isinstance(html, str)
        assert "hm-1" in html

    def test_has_lazy_data_option(self):
        html = diff_heatmap("hm2", ["a"], ["a"], [[0.0]])
        assert "echart-lazy" in html
        assert "data-option" in html

    def test_contains_heatmap_type(self):
        html = diff_heatmap("hm3", ["a", "b"], ["a", "b"], [[0.0, 0.1], [0.1, 0.0]])
        assert "heatmap" in html

    def test_custom_value_range(self):
        html = diff_heatmap(
            "hm4", ["a"], ["a"], [[0.5]], value_range=(-2.0, 2.0)
        )
        assert "-2.0" in html or "-2" in html

    def test_has_fn_attr_for_tooltip(self):
        html = diff_heatmap("hm5", ["a"], ["a"], [[0.0]])
        assert 'data-has-fn="1"' in html
