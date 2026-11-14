import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import numpy as np
from PIL import Image

import cluster_xanes_analysis as clustering


class ClusterMapExportTests(unittest.TestCase):
    def setUp(self):
        plt.switch_backend("Agg")
        self.addCleanup(plt.close, "all")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cluster_map = np.array([[0, 2, -1], [1, 0, 2]], dtype=np.int32)
        self.x = np.array([3., 2., 1.])
        self.y = np.array([5., 7.])

    def test_map_images_preserve_labels_palette_coordinates_and_original_shape(self):
        clustering._save_cluster_map_images(
            self.cluster_map, self.x, self.y, 3, "sdd1", str(self.root), "scan",
        )
        expected = self.cluster_map + 1
        cmap = plt.colormaps["tab10"].resampled(3)
        for extension in ("png", "tiff"):
            with self.subTest(extension=extension):
                with Image.open(self.root / f"scan_sdd1_cluster_map.{extension}") as image:
                    self.assertEqual(image.size, (3, 2))
                    np.testing.assert_array_equal(np.array(image), expected)
                    description = image.info["Description"] if extension == "png" else image.tag_v2[270]
                    metadata = json.loads(description)
                    self.assertEqual(metadata["x_mm"], self.x.tolist())
                    self.assertEqual(metadata["y_mm"], self.y.tolist())
                    self.assertEqual(metadata["dataset"], "sdd1")
                    self.assertIn("0 = masked", metadata["label_encoding"])
                    if extension == "png":
                        rgba = np.array(image.convert("RGBA"))
                        np.testing.assert_array_equal(rgba[0, 0], cmap(0, bytes=True))
                        np.testing.assert_array_equal(rgba[1, 0], cmap(1, bytes=True))
                        self.assertEqual(rgba[0, 2, 3], 0)
                    else:
                        self.assertEqual(np.array(image).dtype, np.dtype("uint16"))

    def test_more_than_255_clusters_do_not_truncate_png_or_tiff_values(self):
        cluster_map = np.array([[0, 255, 300, -1]], dtype=np.int32)
        clustering._save_cluster_map_images(
            cluster_map, np.arange(4.), np.array([0.]), 301, "sdd2", str(self.root), "scan",
        )
        for extension in ("png", "tiff"):
            with Image.open(self.root / f"scan_sdd2_cluster_map.{extension}") as image:
                np.testing.assert_array_equal(np.array(image), cluster_map + 1)

    def test_individual_and_comparison_figures_keep_previews_and_copy_whole_figure(self):
        import plot_sgm_bsky_data as plotting

        spectra = np.array([[1., 2., 3.], [3., 2., 1.], [2., 4., 1.]])
        h5_path = str(self.root / "scan.h5")
        with (
            patch.object(clustering, "_display_scrollable_figure", return_value=False),
            patch.object(clustering.plt, "show"),
        ):
            clustering.plot_results(
                self.x, self.y, np.arange(3.), self.cluster_map, spectra,
                "sdd1", str(self.root), "scan",
            )
            clustering.plot_multi_cluster_results([{
                "dataset": "sdd1", "cluster_map": self.cluster_map,
                "cluster_spectra": spectra, "energy": np.arange(3.),
                "x": self.x, "y": self.y,
            }], h5_path, i0_info="Yes")
        figures = [plt.figure(number) for number in plt.get_fignums()]
        self.assertEqual(len(figures), 3)
        for fig in figures:
            fig.canvas.draw()
            x, y = fig.axes[0].transAxes.transform((0.5, 0.5))
            with patch.object(plotting, "_copy_fig_to_clipboard") as copy:
                single = MouseEvent("button_press_event", fig.canvas, x, y, button=1)
                fig.canvas.callbacks.process("button_press_event", single)
                copy.assert_not_called()
                double = MouseEvent("button_press_event", fig.canvas, x, y, button=1, dblclick=True)
                fig.canvas.callbacks.process("button_press_event", double)
                copy.assert_called_once_with(fig)
        for name in ("scan_sdd1_cluster_preview.png", "scan_cluster_comparison_maps.png",
                     "scan_cluster_comparison_spectra.png"):
            self.assertTrue((self.root / name).exists())

    def test_widget_display_keeps_live_canvas_open_inside_scrollable_container(self):
        from ipympl.backend_nbagg import Canvas

        fig = plt.figure()
        canvas = Canvas(fig)
        self.addCleanup(canvas.close)
        with (
            patch("IPython.get_ipython", return_value=object()),
            patch("IPython.display.display") as display,
            patch.object(clustering.plt, "close") as close,
        ):
            self.assertTrue(clustering._display_scrollable_figure(fig))
        container = display.call_args.args[0]
        self.assertIsInstance(container, widgets.Box)
        self.assertEqual(container.children, (canvas,))
        self.assertEqual(container.layout.overflow, "auto")
        self.assertIsNotNone(canvas.comm)
        close.assert_not_called()

    def test_static_notebook_explains_how_to_enable_double_click(self):
        fig = plt.figure()
        output = io.StringIO()
        with (
            patch("IPython.get_ipython", return_value=object()),
            patch("IPython.display.display"),
            contextlib.redirect_stdout(output),
        ):
            self.assertTrue(clustering._display_scrollable_figure(fig))
        self.assertIn("%matplotlib widget", output.getvalue())


if __name__ == "__main__":
    unittest.main()
