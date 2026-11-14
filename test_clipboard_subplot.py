import unittest
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
from PIL import Image

import plot_sgm_bsky_data as plotting


class ClipboardSubplotTests(unittest.TestCase):
    def setUp(self):
        plt.switch_backend("Agg")
        self.addCleanup(plt.close, "all")
        self.fig, (self.left, self.other) = plt.subplots(1, 2, figsize=(12, 4))
        self.left.plot([100., 101., 102.], [1., 2., 3.])
        self.left.set_yticks([1., 2., 3.])
        self.left.set_xlabel("Energy (eV)")
        self.left.set_ylabel("Raw Au Mesh I0 and TEY (Counts)")
        self.left.set_title("Au Mesh I0 and TEY")
        self.right = self.left.twinx()
        self.right.plot([100., 101., 102.], [6000., 8000., 11000.])
        self.right.set_yticks([6000., 8000., 11000.])
        self.right.set_ylabel("Raw External I0 (Counts)")
        self.other.plot([100., 101., 102.], [2., 3., 4.])
        self.other.set_title("Unrelated spectrum")
        self.fig.subplots_adjust(wspace=0.7)
        self.fig.canvas.draw()

    def copy(self, clicked_axis):
        clipboard = Mock(CF_DIB=8)
        with (
            patch.object(plotting, "HAS_CLIPBOARD", True),
            patch.object(plotting, "win32clipboard", clipboard, create=True),
            patch.object(plotting, "Image", Image, create=True),
            patch.object(self.fig, "savefig", wraps=self.fig.savefig) as save,
        ):
            plotting._copy_fig_to_clipboard(self.fig, clicked_axis)
        clipboard.SetClipboardData.assert_called_once()
        self.assertGreater(len(clipboard.SetClipboardData.call_args.args[1]), 1000)
        return save.call_args.kwargs["bbox_inches"]

    def test_double_click_on_twin_copies_both_scales_and_labels(self):
        x, y = self.left.transAxes.transform((0.5, 0.5))
        event = MouseEvent("button_press_event", self.fig.canvas, x, y, button=1, dblclick=True)
        self.assertIs(event.inaxes, self.right)
        extent = self.copy(event.inaxes)
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        for axis in (self.left, self.right):
            for artist in (axis.title, axis.xaxis.label, axis.yaxis.label,
                           axis.yaxis.get_offset_text(), *axis.get_yticklabels()):
                if artist.get_visible() and artist.get_text():
                    bbox = artist.get_window_extent(renderer).transformed(self.fig.dpi_scale_trans.inverted())
                    self.assertLessEqual(extent.x0, bbox.x0 + 1e-8)
                    self.assertGreaterEqual(extent.x1, bbox.x1 - 1e-8)
                    self.assertLessEqual(extent.y0, bbox.y0 + 1e-8)
                    self.assertGreaterEqual(extent.y1, bbox.y1 - 1e-8)
        unrelated = self.other.get_window_extent(renderer).transformed(self.fig.dpi_scale_trans.inverted())
        self.assertLess(extent.x1, unrelated.x0)

    def test_copying_primary_axis_also_includes_right_axis(self):
        extent = self.copy(self.left)
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        right_bbox = self.right.get_tightbbox(renderer).transformed(self.fig.dpi_scale_trans.inverted())
        self.assertGreaterEqual(extent.x1, right_bbox.x1 - 1e-8)

    def test_single_axis_and_full_figure_copy_keep_existing_crops(self):
        renderer = self.fig.canvas.get_renderer()
        expected = self.other.get_tightbbox(renderer).transformed(self.fig.dpi_scale_trans.inverted())
        extent = self.copy(self.other)
        for actual, wanted in zip(extent.extents, expected.extents):
            self.assertAlmostEqual(actual, wanted)
        self.assertEqual(self.copy(None), "tight")


if __name__ == "__main__":
    unittest.main()
