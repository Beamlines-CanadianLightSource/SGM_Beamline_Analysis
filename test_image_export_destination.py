from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import ipywidgets as widgets
import numpy as np

import plot_sgm_bsky_data as plotting


def find_widget(widget, description):
    if getattr(widget, "description", None) == description:
        return widget
    for child in getattr(widget, "children", ()):
        match = find_widget(child, description)
        if match is not None:
            return match
    return None


class ImageExportDestinationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.sync = SimpleNamespace(
            current_roi=None, current_poly=None, mode="rect",
            status_widget=widgets.Label(), dashboards=[],
            use_color=True, contrast_percentiles=(0, 100), use_log=False,
            all_energies=[1000.]
        )
        self.context = {
            "save_dir": str(self.base / "raw" / "date"), "scan_name": "map_one",
            "x_coords": np.array([0., 1., 0., 1.]),
            "y_coords": np.array([0., 0., 1., 1.]),
            "x_trim": 0., "y_trim": 0., "as_scatter_plot": True
        }
        self.summary = plotting.SummaryDashboard(self.sync, self.context)
        self.headless = patch.object(plotting, "HAS_GUI", False)
        self.headless.start()
        self.addCleanup(self.headless.stop)
        plotting.plt.switch_backend("Agg")

    def test_map_specific_defaults_separate_maps_without_repeating_scan_folder(self):
        expected = self.base / "preprocessed" / "date" / "map_one" / "Images"
        self.assertEqual(self.summary.default_image_export_directory(), str(expected))
        self.context["scan_name"] = "map_two"
        self.assertNotEqual(self.summary.default_image_export_directory(), str(expected))
        self.context["save_dir"] = str(expected.parent)
        self.context["scan_name"] = "map_one"
        self.assertEqual(self.summary.default_image_export_directory(), str(expected))

    def test_prompt_and_cancel_do_not_write_or_export(self):
        with patch.object(self.summary, "save_all_images") as export:
            self.summary.request_image_export()
            self.assertEqual(
                find_widget(self.summary.image_export_panel, "Save folder:").value,
                self.summary.default_image_export_directory()
            )
            find_widget(self.summary.image_export_panel, "Cancel").click()
            export.assert_not_called()
        self.assertEqual(self.summary.image_export_panel.children, ())
        self.assertFalse((self.base / "preprocessed").exists())

    def test_confirmation_writes_real_png_and_tiff_to_edited_preprocessed_path(self):
        row = SimpleNamespace(
            name="mcc1", is_mcc=True, sync=self.sync, ctx=self.context,
            s2d_rep=np.array([1., 2., 3., 4.]), rep_e=1000.
        )
        row.export_images = lambda output_dir: plotting.DashboardRow.export_images(row, output_dir)
        self.sync.dashboards = [row]
        self.summary.request_image_export()
        chosen = self.base / "raw" / "date" / "custom_images"
        find_widget(self.summary.image_export_panel, "Save folder:").value = str(chosen)
        self.assertFalse(chosen.exists())
        with patch.object(plotting, "get_tk_root") as get_root:
            find_widget(self.summary.image_export_panel, "Save Images").click()
            get_root.assert_not_called()
        expected = self.base / "preprocessed" / "date" / "custom_images"
        self.assertTrue((expected / "map_one_mcc1_1000.00eV.png").is_file())
        self.assertTrue((expected / "map_one_mcc1_1000.00eV.tiff").is_file())
        self.assertFalse(chosen.exists())
        self.assertIn(str(expected), self.sync.status_widget.value)
        self.assertEqual(self.summary.image_export_panel.children, ())

    def test_invalid_or_unwritable_destination_keeps_prompt_open(self):
        self.summary.request_image_export()
        path_box = find_widget(self.summary.image_export_panel, "Save folder:")
        path_box.value = ""
        self.assertTrue(find_widget(self.summary.image_export_panel, "Save Images").disabled)
        path_box.value = str(self.base / "chosen")
        with patch.object(plotting.os, "makedirs", side_effect=PermissionError("Read-only folder")):
            find_widget(self.summary.image_export_panel, "Save Images").click()
        self.assertIn("Read-only folder", self.sync.status_widget.value)
        self.assertIsNotNone(find_widget(self.summary.image_export_panel, "Save Images"))

    def test_detector_failure_does_not_report_success(self):
        row = Mock(name="detector")
        row.name = "sdd1"
        row.export_images.side_effect = OSError("Disk full")
        self.sync.dashboards = [row]
        self.assertFalse(self.summary.save_all_images())
        self.assertIn("FAILED", self.sync.status_widget.value)
        self.assertIn("Disk full", self.sync.status_widget.value)

    def test_automatic_hpc_export_uses_default_and_desktop_keeps_original_directory(self):
        row = Mock()
        row.name = "sdd1"
        self.sync.dashboards = [row]
        self.assertTrue(self.summary.save_all_images())
        row.export_images.assert_called_once_with(
            output_dir=self.summary.default_image_export_directory()
        )
        row.export_images.reset_mock()
        with (
            patch.object(plotting, "HAS_GUI", True),
            patch.object(plotting, "get_tk_root", return_value=None),
            patch.object(plotting, "safe_messagebox_showinfo"),
        ):
            self.summary.request_image_export()
        row.export_images.assert_called_once_with(
            output_dir=str(Path(self.context["save_dir"]) / "Images")
        )
        self.assertEqual(self.summary.image_export_panel.children, ())


if __name__ == "__main__":
    unittest.main()
