import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import plot_sgm_bsky_data as plotting


def find(widget, description):
    if getattr(widget, "description", None) == description:
        return widget
    for child in getattr(widget, "children", ()):
        result = find(child, description)
        if result is not None:
            return result
    return None


class I0NotebookPreviewTests(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(15, dtype=float)
        self.y = np.array([4., 7., 2., 8., 1., 9., 3., 10., 2., 8., 1., 7., 3., 9., 4.])
        self.finish = Mock()

    def preview(self, external=False):
        return plotting._i0_preview_widget(
            self.x, self.y, "Test I0", external, self.finish
        )

    def test_smoothing_updates_overlaid_preview_and_apply_returns_processed_data(self):
        ui = self.preview()
        image = next(child for child in ui.children if isinstance(child, plotting.widgets.Image))
        original_png = bytes(image.value)
        find(ui, "Smooth").value = True
        self.assertNotEqual(bytes(image.value), original_png)
        self.finish.assert_not_called()
        find(ui, "Apply").click()
        actual = self.finish.call_args.args[0]
        np.testing.assert_allclose(actual[0], plotting.savgol_filter(self.y, 11, 2))
        self.assertIn("Smoothed w=11", actual[1])

    def test_cancel_uses_unmodified_curve_without_applying_settings(self):
        ui = self.preview()
        find(ui, "Smooth").value = True
        find(ui, "Cancel").click()
        result = self.finish.call_args.args[0]
        np.testing.assert_array_equal(result[0], self.y)
        self.assertEqual(result[1:], ("", 0.0, "Add", False, 0.0))

    def test_validation_blocks_bad_window_and_division_by_zero(self):
        ui = self.preview()
        find(ui, "Window:").value = 4
        find(ui, "Smooth").value = True
        self.assertTrue(find(ui, "Apply").disabled)
        find(ui, "Window:").value = 5
        self.assertFalse(find(ui, "Apply").disabled)
        find(ui, "Constant").value = True
        find(ui, "Operation:").value = "Divide"
        self.assertTrue(find(ui, "Apply").disabled)
        find(ui, "Value:").value = 2.
        self.assertFalse(find(ui, "Apply").disabled)
        find(ui, "Operation:").value = "Multiply"
        find(ui, "Value:").value = 0.
        self.assertTrue(find(ui, "Apply").disabled)

    def test_failed_optical_density_cannot_be_applied(self):
        with patch.object(plotting, "HAS_SF", True), patch.object(plotting, "sf", return_value=None):
            ui = self.preview()
            find(ui, "Divide by OD").value = True
            self.assertTrue(find(ui, "Apply").disabled)
            find(ui, "Apply").click()
            self.finish.assert_not_called()
            find(ui, "Cancel").click()
            np.testing.assert_array_equal(self.finish.call_args.args[0][0], self.y)

    def test_external_shift_and_multiplication_are_applied(self):
        ui = self.preview(external=True)
        find(ui, "Constant").value = True
        find(ui, "Operation:").value = "Multiply"
        find(ui, "Value:").value = 2.
        find(ui, "Energy shift").value = True
        find(ui, "Shift (eV):").value = 1.5
        find(ui, "Apply").click()
        result = self.finish.call_args.args[0]
        np.testing.assert_array_equal(result[0], self.y * 2.)
        self.assertEqual(result[2:], (2., "Multiply", True, 1.5))

    def test_bn_correction_csv_header_is_readable_with_windows_encoding(self):
        with patch.object(plotting, "HAS_SF", True), patch.object(plotting, "sf", return_value=np.full(15, 2.)):
            processed, _, _, description = plotting.compute_modified_i0(
                self.x, self.y, False, 11, False, "Add", 0.,
                False, 0., True, "BN", 2.1, 0.1, False
            )
        np.testing.assert_array_equal(processed, self.y / 2.)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bn_corrected.csv"
            rows = [f"# Normalization: External I0{description}", "100.00,2.000000"]
            plotting.save_csv_idl(path, rows)
            data = path.read_bytes()
            data.decode("cp1252")
            text = data.decode("utf-8-sig")
        self.assertIn("density=2.1", text)
        self.assertIn("thickness=0.1 um", text)
        self.assertTrue(text.endswith("100.00,2.000000"))

    @unittest.skipUnless(plotting.HAS_TK, "Desktop I0 dialog requires tkinter")
    def test_desktop_bn_correction_uses_windows_readable_metadata(self):
        def value(val):
            return Mock(get=Mock(return_value=val))

        dialog = SimpleNamespace(
            do_smooth=value(False), do_const_offset=value(False),
            cb_const_op=value("Add"), i0_calib_enabled=value(False),
            i0_energy_shift=value(0.), do_od_divide=value(True),
            od_compound=value("BN"), od_density=value(2.1),
            od_thickness=value(0.1), do_autoscale=value(True),
            cb_e=value("Energy"), cb_i=value("I0"),
            x_final=self.x, y_final=self.y / 2., _close_dialog=Mock(),
        )
        plotting.ExternalI0PreviewDialog.on_apply(dialog)
        description = dialog.result[4]
        description.encode("utf-8").decode("cp1252")
        self.assertIn("density=2.1", description)
        self.assertIn("thickness=0.1 um", description)
        self.assertIn("auto-scaled", description)
        np.testing.assert_array_equal(dialog.result[3], self.y / 2.)

    def test_synchronous_desktop_return_and_deferred_notebook_continuation(self):
        def direct(*args, **kwargs):
            if False:
                yield
            return "sync"

        with patch.object(plotting, "_plot_sgm_bsky_data", direct):
            self.assertEqual(plotting.plot_sgm_bsky_data({}), "sync")

        results = []
        def deferred(*args, **kwargs):
            result = yield self.x, self.y, "Test I0", False
            results.append(result)
            return "sync"

        with patch.object(plotting, "_plot_sgm_bsky_data", deferred), patch.object(plotting, "display") as display:
            self.assertIsNone(plotting.plot_sgm_bsky_data({}))
            container = display.call_args.args[0]
            self.assertEqual(results, [])
            find(container, "Smooth").value = True
            find(container, "Apply").click()
            np.testing.assert_allclose(results[0][0], plotting.savgol_filter(self.y, 11, 2))


if __name__ == "__main__":
    unittest.main()
