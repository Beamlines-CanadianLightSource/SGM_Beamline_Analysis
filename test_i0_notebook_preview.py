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

    def test_optical_density_adds_a_separate_curve_without_hiding_either_i0(self):
        od = np.linspace(1., 3., len(self.x))
        figures = []
        figure_type = plotting.Figure

        def capture_figure(*args, **kwargs):
            fig = figure_type(*args, **kwargs)
            figures.append(fig)
            return fig

        with patch.object(plotting, "HAS_SF", True), \
             patch.object(plotting, "sf", return_value=od), \
             patch.object(plotting, "Figure", side_effect=capture_figure):
            ui = self.preview()
            self.assertEqual(len(figures[-1].axes), 1)
            find(ui, "Auto-scale OD").value = False
            find(ui, "Divide by OD").value = True
            comparison, od_plot = figures[-1].axes
            original, modified = comparison.lines
            np.testing.assert_array_equal(original.get_ydata(), self.y)
            np.testing.assert_allclose(modified.get_ydata(), self.y / od)
            np.testing.assert_array_equal(od_plot.lines[0].get_xdata(), self.x)
            np.testing.assert_array_equal(od_plot.lines[0].get_ydata(), od)
            self.assertEqual(od_plot.lines[0].get_label(), "OD (BN)")
            find(ui, "Divide by OD").value = False
            self.assertEqual(len(figures[-1].axes), 1)
            self.assertEqual(len(figures[-1].axes[0].lines), 2)

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

    def test_notebook_internal_uses_widgets_regardless_of_desktop_gui(self):
        for desktop_gui in (True, False):
            with self.subTest(desktop_gui=desktop_gui):
                pack = self.scan_pack()
                with patch.object(plotting, "HAS_GUI", desktop_gui), \
                     patch.object(plotting, "get_ipython", return_value=SimpleNamespace(kernel=object())), \
                     patch.object(plotting, "get_tk_root") as root, \
                     patch("builtins.input") as prompt, \
                     patch.object(plotting, "display") as display, \
                     patch.object(plotting, "Synchronizer", side_effect=KeyboardInterrupt):
                    plotting.plot_sgm_bsky_data(pack)
                    ui = display.call_args.args[0]
                    find(ui, "Preview I0").click()
                    with self.assertRaises(KeyboardInterrupt):
                        find(ui, "Apply").click()
                    root.assert_not_called()
                    prompt.assert_not_called()
                np.testing.assert_array_equal(pack["ext_i0_values"], [4., 8.])

    def scan_pack(self):
        return {
            "scan_name": "test scan", "sample_name": "test scan",
            "energies": [100., 101.], "sdd_files": {},
            "mcc_data": {100.: np.array([[4.]]), 101.: np.array([[8.]])},
        }

    def test_external_notebook_browses_and_uses_selected_columns(self):
        pack = self.scan_pack()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reference.csv"
            path.write_text("Energy,I0,Alternate\n101,80,2\n100,40,1\n", encoding="utf-8")
            pack["h5_dir"] = folder
            with patch.object(plotting, "get_ipython", return_value=SimpleNamespace(kernel=object())), \
                 patch.object(plotting, "safe_filedialog_call") as dialog, \
                 patch("builtins.input") as prompt, \
                 patch.object(plotting, "display") as display, \
                 patch.object(plotting, "Synchronizer", side_effect=KeyboardInterrupt):
                plotting.plot_sgm_bsky_data(pack)
                ui = display.call_args.args[0]
                find(ui, "Source:").value = "external"
                self.assertTrue(find(ui, "Preview I0").disabled)
                find(ui, "Files (.csv, .txt):").value = path.name
                find(ui, "Select File").click()
                find(ui, "Intensity:").value = "Energy"
                find(ui, "Preview I0").click()
                self.assertIsNone(find(ui, "Apply"))
                find(ui, "Intensity:").value = "Alternate"
                find(ui, "Preview I0").click()
                with self.assertRaises(KeyboardInterrupt):
                    find(ui, "Apply").click()
                dialog.assert_not_called()
                prompt.assert_not_called()
        np.testing.assert_array_equal(pack["ext_i0_values"], [1., 2.])
        self.assertIn("reference.csv [Alternate]", pack["i0_source"])


if __name__ == "__main__":
    unittest.main()
