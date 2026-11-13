import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import plot_sgm_bsky_data_batch as batch


def find(widget, description):
    if getattr(widget, "description", None) == description:
        return widget
    for child in getattr(widget, "children", ()):
        found = find(child, description)
        if found is not None:
            return found
    return None


class BatchHPCPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.packs = []
        batch.plt.switch_backend("Agg")
        self.addCleanup(batch.plt.close, "all")
        for index in range(2):
            folder = self.root / str(index)
            folder.mkdir()
            spectra = np.zeros((2, 256), dtype=np.uint32)
            spectra[:, 20] = 5
            spectrum_file = folder / "spectrum.dat"
            spectra.tofile(spectrum_file)
            energies = [100., 101., 102.]
            mcc = {}
            for step, energy in enumerate(energies):
                path = folder / f"{step}.csv"
                path.write_text(f"ch1,ch4\n{2 + index * 2 + step},12\n{2 + index * 2 + step},12\n")
                mcc[energy] = str(path)
            self.packs.append({
                "scan_name": f"scan{index}", "sample_name": f"sample{index}",
                "energies": energies, "representative_energy": 101.,
                "x": np.array([0., 1.]), "y": np.array([0., 1.]),
                "sdd_files": {"sdd1": {e: str(spectrum_file) for e in energies}},
                "mcc_files": mcc, "save_dir": str(folder),
            })
        self.calibration = {
            "sdd1": {"gain": 10., "offset": 5.},
            "_metadata": {"scan_used": "calibration scan", "edges_used": ["Al"], "last_updated": "today"},
        }

    def setup_ui(self):
        captured = []
        display = patch("IPython.display.display")
        calibration = patch.object(batch, "load_calibration", return_value=self.calibration)
        mock_display = display.start()
        calibration.start()
        self.addCleanup(display.stop)
        self.addCleanup(calibration.stop)
        batch._batch_hpc_setup(self.packs, None, captured.append)
        return mock_display.call_args.args[0], captured

    def test_spectrum_preview_calibration_choice_and_shared_i0_settings(self):
        ui, captured = self.setup_ui()
        self.assertIn("calibration scan", ui.children[0].children[0].value)
        self.assertIn("gain = 10", ui.children[0].children[3].value)
        self.assertGreater(len(ui.children[0].children[2].value), 0)
        find(ui, "Apply SDD calibration").value = False
        find(ui, "Waterfall offsets").value = True
        find(ui, "Continue").click()
        find(ui, "Scan:").value = 1
        find(ui, "Constant").value = True
        find(ui, "Value:").value = 2.
        find(ui, "Apply").click()
        self.assertEqual(len(captured), 1)
        config = captured[0]
        self.assertFalse(config["use_calibration"])
        self.assertTrue(config["apply_waterfall"])
        self.assertTrue(config["i0_settings"]["do_const"])
        self.assertEqual(config["i0_settings"]["const_val"], 2.)
        self.assertIn("+ Constant", config["i0_source_label"])

    def test_window_must_be_valid_for_every_scan(self):
        ui, captured = self.setup_ui()
        find(ui, "Continue").click()
        find(ui, "Window:").value = 5
        find(ui, "Smooth").value = True
        self.assertTrue(find(ui, "Apply").disabled)
        find(ui, "Cancel").click()
        self.assertEqual(captured[0]["i0_settings"], None)

    def test_external_i0_selection_shift_and_apply(self):
        external = self.root / "external.csv"
        external.write_text("Energy,I0\n100,10\n101,20\n102,30\n")
        ui, captured = self.setup_ui()
        find(ui, "I0 source:").value = "external"
        find(ui, "Continue").click()
        find(ui, "Folder:").value = str(self.root)
        find(ui, "Files (.csv, .txt):").value = external.name
        find(ui, "Select File").click()
        find(ui, "Energy shift").value = True
        find(ui, "Shift (eV):").value = 0.5
        find(ui, "Apply").click()
        config = captured[0]
        self.assertFalse(config["use_internal"])
        np.testing.assert_array_equal(config["external_xy"][0], [100.5, 101.5, 102.5])
        np.testing.assert_array_equal(config["external_xy"][1], [10., 20., 30.])
        with (
            patch("IPython.get_ipython", return_value=None),
            patch.object(batch.plt, "show"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            batch._plot_sgm_bsky_data_batch(self.packs, channel_roi=(20, 20), hpc_config=config)
        norm = next(batch.plt.figure(n) for n in batch.plt.get_fignums()
                    if batch.plt.figure(n)._suptitle
                    and batch.plt.figure(n)._suptitle.get_text().startswith("Batch NORMALIZED"))
        np.testing.assert_allclose(norm.axes[0].lines[0].get_ydata(), 10 / np.array([10., 15., 25.]))

    def test_incomplete_calibration_is_reported_and_cannot_be_applied(self):
        self.packs[0]["sdd_files"]["sdd2"] = dict(self.packs[0]["sdd_files"]["sdd1"])
        ui, captured = self.setup_ui()
        checkbox = find(ui, "Apply SDD calibration")
        self.assertTrue(checkbox.disabled)
        self.assertFalse(checkbox.value)
        self.assertIn("sdd2", ui.children[0].children[0].value)
        self.assertEqual(captured, [])

    def test_shared_i0_values_drive_each_scan_and_export(self):
        ui, captured = self.setup_ui()
        find(ui, "Apply SDD calibration").value = False
        find(ui, "Continue").click()
        find(ui, "Constant").value = True
        find(ui, "Value:").value = 2.
        find(ui, "Apply").click()
        config = captured[0]
        with (
            patch("IPython.get_ipython", return_value=None),
            patch.object(batch.plt, "show"),
            patch.object(batch, "show_custom_dialog", return_value="ok"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            batch._plot_sgm_bsky_data_batch(self.packs, channel_roi=(20, 20), hpc_config=config)
            norm = next(batch.plt.figure(n) for n in batch.plt.get_fignums()
                        if batch.plt.figure(n)._suptitle
                        and batch.plt.figure(n)._suptitle.get_text().startswith("Batch NORMALIZED"))
            np.testing.assert_allclose(norm.axes[0].lines[0].get_ydata(), 10 / np.array([4., 5., 6.]))
            np.testing.assert_allclose(norm.axes[0].lines[1].get_ydata(), 10 / np.array([6., 7., 8.]))
            norm._btn_save_csv._observers.process("clicked", None)
        csvs = list(self.root.glob("**/*summary.csv"))
        self.assertEqual(len(csvs), 2)
        text = csvs[0].read_text(encoding="utf-8-sig")
        self.assertIn("Modified_Au_Mesh_I0 (used for normalization)", text)
        self.assertIn("+ Constant", text)

    def test_missing_i0_reports_error_without_starting_plot(self):
        Path(self.packs[0]["mcc_files"][100.]).unlink()
        ui, captured = self.setup_ui()
        find(ui, "Continue").click()
        self.assertIn("unavailable", ui.children[0].children[-1].value)
        self.assertEqual(captured, [])


if __name__ == "__main__":
    unittest.main()
