import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import plot_sgm_bsky_data_batch as batch
import analyze_sgm_bsky_data_batch as loader


class BatchAutoElementROITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        batch.plt.switch_backend("Agg")
        self.addCleanup(batch.plt.close, "all")
        self.packs = []
        for index in range(2):
            spectra = np.zeros((2, 256), dtype=np.uint32)
            spectra[:, 100:103] = index + 1
            spectra[:, 20:41] = 10
            path = Path(self.temp.name) / f"scan{index}.dat"
            spectra.tofile(path)
            self.packs.append({
                "scan_name": f"scan{index}_si_k", "energies": [1., 2., 3.],
                "sample_name": f"Sample{index}",
                "x": np.array([0., 1.]), "y": np.array([0., 1.]),
                "sdd_files": {"sdd1": {energy: str(path) for energy in (1., 2., 3.)}},
            })

    def run_plot(self, fit_result, calibration=True, waterfall=False, **kwargs):
        with (
            patch.object(batch, "show_custom_dialog", side_effect=[calibration, True, waterfall]),
            patch.object(batch, "load_calibration", return_value={"sdd1": {"gain": 10., "offset": 0.}}),
            patch.object(batch.elem_roi_utils, "auto_detect_and_fit_stack_roi", return_value=fit_result) as fit,
            patch.object(batch, "get_mcc_val", return_value=1.),
            patch.object(batch.plt, "show"),
            patch("IPython.get_ipython", return_value=None),
            contextlib.redirect_stderr(io.StringIO()) as errors,
        ):
            batch.plot_sgm_bsky_data_batch(self.packs, **kwargs)
        raw = next(fig for fig in map(batch.plt.figure, batch.plt.get_fignums())
                   if fig._suptitle and fig._suptitle.get_text().startswith("Batch RAW"))
        values = [line.get_ydata() for line in raw.axes[0].lines]
        return fit, errors.getvalue(), values

    def test_auto_fit_skips_survey_and_applies_shared_roi_to_both_scans(self):
        self.packs.insert(0, {"scan_name": "survey", "energies": [1.], "sdd_files": {}})
        result = {"xrf_roi": (1000., 1020.), "element": "Si", "edge": "K"}
        fit, _, values = self.run_plot(
            result, auto_element_roi=True, xrf_roi=(200., 400.),
            map_roi=[0., 1., 0., 1.], element="Si", n_sigma=3.
        )
        fit.assert_called_once_with(
            self.packs[1], map_roi=[0., 1., 0., 1.],
            element_override="Si", n_sigma=3., verbose=True
        )
        np.testing.assert_array_equal(values[0], [6., 6., 6.])
        np.testing.assert_array_equal(values[1], [12., 12., 12.])
        self.assertEqual(self.packs[1]["auto_element_roi_info"], result)
        figures = [batch.plt.figure(number) for number in batch.plt.get_fignums()]
        for figure in figures:
            if figure._suptitle:
                self.assertIn("SDD Energy ROI: 1000.0-1020.0 eV", figure._suptitle.get_text())
                self.assertNotIn("Channel ROI", figure._suptitle.get_text())
                for axis in figure.axes[:6]:
                    if axis.lines:
                        self.assertEqual("Sample0", axis.lines[0].get_label())

    def test_failed_fit_preserves_manual_roi_with_warning(self):
        fit, errors, values = self.run_plot(
            None, auto_element_roi=True, xrf_roi=(200., 400.)
        )
        fit.assert_called_once()
        self.assertIn("No ROI could be fitted", errors)
        np.testing.assert_array_equal(values[0], [420., 420., 420.])

    def test_calibration_declined_warns_without_fitting(self):
        fit, errors, values = self.run_plot(
            None, calibration=False, auto_element_roi=True, channel_roi=(20, 40)
        )
        fit.assert_not_called()
        self.assertIn("SDD calibration must be applied", errors)
        np.testing.assert_array_equal(values[0], [420., 420., 420.])

    def test_default_does_not_fit_or_change_manual_roi(self):
        fit, _, values = self.run_plot(None, channel_roi=(20, 40))
        fit.assert_not_called()
        np.testing.assert_array_equal(values[0], [420., 420., 420.])

    def test_energy_labels_use_calibration_for_channel_input_and_preserve_uncalibrated_units(self):
        self.run_plot(None, channel_roi=(20, 40))
        titles = [batch.plt.figure(number)._suptitle.get_text()
                  for number in batch.plt.get_fignums() if batch.plt.figure(number)._suptitle]
        self.assertTrue(all("SDD Energy ROI: 200.0-400.0 eV" in title for title in titles))
        self.assertEqual(
            batch._batch_roi_label((20, 40), 200., 400., False),
            "SDD Channel ROI: 20-40 (energy unavailable without calibration)"
        )

    def test_spectrum_label_uses_only_sample_name(self):
        self.assertEqual(
            batch._batch_spectrum_label(
                {"sample_name": "Wheat"}, "2026-06-18_150552_stack_data"
            ), "Wheat"
        )
        self.assertEqual(
            batch._batch_spectrum_label({"sample_name": "Wheat"}, "150552_Wheat_stack"),
            "Wheat"
        )
        self.assertEqual(
            batch._batch_spectrum_label(
                {"sample_name": "N/A", "h5_file_path": str(Path(self.temp.name) / "150552_Wheat_stack" / "2026-06-18_150552_stack_data.h5")},
                "2026-06-18_150552_stack_data"
            ), "Wheat"
        )

    def test_one_external_sample_only_legend_per_figure_in_waterfall_mode(self):
        self.run_plot(None, channel_roi=(20, 40), waterfall=True)
        figures = [batch.plt.figure(number) for number in batch.plt.get_fignums()]
        for figure in figures:
            if not figure._suptitle:
                continue
            self.assertEqual(len(figure.legends), 1)
            legend = figure.legends[0]
            self.assertEqual([text.get_text() for text in legend.get_texts()], ["Sample0", "Sample1"])
            self.assertTrue(all(axis.get_legend() is None for axis in figure.axes))
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            box = legend.get_window_extent(renderer)
            for axis in figure.axes:
                self.assertFalse(box.overlaps(axis.get_window_extent(renderer)))
                if axis.lines:
                    self.assertEqual(axis.lines[0].get_color(), figure.axes[4].lines[0].get_color())
            np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), [420., 420., 420.])
            np.testing.assert_array_equal(figure.axes[0].lines[1].get_ydata(), [421., 421., 421.])

    def test_loader_summary_identifies_scan_and_sample(self):
        path = Path(self.temp.name) / "scan.h5"
        path.touch()
        pack = dict(self.packs[0], h5_file_path=str(path), representative_energy=2.)
        with (
            patch.object(loader, "analyze_sgm_bsky_data", return_value=pack),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            loaded = loader.analyze_sgm_bsky_data_batch([str(path)])
        self.assertEqual(loaded, [pack])
        self.assertIn("Scan Name:  scan0_si_k", output.getvalue())
        self.assertIn("Sample Name: Sample0", output.getvalue())
        self.assertNotIn("File:", output.getvalue())
        self.assertEqual(output.getvalue().count("scan0_si_k"), 1)


if __name__ == "__main__":
    unittest.main()
