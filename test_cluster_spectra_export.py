import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
from PIL import Image

import cluster_xanes_analysis as clustering


class ClusterSpectraExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sample_PCA-CA.h5"
        self.datasets = ["sdd1", "sdd2", "sdd3", "sdd4", "average", "selected_average"]
        self.energy = np.array([100., 101., 102.])
        with h5py.File(self.path, "w") as handle:
            meas = handle.create_group("entry/measurement")
            meas.create_dataset("energy", data=self.energy)
            meas.create_dataset("x", data=[0., 1., 2.])
            meas.create_dataset("y", data=[0., 1.])
            scores = np.array([[1., 1.], [1.2, 1.1], [9., 9.],
                               [9.2, 9.1], [1.1, 1.2], [0., 0.]])
            for index, name in enumerate(self.datasets):
                stack = np.arange(18., dtype=float).reshape(2, 3, 3) + index * 100.
                meas.create_dataset(name, data=stack)
                handle.create_dataset(
                    f"entry/pca_results/{name}/eigenimages",
                    data=np.roll(scores, index, axis=0).reshape(2, 3, 2),
                )

    def test_run_all_exports_separate_plot_matching_cluster_columns(self):
        output = io.StringIO()
        with (
            patch.object(clustering, "plot_results") as individual,
            patch.object(clustering, "plot_multi_cluster_results") as comparison,
            contextlib.redirect_stdout(output),
        ):
            result = clustering.run_clustering_all_detectors(
                str(self.path), n_clusters=2, apply_waterfall=True, waterfall_offset=123.,
                energy_range=(100.5, 101.5),
            )
        run_path = Path(result)
        self.assertEqual(run_path.parent, self.path.parent / "Cluster Analysis" / "Run_001")
        results = comparison.call_args.args[0]
        self.assertEqual([res["dataset"] for res in results], self.datasets)
        with h5py.File(run_path, "r") as handle:
            for index, res in enumerate(results):
                csv_path = run_path.with_name(f"{run_path.stem}_{res['dataset']}_cluster_spectra.csv")
                self.assertTrue(csv_path.exists(), csv_path)
                text = csv_path.read_text(encoding="utf-8-sig")
                self.assertIn("# Column 2: Cluster_1", text)
                self.assertIn("# Column 3: Cluster_2", text)
                self.assertIn("# Dataset Source:", text)
                self.assertIn("# Normalized: Yes", text)
                data = pd.read_csv(csv_path, comment="#", header=None, encoding="utf-8-sig").to_numpy()
                self.assertEqual(data.shape, (3, 3))
                np.testing.assert_array_equal(data[:, 0], self.energy)
                np.testing.assert_allclose(data[:, 1:].T, individual.call_args_list[index].args[4])
                stack = handle[f"entry/measurement/{res['dataset']}"][()]
                for cluster_id in range(2):
                    expected = stack[res["cluster_map"] == cluster_id].mean(axis=0)
                    np.testing.assert_allclose(data[:, cluster_id + 1], expected)
                self.assertIn(str(csv_path), output.getvalue())
                for extension in ("png", "tiff"):
                    image_path = run_path.with_name(
                        f"{run_path.stem}_{res['dataset']}_cluster_map.{extension}"
                    )
                    self.assertTrue(image_path.exists(), image_path)
                    with Image.open(image_path) as image:
                        np.testing.assert_array_equal(np.array(image), res["cluster_map"] + 1)
        self.assertTrue(run_path.with_name(f"{run_path.stem}_all_detectors_cluster_sums.csv").exists())
        self.assertTrue(run_path.with_name(f"{run_path.stem}_sdd1_cluster_spectra_summary.csv").exists())

    def test_ipfy_export_matches_plotted_pfy_not_original_or_sum(self):
        with h5py.File(self.path, "r+") as handle:
            handle["entry/measurement"].attrs["ipfy_mode"] = True
        with patch.object(clustering, "plot_results") as plot:
            result = clustering.cluster_xanes_analysis(
                str(self.path), dataset_name="sdd1", n_clusters=2, return_dict=True,
                apply_waterfall=True, waterfall_offset=123.,
            )
        csv_path = self.path.with_name(f"{self.path.stem}_sdd1_cluster_spectra.csv")
        self.assertTrue(csv_path.exists(), csv_path)
        data = pd.read_csv(csv_path, comment="#", header=None, encoding="utf-8-sig").to_numpy()
        np.testing.assert_allclose(data[:, 1:].T, plot.call_args.args[4])
        np.testing.assert_allclose(data[:, 1:].T, result["cluster_spectra"])
        with h5py.File(self.path, "r") as handle:
            original = handle["entry/pca_results/sdd1/clustering/cluster_spectra"][()]
        self.assertFalse(np.allclose(data[:, 1:].T, original))

    def test_map_export_is_independent_of_showing_individual_plots(self):
        with (
            patch.object(clustering, "plot_results") as individual,
            patch.object(clustering, "plot_multi_cluster_results"),
        ):
            result = clustering.run_clustering_all_detectors(
                str(self.path), n_clusters=2, show_individual_plots=False,
            )
        run_path = Path(result)
        individual.assert_not_called()
        for dataset in self.datasets:
            for extension in ("png", "tiff"):
                self.assertTrue(run_path.with_name(
                    f"{run_path.stem}_{dataset}_cluster_map.{extension}"
                ).exists())

    def test_csv_write_failure_is_not_swallowed(self):
        with patch("builtins.open", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                clustering.save_csv_with_header(
                    "unwritable.csv", pd.DataFrame({"Energy_eV": self.energy}), {},
                )


if __name__ == "__main__":
    unittest.main()
