import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import matplotlib.pyplot as plt
import numpy as np

import cluster_xanes_analysis as clustering
import pca_xanes_analysis as pca


class AnalysisRunFolderTests(unittest.TestCase):
    def setUp(self):
        plt.switch_backend("Agg")
        self.addCleanup(plt.close, "all")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "sample_PCA-CA.h5"
        self.datasets = ["sdd1", "sdd2", "sdd3", "sdd4", "average", "selected_average"]
        rng = np.random.default_rng(42)
        with h5py.File(self.source, "w") as handle:
            meas = handle.create_group("entry/measurement")
            meas.create_dataset("energy", data=[100., 101., 102., 103.])
            meas.create_dataset("x", data=[0., 1., 2.])
            meas.create_dataset("y", data=[0., 1., 2.])
            for name in self.datasets:
                meas.create_dataset(name, data=rng.uniform(1., 10., (3, 3, 4)))

    def test_pca_and_clustering_runs_preserve_source_and_previous_results(self):
        original = self.source.read_bytes()
        with (
            patch.object(pca, "_display_scrollable_figure", return_value=False),
            patch.object(clustering, "_display_scrollable_figure", return_value=False),
            patch.object(plt, "show"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            pca_path = Path(pca.run_pca_all_detectors(str(self.source), n_components=2))
            self.assertEqual(pca_path.parent, self.root / "PCA Analysis" / "Run_001")
            self.assertEqual(self.source.read_bytes(), original)
            pca_snapshot = {path.name: path.read_bytes() for path in pca_path.parent.iterdir()}

            cluster_path = Path(clustering.run_clustering_all_detectors(str(pca_path), n_clusters=2))
            self.assertEqual(cluster_path.parent, self.root / "Cluster Analysis" / "Run_001")
            self.assertEqual(pca_snapshot, {
                path.name: path.read_bytes() for path in pca_path.parent.iterdir()
            })
            cluster_snapshot = {
                path.name: path.read_bytes() for path in cluster_path.parent.iterdir()
            }

            with (
                patch.object(pca, "plot_multi_detector_results"),
                patch.object(clustering, "plot_results"),
                patch.object(clustering, "plot_multi_cluster_results"),
            ):
                next_pca = Path(pca.run_pca_all_detectors(str(pca_path), n_components=3))
                next_cluster = Path(clustering.run_clustering_all_detectors(
                    str(cluster_path), n_clusters=3,
                ))
            self.assertEqual(next_pca.parent, self.root / "PCA Analysis" / "Run_002")
            self.assertEqual(next_cluster.parent, self.root / "Cluster Analysis" / "Run_002")

        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(pca_snapshot, {
            path.name: path.read_bytes() for path in pca_path.parent.iterdir()
        })
        self.assertEqual(cluster_snapshot, {
            path.name: path.read_bytes() for path in cluster_path.parent.iterdir()
        })
        for suffix in ("pca_comparison_images.png", "pca_comparison_vectors.png"):
            self.assertTrue(pca_path.with_name(f"{pca_path.stem}_{suffix}").exists())
        for dataset in self.datasets:
            self.assertTrue(pca_path.with_name(
                f"{pca_path.stem}_pca_vectors_{dataset}.csv"
            ).exists())
            for suffix in ("cluster_map.png", "cluster_map.tiff", "cluster_preview.png",
                           "cluster_spectra.csv", "cluster_spectra_summary.csv"):
                self.assertTrue(cluster_path.with_name(
                    f"{cluster_path.stem}_{dataset}_{suffix}"
                ).exists())
        for suffix in ("cluster_comparison_maps.png", "cluster_comparison_spectra.png",
                       "all_detectors_cluster_sums.csv"):
            self.assertTrue(cluster_path.with_name(f"{cluster_path.stem}_{suffix}").exists())
        with h5py.File(cluster_path, "r") as handle:
            self.assertEqual(handle.attrs["analysis_source_h5"], str(pca_path))
            self.assertEqual(json.loads(handle.attrs["analysis_parameters"]), {"n_clusters": 2})
            for dataset in self.datasets:
                self.assertIn(f"entry/pca_results/{dataset}/clustering/cluster_map", handle)
        self.assertEqual(
            {path.name for path in self.root.iterdir()},
            {self.source.name, "PCA Analysis", "Cluster Analysis"},
        )

    def test_existing_run_is_not_reused_and_failed_copy_is_reported_and_cleaned(self):
        import analysis_output_utils as output

        existing = self.root / "PCA Analysis" / "Run_001"
        existing.mkdir(parents=True)
        sentinel = existing / "keep.txt"
        sentinel.write_text("previous analysis")
        original = self.source.read_bytes()
        def partial_copy(source, destination):
            Path(destination).write_bytes(b"partial HDF5 copy")
            raise OSError("disk full")

        with patch.object(output.shutil, "copyfile", side_effect=partial_copy):
            with self.assertRaisesRegex(OSError, "disk full"):
                output.create_analysis_run(str(self.source), "PCA Analysis", {"n_components": 2})
        self.assertFalse((existing.parent / "Run_002").exists())
        self.assertEqual(sentinel.read_text(), "previous analysis")
        self.assertEqual(self.source.read_bytes(), original)
        result = Path(output.create_analysis_run(
            str(self.source), "PCA Analysis", {"n_components": 2},
        ))
        self.assertEqual(result.parent, existing.parent / "Run_002")
        self.assertEqual(sentinel.read_text(), "previous analysis")


if __name__ == "__main__":
    unittest.main()
