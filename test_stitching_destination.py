from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np

import stitching_utils as stitching


def find_widget(widget, description):
    if getattr(widget, "description", None) == description:
        return widget
    for child in getattr(widget, "children", ()):
        match = find_widget(child, description)
        if match is not None:
            return match
    return None


class DestinationSelectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.confirm = Mock()
        self.cancel = Mock()

    def selector(self, path):
        return stitching._stitching_destination_selector(str(path), self.confirm, self.cancel)

    def test_navigation_and_creating_subfolder_only_on_confirmation(self):
        (self.base / "existing").mkdir()
        ui = self.selector(self.base)
        find_widget(ui, "Subfolders:").value = "existing"
        find_widget(ui, "Open Selected Folder").click()
        self.assertEqual(find_widget(ui, "Folder:").value, str(self.base / "existing"))
        find_widget(ui, "Up").click()
        self.assertEqual(find_widget(ui, "Folder:").value, str(self.base))
        find_widget(ui, "New subfolder:").value = "chosen"
        self.assertFalse((self.base / "chosen").exists())
        self.confirm.assert_not_called()
        find_widget(ui, "Save and Bake").click()
        self.assertTrue((self.base / "chosen").is_dir())
        self.confirm.assert_called_once_with(str(self.base / "chosen"))

    def test_cancel_does_not_create_missing_destination(self):
        proposed = self.base / "missing" / "date"
        ui = self.selector(proposed)
        find_widget(ui, "New subfolder:").value = "stitched"
        find_widget(ui, "Cancel").click()
        self.cancel.assert_called_once_with()
        self.confirm.assert_not_called()
        self.assertFalse((self.base / "missing").exists())

    def test_raw_destination_is_redirected_before_creation(self):
        raw = self.base / "raw" / "date"
        raw.mkdir(parents=True)
        ui = self.selector(raw)
        find_widget(ui, "New subfolder:").value = "chosen"
        find_widget(ui, "Save and Bake").click()
        expected = self.base / "preprocessed" / "date" / "chosen"
        self.confirm.assert_called_once_with(str(expected))
        self.assertTrue(expected.is_dir())
        self.assertFalse((raw / "chosen").exists())

    def test_invalid_input_and_write_errors_do_not_start_baking(self):
        ui = self.selector(self.base)
        for invalid in ("..", "nested/folder"):
            find_widget(ui, "New subfolder:").value = invalid
            self.assertTrue(find_widget(ui, "Save and Bake").disabled)
        find_widget(ui, "New subfolder:").value = ""
        find_widget(ui, "Folder:").value = ""
        self.assertTrue(find_widget(ui, "Save and Bake").disabled)
        find_widget(ui, "Folder:").value = str(self.base)
        with patch.object(stitching.os, "makedirs", side_effect=PermissionError("Read-only folder")):
            find_widget(ui, "Save and Bake").click()
        self.confirm.assert_not_called()
        self.assertIn("Read-only folder", ui.children[-2].value)


class StitchedInputIsolationTests(unittest.TestCase):
    def test_rebake_uses_original_detector_rows_not_nested_stitched_output(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            files = []
            expected = []
            for name, shift in (("left", 0), ("right", 3)):
                source = base / name
                source.mkdir()
                path = source / f"{name}.h5"
                files.append(str(path))
                with h5py.File(path, "w") as hf:
                    hf["map_data/energy"] = [425.]
                    hf["hexapod_waves/x"] = np.array([0., 1., 0., 1.]) + shift
                    hf["hexapod_waves/y"] = [0., 0., 1., 1.]
                rows = np.arange(4 * 256, dtype=np.uint32).reshape(4, 256) + 1000 * shift
                rows.tofile(source / "sdd1_0.bin")
                np.savetxt(source / "mcc1.csv", rows[:, :1], delimiter=",", header="mcc1", comments="")
                expected.append(rows)

            old_output = base / "left" / "Stitched_previous_425_00eV"
            old_output.mkdir()
            np.full((8, 256), 99, dtype=np.uint32).tofile(old_output / "sdd1_0.bin")
            np.savetxt(old_output / "mcc1.csv", np.full((8, 1), 99), header="mcc1", comments="")

            stitched = stitching.stitch_quadrant_maps(files, output_dir=str(base / "new"), verbose=False)
            pack = stitching.analyze_sgm_bsky_data(stitched, verbose=False)
            rows = np.fromfile(pack["sdd_files"]["sdd1"][425.], dtype=np.uint32).reshape(-1, 256)
            np.testing.assert_array_equal(rows, np.concatenate(expected))
            np.testing.assert_array_equal(pack["mcc_data"][425.], np.concatenate(expected)[:, 0])


class InteractiveBakeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.raw = base / "raw" / "date"
        self.raw.mkdir(parents=True)
        self.file = self.raw / "tile.h5"
        with h5py.File(self.file, "w") as hf:
            hf["map_data"] = np.arange(4).reshape(2, 2)
        self.expected_default = base / "preprocessed" / "date"
        self.data_pack = {
            "x": np.array([0., 1., 0., 1.]),
            "y": np.array([0., 0., 1., 1.]),
            "sdd_files": {},
        }
        stitching.plt.switch_backend("Agg")

    def launch(self, has_gui):
        display = Mock()
        patches = (
            patch.object(stitching, "HAS_GUI", has_gui),
            patch.object(stitching, "analyze_sgm_bsky_data", return_value=self.data_pack),
            patch.object(stitching, "display", display),
        )
        for context in patches:
            context.start()
            self.addCleanup(context.stop)
        stitching.interactive_stitching_trim([str(self.file)])
        return display.call_args.args[0]

    def test_hpc_waits_for_confirmation_then_uses_chosen_destination(self):
        with patch.object(stitching, "stitch_quadrant_maps") as bake:
            ui = self.launch(False)
            button = find_widget(ui, "Bake Stitched Image")
            button.click()
            bake.assert_not_called()
            self.assertFalse(self.expected_default.exists())
            self.assertEqual(find_widget(ui, "Folder:").value, str(self.expected_default))
            find_widget(ui, "New subfolder:").value = "selected"
            find_widget(ui, "Save and Bake").click()
            bake.assert_called_once_with(
                [str(self.file)], output_dir=str(self.expected_default / "selected"),
                trims=[{"left": 0., "right": 0., "top": 0., "bottom": 0.}],
                trim_mode="full_edge"
            )
            self.assertFalse(button.disabled)
            self.assertIsNone(find_widget(ui, "Save and Bake"))

    def test_hpc_cancel_keeps_trimming_available_without_writing(self):
        with patch.object(stitching, "stitch_quadrant_maps") as bake:
            ui = self.launch(False)
            button = find_widget(ui, "Bake Stitched Image")
            button.click()
            find_widget(ui, "Cancel").click()
            bake.assert_not_called()
            self.assertFalse(self.expected_default.exists())
            self.assertFalse(button.disabled)
            self.assertIsNone(find_widget(ui, "Save and Bake"))
            button.click()
            self.assertIsNotNone(find_widget(ui, "Save and Bake"))

    def test_desktop_waits_for_confirmation_then_uses_edited_folder(self):
        with patch.object(stitching, "stitch_quadrant_maps") as bake:
            ui = self.launch(True)
            find_widget(ui, "Bake Stitched Image").click()
            bake.assert_not_called()
            self.assertFalse(self.expected_default.exists())
            chosen = self.raw.parent.parent / "stitched-results"
            find_widget(ui, "Folder:").value = str(chosen)
            self.assertFalse(chosen.exists())
            find_widget(ui, "Save and Bake").click()
            bake.assert_called_once()
            self.assertEqual(bake.call_args.kwargs["output_dir"], str(chosen))
            self.assertTrue(chosen.is_dir())
            self.assertFalse(self.expected_default.exists())
            self.assertIsNone(find_widget(ui, "Save and Bake"))


if __name__ == "__main__":
    unittest.main()
