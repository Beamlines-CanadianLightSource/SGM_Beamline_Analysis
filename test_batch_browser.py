from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_sgm_bsky_data_batch as batch


def find(widget, description):
    if getattr(widget, "description", None) == description:
        return widget
    for child in getattr(widget, "children", ()):
        found = find(child, description)
        if found is not None:
            return found
    return None


class BatchBrowserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.first = self.root / "first.h5"
        self.first.touch()
        nested = self.root / "project" / "scan"
        nested.mkdir(parents=True)
        self.second = nested / "second.h5"
        self.second.touch()
        (nested / ".hidden.h5").touch()
        checkpoint = self.root / ".ipynb_checkpoints"
        checkpoint.mkdir()
        (checkpoint / "ignored.h5").touch()

    def launch(self):
        class Notebook:
            kernel = object()

        patches = (
            patch.object(batch, "HAS_GUI", False),
            patch("IPython.get_ipython", return_value=Notebook()),
            patch("IPython.display.display"),
        )
        active = []
        for item in patches:
            active.append(item.start())
            self.addCleanup(item.stop)
        result = batch.analyze_sgm_bsky_data_batch(verbose=False)
        display = active[-1]
        return result, display.call_args.args[0]

    def test_select_files_and_folder_then_load_into_returned_list(self):
        result, ui = self.launch()
        folder = find(ui, "Folder:")
        folder.value = str(self.root)
        files = find(ui, "Files (.h5):")
        files.value = self.first.name
        find(ui, "Add Selected File").click()
        find(ui, "Add Current Folder").click()
        selected = find(ui, "Selected:")
        self.assertEqual(set(selected.options), {str(self.first), str(self.second)})
        self.assertEqual(result, [])

        def make_pack(path, verbose):
            return {"scan_name": Path(path).stem}

        with patch.object(batch, "analyze_sgm_bsky_data", side_effect=make_pack) as load:
            find(ui, "Load Batch").click()
        self.assertEqual([pack["scan_name"] for pack in result], ["first", "second"])
        self.assertEqual(load.call_count, 2)

    def test_files_can_be_added_across_folders_and_removed(self):
        result, ui = self.launch()
        folder = find(ui, "Folder:")
        folder.value = str(self.root)
        find(ui, "Files (.h5):").value = self.first.name
        find(ui, "Add Selected File").click()
        folder.value = str(self.second.parent)
        find(ui, "Files (.h5):").value = self.second.name
        find(ui, "Add Selected File").click()
        selected = find(ui, "Selected:")
        selected.value = (str(self.first),)
        find(ui, "Remove Selected").click()
        self.assertEqual(selected.options, (str(self.second),))
        find(ui, "Clear Selection").click()
        self.assertEqual(selected.options, ())
        with patch.object(batch, "analyze_sgm_bsky_data") as load:
            find(ui, "Load Batch").click()
            find(ui, "Cancel").click()
        load.assert_not_called()
        self.assertEqual(result, [])

    def test_directory_errors_are_visible_and_do_not_add_partial_results(self):
        result, ui = self.launch()
        find(ui, "Folder:").value = str(self.root)
        with patch.object(batch.os, "walk", side_effect=PermissionError("access denied")):
            find(ui, "Add Current Folder").click()
        self.assertIn("access denied", ui.children[-2].value)
        self.assertEqual(find(ui, "Selected:").options, ())
        self.assertEqual(result, [])

    def test_loading_without_a_successful_scan_shows_error(self):
        result, ui = self.launch()
        find(ui, "Folder:").value = str(self.root)
        find(ui, "Files (.h5):").value = self.first.name
        find(ui, "Add Selected File").click()
        with patch.object(batch, "analyze_sgm_bsky_data", return_value=None):
            find(ui, "Load Batch").click()
        self.assertIn("No scans were loaded", ui.children[-2].value)
        self.assertEqual(result, [])

    def test_explicit_paths_bypass_browser(self):
        with (
            patch.object(batch, "HAS_GUI", False),
            patch.object(batch, "_browse_batch_scans") as browser,
            patch.object(batch, "analyze_sgm_bsky_data", return_value={"scan_name": "first"}),
        ):
            result = batch.analyze_sgm_bsky_data_batch([str(self.first)], verbose=False)
        browser.assert_not_called()
        self.assertEqual(result, [{"scan_name": "first"}])

    def test_non_notebook_keeps_existing_dialog_flow(self):
        with (
            patch.object(batch, "HAS_GUI", False),
            patch("IPython.get_ipython", return_value=None),
            patch.object(batch, "show_custom_dialog", return_value=False),
            patch.object(batch, "safe_filedialog_call", return_value=[str(self.first)]),
            patch.object(batch, "analyze_sgm_bsky_data", return_value={"scan_name": "first"}),
        ):
            result = batch.analyze_sgm_bsky_data_batch(verbose=False)
        self.assertEqual(result, [{"scan_name": "first"}])

    def test_desktop_notebook_keeps_existing_dialog_flow(self):
        with (
            patch.object(batch, "HAS_GUI", True),
            patch.object(batch, "_browse_batch_scans") as browser,
            patch.object(batch, "show_custom_dialog", return_value=False),
            patch.object(batch, "safe_filedialog_call", return_value=[str(self.first)]),
            patch.object(batch, "analyze_sgm_bsky_data", return_value={"scan_name": "first"}),
        ):
            result = batch.analyze_sgm_bsky_data_batch(verbose=False)
        browser.assert_not_called()
        self.assertEqual(result, [{"scan_name": "first"}])


if __name__ == "__main__":
    unittest.main()
