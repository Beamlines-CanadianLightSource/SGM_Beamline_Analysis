# How to Export a Jupyter Notebook to PDF or HTML

You can save your Jupyter Notebook (`.ipynb`) as a PDF or HTML file to easily share your analysis, preserve formatting, and keep output plots intact. Depending on your environment, here are the easiest ways to accomplish this:

## Option 1: From your IDE (PyCharm / VS Code)
If you are working within a modern IDE like PyCharm or Visual Studio Code:
1. Open the notebook in the editor.
2. Look at the toolbar located at the very top of the notebook editor window.
3. Find the **Export** button (it often looks like an arrow pointing down or an arrow pointing out of a box).
4. Select **Export to HTML** or **Export to PDF**.

## Option 2: From the Jupyter Web Browser Interface
If you are running JupyterLab or the classic Jupyter Notebook in your web browser:
1. Go to the top menu and click **File**.
2. Select **Save and Export Notebook As...** (in older versions of Jupyter, this is listed as "Download as").
3. Choose **HTML** or **PDF** from the dropdown options.

## Option 3: Using the Terminal/Console
If you prefer using the command line, you can use the built-in `nbconvert` tool. If you are using a virtual environment (like in this project), you should run it through your environment's Python to ensure the tool is found:
```bash
.venv\Scripts\python -m nbconvert --to html SGM_BSky_Data_Analysis-N.ipynb
```
*(If it says "No module named nbconvert", run `.venv\Scripts\python -m pip install nbconvert` first. You can also use `--to pdf`, but note that PDF conversion via the command line typically requires you to have LaTeX and Pandoc installed on your system).*

> [!TIP]
> **Recommendation:** 
> Exporting to **HTML** is highly recommended over PDF if you simply want to share the document or keep a record. HTML preserves all your formatting, ensures your interactive plots aren't awkwardly cut off across printed pages, and opens seamlessly in any modern web browser!

## Troubleshooting: Rendering Interactive Widgets in HTML

If your interactive widgets (like VBox or sliders) are not showing up in the exported HTML, try these solutions:

1. **Save Widget State (Most Reliable)**
   Before converting, run all cells, and in the classical Jupyter Notebook menu, select **Widgets -> Save Notebook Widget State**.

2. **Use Browser "Save As"**
   Open the notebook in your browser, and use **File -> Save Page As...** and choose **"Web Page, Complete"**.

3. **Use nbconvert with Embed Option**
   Ensure your command includes metadata for widgets to embed images properly:
   ```bash
   .venv\Scripts\python -m nbconvert --to html --EmbedImagesWidget.embed_images=True SGM_BSky_Data_Analysis.ipynb
   ```

4. **Update Packages**
   Ensure you are using updated versions of `nbconvert` and `ipywidgets` (v6.0+), as older versions had issues rendering VBox properly.

## Troubleshooting: Figure Width Clipping & Page Split Issues in HTML

If exported HTML figures are clipped on the right side or split awkwardly across printed pages:

### 1. Cause of Figure Clipping
* **Container Overflow**: High-resolution or wide multi-detector Matplotlib figures (e.g. `figsize=(16, 5)`) can exceed 2000px in width. Default HTML containers (`.output_png` or `.jp-OutputArea-child`) clip images exceeding viewport width.

### 2. Inject Responsive Image CSS
Add a cell near the top of your notebook to automatically scale wide figures to 100% container width:
```python
from IPython.display import HTML
HTML("""
<style>
/* Prevent image clipping in Jupyter HTML export */
.output_png img, .jp-OutputArea-child img, div.output_subarea img {
    max-width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
}
</style>
""")
```

### 3. Prevent Figures from Splitting Across Pages
To ensure output images do not split across printed page breaks:
```python
HTML("""
<style>
@media print {
    .output_png, .jp-OutputArea-child {
        page-break-inside: avoid;
        break-inside: avoid;
    }
}
</style>
""")
```

### 4. Matplotlib `figsize` & `dpi` Recommendations
* For 5-detector multi-panel plots (SDD1–4 + Average), use `figsize=(12, 4)` to `(14, 5)` with `dpi=100`–`120` for optimal inline web resolution without layout clipping.
