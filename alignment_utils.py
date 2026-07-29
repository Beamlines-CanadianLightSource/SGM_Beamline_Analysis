import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import ipywidgets as widgets
from IPython.display import display
import traceback
from scipy.interpolate import griddata
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
from matplotlib.figure import Figure

# --- Jupyter/macOS Tkinter Stability Patch ---
_orig_showinfo = messagebox.showinfo
_orig_showerror = messagebox.showerror
_orig_showwarning = messagebox.showwarning
_orig_askyesno = messagebox.askyesno
_orig_askokcancel = messagebox.askokcancel

def _in_jupyter():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except:
        return False

class CustomTkDialog(tk.Toplevel):
    def __init__(self, parent, title, message, dialog_type="info"):
        super().__init__(parent)
        self.title(title)
        self.result = None
        
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self.attributes("-topmost", True)
        self.grab_set() # Modal dialog behavior
        
        # Padding frame
        frame = tk.Frame(self, padx=15, pady=15)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Message label
        lbl = tk.Label(frame, text=message, justify=tk.LEFT, wraplength=400)
        lbl.pack(side=tk.TOP, pady=(0, 15))
        
        # Button frame
        btn_frame = tk.Frame(frame)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        if dialog_type == "info":
            btn = tk.Button(btn_frame, text="OK", command=self.on_ok, width=10, bg='lightgray')
            btn.pack(side=tk.BOTTOM, pady=5)
        elif dialog_type == "yesno":
            btn_yes = tk.Button(btn_frame, text="Yes", command=self.on_yes, width=10, bg='lightgreen')
            btn_yes.pack(side=tk.LEFT, padx=5)
            btn_no = tk.Button(btn_frame, text="No", command=self.on_no, width=10, bg='lightcoral')
            btn_no.pack(side=tk.RIGHT, padx=5)
        elif dialog_type == "okcancel":
            btn_ok = tk.Button(btn_frame, text="OK", command=self.on_ok, width=10, bg='lightgreen')
            btn_ok.pack(side=tk.LEFT, padx=5)
            btn_cancel = tk.Button(btn_frame, text="Cancel", command=self.on_cancel, width=10, bg='lightcoral')
            btn_cancel.pack(side=tk.RIGHT, padx=5)
        elif dialog_type == "askstring":
            self.entry = tk.Entry(frame, width=40)
            self.entry.pack(side=tk.TOP, pady=(0, 15))
            self.entry.focus_set()
            self.entry.bind("<Return>", lambda e: self.on_string_ok())
            
            btn_ok = tk.Button(btn_frame, text="OK", command=self.on_string_ok, width=10, bg='lightgreen')
            btn_ok.pack(side=tk.LEFT, padx=5)
            btn_cancel = tk.Button(btn_frame, text="Cancel", command=self.on_cancel, width=10, bg='lightcoral')
            btn_cancel.pack(side=tk.RIGHT, padx=5)
            
        # Center the window
        self.update_idletasks()
        width = max(350, self.winfo_width())
        height = max(150, self.winfo_height())
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')
        
    def on_ok(self):
        self.result = True
        self.close()
        
    def on_yes(self):
        self.result = True
        self.close()
        
    def on_no(self):
        self.result = False
        self.close()
        
    def on_string_ok(self):
        self.result = self.entry.get()
        self.close()
        
    def on_cancel(self):
        self.result = None
        self.close()
        
    def close(self):
        try: self.grab_release()
        except: pass
        self.withdraw()
        self.update_idletasks()
        try: self.quit()  # Safely exit local mainloop
        except: pass
        self.destroy()

def show_custom_dialog_subprocess(title, message, dialog_type="info"):
    import subprocess
    import json
    import sys
    
    script = f"""
import tkinter as tk
import json
import sys

class CustomTkDialog(tk.Toplevel):
    def __init__(self, parent, title, message, dialog_type="info"):
        super().__init__(parent)
        self.title(title)
        self.result = None
        
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self.attributes("-topmost", True)
        self.grab_set() # Modal dialog behavior
        
        frame = tk.Frame(self, padx=15, pady=15)
        frame.pack(fill=tk.BOTH, expand=True)
        
        lbl = tk.Label(frame, text=message, justify=tk.LEFT, wraplength=400)
        lbl.pack(side=tk.TOP, pady=(0, 15))
        
        btn_frame = tk.Frame(frame)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        if dialog_type == "info":
            btn = tk.Button(btn_frame, text="OK", command=self.on_ok, width=10, bg='lightgray')
            btn.pack(side=tk.BOTTOM, pady=5)
        elif dialog_type == "yesno":
            btn_yes = tk.Button(btn_frame, text="Yes", command=self.on_yes, width=10, bg='lightgreen')
            btn_yes.pack(side=tk.LEFT, padx=5)
            btn_no = tk.Button(btn_frame, text="No", command=self.on_no, width=10, bg='lightcoral')
            btn_no.pack(side=tk.RIGHT, padx=5)
        elif dialog_type == "okcancel":
            btn_ok = tk.Button(btn_frame, text="OK", command=self.on_ok, width=10, bg='lightgreen')
            btn_ok.pack(side=tk.LEFT, padx=5)
            btn_cancel = tk.Button(btn_frame, text="Cancel", command=self.on_cancel, width=10, bg='lightcoral')
            btn_cancel.pack(side=tk.RIGHT, padx=5)
        elif dialog_type == "askstring":
            self.entry = tk.Entry(frame, width=40)
            self.entry.pack(side=tk.TOP, pady=(0, 15))
            self.entry.focus_set()
            self.entry.bind("<Return>", lambda e: self.on_string_ok())
            
            btn_ok = tk.Button(btn_frame, text="OK", command=self.on_string_ok, width=10, bg='lightgreen')
            btn_ok.pack(side=tk.LEFT, padx=5)
            btn_cancel = tk.Button(btn_frame, text="Cancel", command=self.on_cancel, width=10, bg='lightcoral')
            btn_cancel.pack(side=tk.RIGHT, padx=5)
            
        self.update_idletasks()
        width = max(350, self.winfo_width())
        height = max(150, self.winfo_height())
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{{width}}x{{height}}+{{x}}+{{y}}')
        
    def on_ok(self):
        self.result = True
        self.close()
        
    def on_yes(self):
        self.result = True
        self.close()
        
    def on_no(self):
        self.result = False
        self.close()
        
    def on_string_ok(self):
        self.result = self.entry.get()
        self.close()
        
    def on_cancel(self):
        self.result = None
        self.close()
        
    def close(self):
        try: self.grab_release()
        except: pass
        self.withdraw()
        self.update_idletasks()
        try: self.quit()
        except: pass
        self.destroy()

try:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.deiconify()
    root.lift()
    root.update()
    root.focus_force()
    
    try:
        import os
        import sys
        if sys.platform == 'darwin':
            os.system('''osascript -e 'tell app "System Events" to set frontmost of process "Python" to true' 2>/dev/null''')
            os.system('''osascript -e 'tell app "System Events" to set frontmost of process "python3" to true' 2>/dev/null''')
            os.system('''osascript -e 'tell app "System Events" to set frontmost of process "python" to true' 2>/dev/null''')
    except:
        pass
        
    dialog = CustomTkDialog(root, {repr(title)}, {repr(message)}, dialog_type={repr(dialog_type)})
    dialog.mainloop()
    print(json.dumps({{"result": dialog.result}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
finally:
    try: root.destroy()
    except: pass
"""
    try:
        res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
        out = res.stdout.strip()
        data = json.loads(out)
        if "error" in data:
            return (False, None)
        return (True, data.get("result"))
    except Exception as e:
        return (False, None)

def show_custom_dialog(title, message, dialog_type="info"):
    if _in_jupyter():
        success, res = show_custom_dialog_subprocess(title, message, dialog_type=dialog_type)
        if success:
            return res
            
    root = get_tk_root()
    # DO NOT deiconify the root, as it causes a blank tk window to appear.
    # The CustomTkDialog is a Toplevel and handles its own topmost attribute.
    try:
        import sys
        if sys.platform == 'darwin':
            import os
            # AppleScript to bring Python to front on macOS
            os.system('osascript -e \'tell app "System Events" to set frontmost of process "Python" to true\' 2>/dev/null')
            os.system('osascript -e \'tell app "System Events" to set frontmost of process "python3" to true\' 2>/dev/null')
            os.system('osascript -e \'tell app "System Events" to set frontmost of process "python" to true\' 2>/dev/null')
    except:
        pass
    
    dialog = CustomTkDialog(root, title, message, dialog_type=dialog_type)
    dialog.mainloop()  # Run local event loop to prevent Jupyter freezes
    
    return dialog.result

def _safe_showinfo(title, message, **kwargs):
    print(f"\n[{title}] {message}\n")
    show_custom_dialog(title, message, dialog_type="info")
    return "ok"

def _safe_showerror(title, message, **kwargs):
    print(f"\n[ERROR] [{title}] {message}\n")
    show_custom_dialog(title, message, dialog_type="info")
    return "ok"

def _safe_showwarning(title, message, **kwargs):
    print(f"\n[WARNING] [{title}] {message}\n")
    show_custom_dialog(title, message, dialog_type="info")
    return "ok"

def _safe_askyesno(title, message, **kwargs):
    res = show_custom_dialog(title, message, dialog_type="yesno")
    if res is None:
        return False
    return res

def _safe_askokcancel(title, message, **kwargs):
    res = show_custom_dialog(title, message, dialog_type="okcancel")
    if res is None:
        return False
    return res

# Apply monkeypatching globally to Tkinter modules
import tkinter.messagebox as tk_messagebox
tk_messagebox.showinfo = _safe_showinfo
tk_messagebox.showerror = _safe_showerror
tk_messagebox.showwarning = _safe_showwarning
tk_messagebox.askyesno = _safe_askyesno
tk_messagebox.askokcancel = _safe_askokcancel
messagebox.showinfo = _safe_showinfo
messagebox.showerror = _safe_showerror
messagebox.showwarning = _safe_showwarning
messagebox.askyesno = _safe_askyesno
messagebox.askokcancel = _safe_askokcancel
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.widgets import SpanSelector, Button, RadioButtons
import matplotlib.tri as tri

def get_masked_triangulation(x, y, max_edge=None):
    """
    Creates a Matplotlib Triangulation object and masks triangles with long edges.
    Helps prevent 'smearing' across missing quadrants in stitched maps.
    """
    try:
        if len(x) < 3:
            return None
            
        triang = tri.Triangulation(x, y)
        
        # Calculate edge lengths
        x_tri = x[triang.triangles]
        y_tri = y[triang.triangles]
        
        e1 = np.sqrt((x_tri[:, 0] - x_tri[:, 1])**2 + (y_tri[:, 0] - y_tri[:, 1])**2)
        e2 = np.sqrt((x_tri[:, 1] - x_tri[:, 2])**2 + (y_tri[:, 1] - y_tri[:, 2])**2)
        e3 = np.sqrt((x_tri[:, 2] - x_tri[:, 0])**2 + (y_tri[:, 2] - y_tri[:, 0])**2)
        
        if max_edge is None:
            # More robust heuristic: use 5x the median edge length
            # This handles both high-res and low-res maps correctly.
            all_edges = np.concatenate([e1, e2, e3])
            max_edge = np.median(all_edges) * 5.0
            
            # Absolute sanity check: don't mask if max_edge is too small relative to total range
            range_max = max(np.max(x) - np.min(x), np.max(y) - np.min(y))
            if max_edge < range_max * 0.01:
                max_edge = range_max * 0.5 # Effectively disable masking if it seems wrong
        
        # Mask triangles where any edge is too long
        mask = (e1 > max_edge) | (e2 > max_edge) | (e3 > max_edge)
        triang.set_mask(mask)
        return triang
    except Exception as e:
        print(f"  [Warning] Masked triangulation failed: {e}")
        return None



def format_num_val(val, decimals=2):
    """Formats numeric values to a fixed number of decimal places, returning 'N/A' for non-numeric or missing values."""
    if val is None or str(val).strip() in ('N/A', 'None', ''):
        return 'N/A'
    try:
        f_val = float(val)
        return f"{f_val:.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)

_TK_ROOT = None
def get_tk_root():
    global _TK_ROOT
    if _TK_ROOT is None:
        _TK_ROOT = tk.Tk()
        _TK_ROOT.withdraw()
    return _TK_ROOT

def console_log(msg):
    import sys
    print(msg, flush=True)
    try:
        sys.__stdout__.write("[CONSOLE] " + str(msg) + "\n")
        sys.__stdout__.flush()
    except Exception:
        pass

def safe_filedialog_call_fallback(func, *args, **kwargs):
    console_log(f"safe_filedialog_call_fallback: using fallback (in-process) filedialog for {func.__name__}")
    root = get_tk_root()
    root.withdraw()
    
    # Create the temporary parent window
    temp_win = tk.Toplevel(root)
    temp_win.withdraw()
    temp_win.attributes("-alpha", 0.0)
    temp_win.attributes("-topmost", True)
    
    # Force layout update and handle mapping before showing the dialog
    temp_win.deiconify()
    temp_win.update()
    temp_win.focus_force()
    
    # Run AppleScript focus if on macOS, or equivalent OS helper if necessary
    try:
        import sys
        if sys.platform == 'darwin':
            import os
            os.system('osascript -e \'tell app "System Events" to set frontmost of process "Python" to true\' 2>/dev/null')
    except:
        pass
    
    kwargs['parent'] = temp_win
    if func.__name__ == 'asksaveasfilename':
        kwargs['confirmoverwrite'] = True
    
    # Call the dialog function
    console_log(f"safe_filedialog_call_fallback: calling {func.__name__} in-process")
    result = func(*args, **kwargs)
    console_log(f"safe_filedialog_call_fallback: returned {result}")
    
    # Clean up
    temp_win.destroy()
    root.update()
    
    return result

def safe_filedialog_call(func, *args, **kwargs):
    """
    Safely invokes a Tkinter filedialog function (e.g. asksaveasfilename, askopenfilename, askdirectory)
    in a Jupyter notebook environment by launching a subprocess to avoid cross-thread hangs.
    """
    import subprocess
    import json
    import sys
    import os
    
    # Extract common arguments
    title = kwargs.get('title', '')
    initialdir = kwargs.get('initialdir', '')
    initialfile = kwargs.get('initialfile', '')
    defaultextension = kwargs.get('defaultextension', '')
    filetypes = kwargs.get('filetypes', [])
    
    # Format script based on the function type
    func_name = func.__name__
    
    console_log(f"safe_filedialog_call: launching subprocess for {func_name}")
    console_log(f"  title: {title}")
    console_log(f"  initialdir: {initialdir}")
    console_log(f"  initialfile: {initialfile}")
    
    script = f"""
import tkinter as tk
from tkinter import filedialog
import json
import sys

try:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.deiconify()
    root.update()
    root.focus_force()
    
    if "{func_name}" == "asksaveasfilename":
        path = filedialog.asksaveasfilename(
            title={repr(title)},
            initialdir={repr(initialdir)},
            initialfile={repr(initialfile)},
            defaultextension={repr(defaultextension)},
            filetypes={repr(filetypes)},
            confirmoverwrite=True
        )
    elif "{func_name}" == "askopenfilename":
        path = filedialog.askopenfilename(
            title={repr(title)},
            initialdir={repr(initialdir)},
            filetypes={repr(filetypes)}
        )
    elif "{func_name}" == "askopenfilenames":
        path = filedialog.askopenfilenames(
            title={repr(title)},
            initialdir={repr(initialdir)},
            filetypes={repr(filetypes)}
        )
    elif "{func_name}" == "askdirectory":
        path = filedialog.askdirectory(
            title={repr(title)},
            initialdir={repr(initialdir)}
        )
    else:
        path = ""
        
    print(json.dumps({{"path": path}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
finally:
    try: root.destroy()
    except: pass
"""
    try:
        # Run python in a subprocess using the current virtual environment's executable
        console_log(f"  Executing subprocess with python: {sys.executable}")
        res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
        out = res.stdout.strip()
        console_log(f"  Subprocess finished. Raw stdout: {repr(out)}")
        if res.stderr:
            console_log(f"  Subprocess stderr: {repr(res.stderr)}")
        data = json.loads(out)
        if "error" in data:
            console_log(f"Subprocess filedialog error: {data['error']}")
            return None
        console_log(f"  Subprocess returned path: {data.get('path')}")
        return data.get("path")
    except Exception as e:
        console_log(f"Failed to launch subprocess filedialog: {e}. Falling back to in-process dialog.")
        import traceback
        try:
            sys.__stderr__.write(traceback.format_exc() + "\n")
            sys.__stderr__.flush()
        except:
            pass
        return safe_filedialog_call_fallback(func, *args, **kwargs)

_GLOBAL_METADATA_MEMORY = {}
_METADATA_FILE = os.path.join(os.path.expanduser('~'), '.sgm_last_metadata.json')

def load_last_metadata():
    """Loads the last used research metadata from memory cache or persistent file."""
    global _GLOBAL_METADATA_MEMORY
    if _GLOBAL_METADATA_MEMORY:
        return dict(_GLOBAL_METADATA_MEMORY)
    
    try:
        if os.path.exists(_METADATA_FILE) and os.path.getsize(_METADATA_FILE) > 0:
            with open(_METADATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and data:
                    _GLOBAL_METADATA_MEMORY = dict(data)
                    return dict(data)
    except Exception as e:
        console_log(f"Warning loading metadata cache: {e}")
    return {}

def save_last_metadata(data):
    """Saves the latest research metadata to memory and persistent file atomically."""
    global _GLOBAL_METADATA_MEMORY
    if isinstance(data, dict) and data:
        # Preserve existing valid entries in memory cache if new input is N/A or empty
        for k, v in data.items():
            if v and str(v).strip() not in ("N/A", "None", "null", ""):
                _GLOBAL_METADATA_MEMORY[k] = str(v).strip()
            elif k in ("Name", "Formula"):
                _GLOBAL_METADATA_MEMORY[k] = str(v).strip()

        # Atomic file write
        try:
            tmp_file = _METADATA_FILE + ".tmp"
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(_GLOBAL_METADATA_MEMORY, f, indent=2)
            os.replace(tmp_file, _METADATA_FILE)
        except Exception as e:
            console_log(f"Warning saving metadata cache: {e}")

def safe_metadata_dialog_call(initial_data=None):
    """
    Safely invokes the MetadataDialog in a separate python subprocess
    to prevent Jupyter thread locks and GUI freezes on Windows.
    Pre-populates entries with persistent last-used research metadata.
    """
    import subprocess
    import json
    import sys
    import os
    import tkinter as tk
    
    # Load persistent last metadata and overlay initial_data (e.g. current scan Name)
    merged_initial = load_last_metadata()
    if initial_data:
        for k, v in initial_data.items():
            if v and str(v).strip() not in ("N/A", "None", "null", ""):
                merged_initial[k] = str(v).strip()
        
    script = f"""
import tkinter as tk
from tkinter import simpledialog
import json
import sys

class MetadataDialog(simpledialog.Dialog):
    def __init__(self, parent, title, initial_data=None):
        self.initial_data = initial_data or {{}}
        super().__init__(parent, title)

    def body(self, master):
        fields = [
            ("Sample Name", "Name"),
            ("Sample Formula", "Formula"),
            ("Authors", "Authors"),
            ("Affiliation", "Affiliation"),
            ("Element", "Element"),
            ("Edge", "Edge"),
            ("Preparation Method", "Prep"),
            ("Calibrated To", "Calib"),
            ("Calibration Reference", "CalibRef"),
            ("Temperature", "Temp"),
            ("Scan Mode", "Mode"),
            ("Chamber Conditions", "Chamber"),
            ("Comments", "Comments")
        ]
        self.entries = {{}}
        for i, (label_text, key) in enumerate(fields):
            tk.Label(master, text=f"{{label_text}}:").grid(row=i, column=0, sticky='w', padx=5, pady=2)
            entry = tk.Entry(master, width=40)
            entry.grid(row=i, column=1, padx=5, pady=2)
            val = self.initial_data.get(key)
            if val and str(val).strip() not in ("N/A", "None", "null", ""):
                entry.insert(0, str(val).strip())
            self.entries[key] = entry
        return self.entries["Name"]

    def apply(self):
        self.result = {{key: (entry.get().strip() or "N/A") for key, entry in self.entries.items()}}

try:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.deiconify()
    root.update()
    root.focus_force()
    
    initial_data = {repr(merged_initial)}
    d = MetadataDialog(root, "Research Metadata Input", initial_data=initial_data)
    if hasattr(d, 'result') and d.result:
        print(json.dumps({{"result": d.result}}))
    else:
        print(json.dumps({{"result": None}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
finally:
    try: root.destroy()
    except: pass
"""
    try:
        console_log("safe_metadata_dialog_call: launching subprocess...")
        res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
        out = res.stdout.strip()
        console_log(f"safe_metadata_dialog_call: subprocess finished. stdout: {repr(out)}")
        if res.stderr:
            console_log(f"safe_metadata_dialog_call: subprocess stderr: {repr(res.stderr)}")
        data = json.loads(out)
        if "error" in data:
            console_log(f"safe_metadata_dialog_call subprocess error: {data['error']}")
            return None
        res_dict = data.get("result")
        if res_dict:
            save_last_metadata(res_dict)
        return res_dict
    except Exception as e:
        console_log(f"Failed to launch subprocess metadata dialog: {e}. Falling back to in-process.")
        import traceback
        try:
            sys.__stderr__.write(traceback.format_exc() + "\n")
            sys.__stderr__.flush()
        except:
            pass
        
        # Fallback to in-process dialog
        root = get_tk_root()
        root.withdraw()
        temp_win = tk.Toplevel(root)
        temp_win.withdraw()
        temp_win.attributes("-alpha", 0.0)
        temp_win.attributes("-topmost", True)
        temp_win.deiconify()
        temp_win.update()
        temp_win.focus_force()
        from tkinter import simpledialog
        class LocalMetadataDialog(simpledialog.Dialog):
            def __init__(self, parent, title, initial_data=None):
                self.initial_data = initial_data or {}
                super().__init__(parent, title)

            def body(self, master):
                fields = [
                    ("Sample Name", "Name"),
                    ("Sample Formula", "Formula"),
                    ("Authors", "Authors"),
                    ("Affiliation", "Affiliation"),
                    ("Element", "Element"),
                    ("Edge", "Edge"),
                    ("Preparation Method", "Prep"),
                    ("Calibrated To", "Calib"),
                    ("Calibration Reference", "CalibRef"),
                    ("Temperature", "Temp"),
                    ("Scan Mode", "Mode"),
                    ("Chamber Conditions", "Chamber"),
                    ("Comments", "Comments")
                ]
                self.entries = {}
                for i, (label_text, key) in enumerate(fields):
                    tk.Label(master, text=f"{label_text}:").grid(row=i, column=0, sticky='w', padx=5, pady=2)
                    entry = tk.Entry(master, width=40)
                    entry.grid(row=i, column=1, padx=5, pady=2)
                    val = self.initial_data.get(key)
                    if val and str(val).strip() not in ("N/A", "None", "null", ""):
                        entry.insert(0, str(val).strip())
                    self.entries[key] = entry
                return self.entries["Name"]

            def apply(self):
                self.result = {key: (entry.get().strip() or "N/A") for key, entry in self.entries.items()}

        d = LocalMetadataDialog(temp_win, "Research Metadata Input", initial_data=merged_initial)
        res = d.result if hasattr(d, 'result') else None
        temp_win.destroy()
        if res:
            save_last_metadata(res)
        return res

def get_safe_save_path(save_dir, default_name):
    """
    Auto-increments the file suffix if the file already exists to prevent Jupyter freezes
    caused by Tkinter simpledialog.
    """
    save_path = os.path.join(save_dir, default_name)
    if not os.path.exists(save_path):
        return save_path
    
    base, ext = os.path.splitext(default_name)
    try:
        while True:
            suffix = show_custom_dialog("File Exists", 
                                        f"'{default_name}' already exists in the folder.\n\n"
                                        "Please enter a suffix to append (e.g., '_v2', '_new'), or leave blank to overwrite:",
                                        dialog_type="askstring")
            
            # User clicked Cancel
            if suffix is None:
                return None
                
            # User left blank -> Overwrite
            if suffix.strip() == "":
                return save_path
                
            # Try new name
            new_name = f"{base}{suffix}{ext}"
            new_path = os.path.join(save_dir, new_name)
            if not os.path.exists(new_path):
                return new_path
            
            # If new name also exists, loop continues
            default_name = new_name
            base, ext = os.path.splitext(default_name)
    except Exception as e:
        print(f"  [Custom Dialog Error] {e}. Falling back to auto-increment.")
        counter = 1
        while True:
            new_name = f"{base}_{counter}{ext}"
            new_path = os.path.join(save_dir, new_name)
            if not os.path.exists(new_path):
                return new_path
            counter += 1

def apply_spatial_trim(x, y, data, x_trim=0.0, y_trim=0.0):
    """
    Filters coordinates and data based on a distance (in mm) from the min/max of each axis.
    """
    if x_trim <= 0 and y_trim <= 0:
        return x, y, data
        
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    mask = (x >= x_min + x_trim) & (x <= x_max - x_trim) & \
           (y >= y_min + y_trim) & (y <= y_max - y_trim)
           
    return x[mask], y[mask], data[mask]

def get_sdd_intensity_map(file_path, x_coords, y_coords, channel_roi=None, xrf_roi=None, det_name=None, calib_data=None, roll_shift=0, x_trim=0.0, y_trim=0.0):
    """
    Loads raw SDD binary data, applies roll shift and spatial trim, and returns the 2D intensity map.
    Supports channel_roi=(ch1, ch2) or xrf_roi=(min_eV, max_eV) with optional detector energy calibration.
    """
    if not os.path.exists(file_path):
        return None, None
        
    try:
        pixels_per_spectrum = 256
        data_1d = np.fromfile(file_path, dtype=np.uint32)
        
        num_spectra = len(data_1d) // pixels_per_spectrum
        
        # Truncate to match coordinates
        num_points = min(num_spectra, x_coords.size)
        intensity = data_1d[:num_points * pixels_per_spectrum].reshape((num_points, pixels_per_spectrum))
        
        # Apply roll shift
        if roll_shift != 0:
            intensity = np.roll(intensity, shift=roll_shift, axis=0)
            
        # Resolve channel ROI bounds
        if xrf_roi is not None:
            min_e, max_e = float(xrf_roi[0]), float(xrf_roi[1])
            if calib_data and det_name and det_name in calib_data:
                gain = calib_data[det_name].get('gain', 1.0)
                offset = calib_data[det_name].get('offset', 0.0)
                if gain != 1.0 or offset != 0.0:
                    ch_start = int(max(0, np.floor((min_e - offset) / gain)))
                    ch_end = int(min(256, np.ceil((max_e - offset) / gain)))
                else:
                    ch_start = int(max(0, np.floor(min_e / 10.0)))
                    ch_end = int(min(256, np.ceil(max_e / 10.0)))
            else:
                ch_start = int(max(0, np.floor(min_e / 10.0)))
                ch_end = int(min(256, np.ceil(max_e / 10.0)))
        elif channel_roi is not None:
            ch_start, ch_end = int(channel_roi[0]), int(channel_roi[1])
        else:
            ch_start, ch_end = 80, 120

        # Ensure valid range
        ch_start = max(0, min(255, ch_start))
        ch_end = max(ch_start + 1, min(256, ch_end))

        # Sum ROI
        roi_sum = np.sum(intensity[:, ch_start:ch_end], axis=1)
        
        # Apply spatial trim
        trimmed_x, trimmed_y, trimmed_intensity = apply_spatial_trim(
            x_coords[:num_points], y_coords[:num_points], roi_sum, x_trim, y_trim
        )
        
        return trimmed_intensity, (trimmed_x, trimmed_y)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def auto_align_shift(target_img, ref_img, max_shift=50):
    """
    Finds the roll_shift that maximizes correlation between target and reference images.
    Input images are expected to be 1D flattened ROI sums.
    """
    best_shift = 0
    max_corr = -1.0
    
    # We test shifts in the range [-max_shift, max_shift]
    for s in range(-max_shift, max_shift + 1):
        shifted = np.roll(target_img, shift=s)
        corr = np.corrcoef(shifted, ref_img)[0, 1]
        if corr > max_corr:
            max_corr = corr
            best_shift = s
            
    return best_shift, max_corr

def interactive_roll_align(path_pack, channel_roi=None, max_shift=100, use_color=True, xrf_roi=None):
    """
    Creates an interactive widget to adjust roll_shift and spatial trim in a Jupyter Notebook.
    Handles both 'analyze_map' and 'analyze_stack' data packets.
    Supports channel_roi=(ch1, ch2) or xrf_roi=(min_eV, max_eV).
    """
    if path_pack is None:
        print("Error: path_pack is None. Please check if the analysis function (analyze_stack or analyze_map) succeeded.")
        return

    # Check if channel_roi was passed positionally as an xrf_roi (energy in eV > 255)
    if channel_roi is not None and xrf_roi is None:
        if isinstance(channel_roi, (tuple, list)) and len(channel_roi) >= 2:
            if max(channel_roi) > 255:
                xrf_roi = channel_roi
                channel_roi = None

    # Fallback to path_pack if both are None
    if xrf_roi is None and channel_roi is None:
        if 'xrf_roi' in path_pack:
            xrf_roi = path_pack['xrf_roi']
        elif 'channel_roi' in path_pack:
            channel_roi = path_pack['channel_roi']
        else:
            channel_roi = (80, 120)

    # Load energy calibration data if available
    try:
        import sdd_calibration_utils as sdd_calib
        calib_data = sdd_calib.load_calibration()
    except Exception:
        calib_data = None

    x = path_pack.get('x', np.array([]))
    y = path_pack.get('y', np.array([]))
    sdd_files = path_pack.get('sdd_files', {})
    
    if not sdd_files:
        print("No SDD files found in path_pack.")
        return

    # Determine if it's a stack or a map
    is_stack = isinstance(next(iter(sdd_files.values())), dict)
    
    detectors = sorted(sdd_files.keys())
    energies = []
    if is_stack:
        # Get energies from the first detector's dict
        energies = sorted(sdd_files[detectors[0]].keys())
    
    # --- UI Components ---
    det_dropdown = widgets.Dropdown(options=detectors, description='Detector:')
    en_dropdown = widgets.Dropdown(options=energies, description='Energy (eV):') if is_stack else None
    ref_en_dropdown = widgets.Dropdown(options=energies, description='Ref Energy:') if is_stack else None
    shift_slider = widgets.IntSlider(value=0, min=-max_shift, max=max_shift, step=1, description='Roll Shift:', layout=widgets.Layout(width='50%'))
    
    # Spatial Trim Sliders - expanded to allow full scan range
    x_range = np.max(x) - np.min(x) if x.size > 0 else 1.0
    y_range = np.max(y) - np.min(y) if y.size > 0 else 1.0
    x_trim_slider = widgets.FloatSlider(value=0.0, min=0.0, max=x_range*0.99, step=x_range/200, description='X-Trim (mm):', layout=widgets.Layout(width='50%'))
    y_trim_slider = widgets.FloatSlider(value=0.0, min=0.0, max=y_range*0.99, step=y_range/200, description='Y-Trim (mm):', layout=widgets.Layout(width='50%'))
    
    color_toggle = widgets.Checkbox(value=use_color, description='Use Color', indent=False)
    
    # Contrast Slider
    contrast_slider = widgets.FloatRangeSlider(
        value=[0, 100], min=0, max=100, step=0.1,
        description='Contrast %:', layout=widgets.Layout(width='80%')
    )
    
    auto_btn = widgets.Button(description="Auto-Align", button_style='info')
    save_btn = widgets.Button(description="Save Current Map", button_style='success')
    output = widgets.Output()

    fig_id = f"align_fig_{id(path_pack)}"
    
    def update_plot(change=None):
        with output:
            det = det_dropdown.value
            shift = shift_slider.value
            xt = x_trim_slider.value
            yt = y_trim_slider.value
            use_color = color_toggle.value
            
            if is_stack:
                en = en_dropdown.value
                f_path = sdd_files[det].get(en)
                title = f"{det} at {en:.2f} eV\nShift: {shift} | Trim: X={xt}, Y={yt}"
            else:
                f_path = sdd_files[det]
                title = f"{det}\nShift: {shift} | Trim: X={xt}, Y={yt}"
                
            intensity, coords = get_sdd_intensity_map(
                f_path, x, y,
                channel_roi=channel_roi,
                xrf_roi=xrf_roi,
                det_name=det,
                calib_data=calib_data,
                roll_shift=shift,
                x_trim=xt,
                y_trim=yt
            )
            
            path_pack['roll_shift'] = shift
            path_pack['x_trim'] = xt
            path_pack['y_trim'] = yt
            if xrf_roi is not None:
                path_pack['xrf_roi'] = xrf_roi
            if channel_roi is not None:
                path_pack['channel_roi'] = channel_roi
            
            if intensity is not None:
                # Reuse or create figure
                if not (hasattr(update_plot, 'fig') and plt.fignum_exists(fig_id)):
                    output.clear_output(wait=True)
                    fig, ax = plt.subplots(1, 2, figsize=(9, 4), num=fig_id)
                    update_plot.fig = fig
                    update_plot.ax = ax
                else:
                    fig = update_plot.fig
                    ax = update_plot.ax
                    for a in ax: a.clear()

                # --- Left: 2D Intensity Map ---
                map_ax = ax[0]
                cmap_name = 'viridis' if use_color else 'gray'
                
                # Calculate contrast limits
                p_low, p_high = contrast_slider.value
                vmin = np.percentile(intensity, p_low)
                vmax = np.percentile(intensity, p_high)
                if vmin == vmax: vmax = vmin + 1
                
                try:
                    triang = get_masked_triangulation(coords[0], coords[1])
                    if triang is not None:
                        sc = map_ax.tripcolor(triang, intensity, shading='gouraud', 
                                             edgecolors='none', rasterized=True, cmap=cmap_name,
                                             vmin=vmin, vmax=vmax)
                    else:
                        sc = map_ax.tripcolor(coords[0], coords[1], intensity, shading='gouraud', 
                                             edgecolors='none', rasterized=True, cmap=cmap_name,
                                             vmin=vmin, vmax=vmax)
                    # Clear existing colorbars for this axis
                    if hasattr(update_plot, 'cbar') and update_plot.cbar is not None:
                        try:
                            update_plot.cbar.remove()
                        except:
                            pass
                    update_plot.cbar = plt.colorbar(sc, ax=map_ax, label='Counts')
                except Exception as e:
                    map_ax.text(0.5, 0.5, f"Plot error: {e}", transform=map_ax.transAxes)

                map_ax.set_title(title, fontsize='small')
                map_ax.set_xlabel('X (mm)')
                map_ax.set_ylabel('Y (mm)')
                map_ax.set_aspect('equal')
                
                map_roi = path_pack.get('map_roi')
                if map_roi is not None:
                    x1, x2 = sorted(map_roi[0:2])
                    y1, y2 = sorted(map_roi[2:4])
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=1.5, ec='r', fc='none', ls='--')
                    map_ax.add_patch(rect)
                
                # --- Right: Count Distribution Histogram ---
                hist_ax = ax[1]
                hist_ax.hist(intensity, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
                hist_ax.set_title(f"Intensity Distribution", fontsize='small')
                hist_ax.set_xlabel('Total Counts')
                hist_ax.set_ylabel('Pixel Frequency')
                hist_ax.grid(True, linestyle=':', alpha=0.6)
                
                plt.tight_layout()
                fig.canvas.draw_idle()
            else:
                print("Failed to load map data.")

    def run_auto(b):
        if not is_stack:
            print("Auto-align requires multiple images (Stack file required).")
            return
            
        with output:
            det = det_dropdown.value
            target_en = en_dropdown.value
            ref_en = ref_en_dropdown.value
            xt = x_trim_slider.value
            yt = y_trim_slider.value
            
            if target_en == ref_en:
                print("Target and Reference must be different energies.")
                return
            
            print(f"Finding best shift relative to {ref_en:.2f} eV (using current trim)...")
            
            target_path = sdd_files[det].get(target_en)
            ref_path = sdd_files[det].get(ref_en)
            
            target_data, _ = get_sdd_intensity_map(
                target_path, x, y,
                channel_roi=channel_roi,
                xrf_roi=xrf_roi,
                det_name=det,
                calib_data=calib_data,
                roll_shift=0,
                x_trim=xt,
                y_trim=yt
            )
            ref_data, _ = get_sdd_intensity_map(
                ref_path, x, y,
                channel_roi=channel_roi,
                xrf_roi=xrf_roi,
                det_name=det,
                calib_data=calib_data,
                roll_shift=0,
                x_trim=xt,
                y_trim=yt
            )
            
            if target_data is not None and ref_data is not None:
                best_s, corr = auto_align_shift(target_data, ref_data, max_shift=max_shift)
                print(f"Found Optimal Shift: {best_s} (Correlation: {corr:.3f})")
                shift_slider.value = best_s
            else:
                print("Error loading data for auto-alignment.")
                
    def run_save(b):
        """Saves current interactive view (clean PNG)."""
        det = det_dropdown.value
        shift = shift_slider.value
        xt = x_trim_slider.value
        yt = y_trim_slider.value
        use_color = color_toggle.value
        
        if is_stack:
            en = en_dropdown.value
            f_path = sdd_files[det].get(en)
            prefix = f"aligned_{det}_{en:.2f}eV"
        else:
            f_path = sdd_files[det]
            prefix = f"aligned_{det}"
            
        intensity, coords = get_sdd_intensity_map(
            f_path, x, y,
            channel_roi=channel_roi,
            xrf_roi=xrf_roi,
            det_name=det,
            calib_data=calib_data,
            roll_shift=shift,
            x_trim=xt,
            y_trim=yt
        )
        
        if intensity is not None:
            h5_path = path_pack.get('h5_file_path')
            save_dir = os.path.abspath(os.path.dirname(h5_path)) if h5_path else os.getcwd()
            if xrf_roi is not None:
                default_name = f"{prefix}_XRF_ROI_{xrf_roi[0]:.1f}-{xrf_roi[1]:.1f}eV.png"
            else:
                default_name = f"{prefix}_ROI_{channel_roi[0]}-{channel_roi[1]}.png"
            save_filename = get_safe_save_path(save_dir, default_name)
            
            if not save_filename:
                print("    [CANCEL] Save cancelled by user.")
                return

            clean_fig = Figure(figsize=(6, 6))
            canvas = FigureCanvasAgg(clean_fig)
            clean_ax = clean_fig.add_subplot(111)
            triang = get_masked_triangulation(coords[0], coords[1])
            if triang is not None:
                clean_ax.tripcolor(triang, intensity, shading='gouraud', 
                                 edgecolors='none', rasterized=True, cmap='viridis')
            else:
                clean_ax.tripcolor(coords[0], coords[1], intensity, shading='gouraud', 
                                 edgecolors='none', rasterized=True, cmap='viridis')
            clean_ax.set_aspect('equal')
            clean_ax.axis('off')
            clean_fig.savefig(save_filename, bbox_inches='tight', pad_inches=0, transparent=True)
            print(f"    -> [SAVE] Image saved to: {save_filename}")
            
            root_fin = get_tk_root()
            root_fin.attributes("-topmost", True)
            messagebox.showinfo("Save Successful", f"Image saved to:\n{save_filename}", parent=root_fin)
        else:
            with output:
                print("Error: Could not load data for saving.")

    # Observers
    det_dropdown.observe(update_plot, names='value')
    shift_slider.observe(update_plot, names='value')
    x_trim_slider.observe(update_plot, names='value')
    y_trim_slider.observe(update_plot, names='value')
    color_toggle.observe(update_plot, names='value')
    contrast_slider.observe(update_plot, names='value')
    if is_stack:
        en_dropdown.observe(update_plot, names='value')
    auto_btn.on_click(run_auto)
    save_btn.on_click(run_save)

    header = widgets.HBox([det_dropdown, en_dropdown]) if is_stack else widgets.HBox([det_dropdown])
    trim_controls = widgets.VBox([x_trim_slider, y_trim_slider])
    align_controls = widgets.HBox([ref_en_dropdown, auto_btn, save_btn]) if is_stack else widgets.HBox([save_btn])
    
    display(widgets.VBox([header, shift_slider, trim_controls, color_toggle, contrast_slider, align_controls, output]))
    update_plot()

def interactive_channel_selector(path_pack, initial_roi=(20, 40), use_calibration=False, calib_data=None, default_detector='sdd3'):
    """
    Opens an interactive XRF spectrum plot to select the channel / energy ROI.
    Supports displaying calibrated energy (eV) on the x-axis alongside channel numbers.
    Defaults to displaying SDD3 (or first available detector) with detector switching.
    Returns the selected (start, end) channel tuple.
    """
    if not path_pack:
        print("Error: path_pack is None.")
        return initial_roi

    # 1. Load available detectors and spectrum data
    rep_e = path_pack.get('representative_energy')
    sdd_files = path_pack.get('sdd_files', {})
    if not sdd_files:
        print("Error: No SDD files found in path_pack.")
        return initial_roi
    
    available_dets = sorted(sdd_files.keys())
    active_det = default_detector if default_detector in available_dets else available_dets[0]

    def load_detector_spectrum(det_name):
        f_path = sdd_files[det_name].get(rep_e) if isinstance(sdd_files[det_name], dict) else sdd_files[det_name]
        if not f_path or not os.path.exists(f_path):
            if isinstance(sdd_files[det_name], dict) and sdd_files[det_name]:
                alt_e = next(iter(sdd_files[det_name]))
                f_path = sdd_files[det_name][alt_e]
            else:
                return None
        try:
            data_1d = np.fromfile(f_path, dtype=np.uint32)
            num_s = min(len(data_1d) // 256, path_pack.get('x', np.array([])).size)
            s2d = data_1d[:num_s * 256].reshape((num_s, 256))
            return np.sum(s2d, axis=0)
        except Exception:
            return None

    total_spec = load_detector_spectrum(active_det)
    if total_spec is None:
        # Fallback to any detector that loads
        for fallback_det in available_dets:
            total_spec = load_detector_spectrum(fallback_det)
            if total_spec is not None:
                active_det = fallback_det
                break
        if total_spec is None:
            print("Error: Could not load XRF spectrum for ROI selection.")
            return initial_roi

    # Determine calibration parameters for active detector
    gain, offset = 1.0, 0.0
    is_calibrated = False
    if use_calibration and calib_data and active_det in calib_data:
        gain = calib_data[active_det].get("gain", 1.0)
        offset = calib_data[active_det].get("offset", 0.0)
        is_calibrated = True

    channels = np.arange(256)
    x_data = (gain * channels + offset) if is_calibrated else channels

    # 2. UI Setup
    fig, ax = plt.subplots(figsize=(10, 4.5))
    plt.subplots_adjust(bottom=0.20, right=0.82)

    line, = ax.plot(x_data, total_spec, color='blue', lw=1.5)
    
    scan_name = path_pack.get('scan_name', 'Scan')
    
    if is_calibrated:
        ax.set_xlabel("Calibrated Energy (eV)", fontsize='medium', fontweight='bold')
        ax.set_title(f"XRF ROI Selector ({active_det.upper()}): {scan_name}\n(Drag to select range, click 'Confirm' to finish)", fontsize='medium')
    else:
        ax.set_xlabel("Channel Index", fontsize='medium', fontweight='bold')
        ax.set_title(f"Channel ROI Selector ({active_det.upper()}): {scan_name}\n(Drag to select range, click 'Confirm' to finish)", fontsize='medium')

    ax.set_ylabel("Total Counts", fontsize='medium', fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)

    # Visual span region setup with SpanSelector
    selected_roi = [initial_roi[0], initial_roi[1]]
    
    # Calculate span start/end in plot coordinates
    span_x1 = (gain * selected_roi[0] + offset) if is_calibrated else selected_roi[0]
    span_x2 = (gain * selected_roi[1] + offset) if is_calibrated else selected_roi[1]

    # Info text box below plot
    info_text = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8, ec="orange"))

    def update_info_text():
        ch_s, ch_e = selected_roi[0], selected_roi[1]
        if is_calibrated:
            e_s = gain * ch_s + offset
            e_e = gain * ch_e + offset
            info_text.set_text(f"ROI: {e_s:.1f} eV to {e_e:.1f} eV  (Ch {ch_s} - {ch_e})")
        else:
            info_text.set_text(f"ROI: Channels {ch_s} to {ch_e}")

    update_info_text()

    def onselect(xmin, xmax):
        if is_calibrated:
            # Convert energy bounds back to channel indices
            ch_a = (xmin - offset) / gain if gain != 0 else xmin
            ch_b = (xmax - offset) / gain if gain != 0 else xmax
            ch_start = int(max(0, np.floor(min(ch_a, ch_b))))
            ch_end = int(min(255, np.ceil(max(ch_a, ch_b))))
        else:
            ch_start = int(max(0, np.floor(min(xmin, xmax))))
            ch_end = int(min(255, np.ceil(max(xmin, xmax))))

        selected_roi[0] = ch_start
        selected_roi[1] = ch_end

        update_info_text()
        fig.canvas.draw_idle()

    # SpanSelector with initial extents pre-loaded (no static axvspan ghost rectangle)
    ax._span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                            props=dict(alpha=0.3, facecolor='red'), interactive=True)
    try:
        ax._span.extents = (span_x1, span_x2)
    except Exception:
        pass

    # 3. Add Detector Switching RadioButtons if multiple detectors exist
    if len(available_dets) > 1:
        ax_radio = fig.add_axes([0.84, 0.40, 0.14, 0.35])
        active_idx = available_dets.index(active_det) if active_det in available_dets else 0
        radio = RadioButtons(ax_radio, [d.upper() for d in available_dets], active=active_idx)
        
        def set_detector(label):
            nonlocal active_det, gain, offset, is_calibrated, x_data, total_spec
            active_det = label.lower()
            new_spec = load_detector_spectrum(active_det)
            if new_spec is not None:
                total_spec = new_spec
                if use_calibration and calib_data and active_det in calib_data:
                    gain = calib_data[active_det].get("gain", 1.0)
                    offset = calib_data[active_det].get("offset", 0.0)
                    is_calibrated = True
                else:
                    gain, offset = 1.0, 0.0
                    is_calibrated = False

                x_data = (gain * channels + offset) if is_calibrated else channels
                line.set_xdata(x_data)
                line.set_ydata(total_spec)

                if is_calibrated:
                    ax.set_xlabel("Calibrated Energy (eV)", fontsize='medium', fontweight='bold')
                    ax.set_title(f"XRF ROI Selector ({active_det.upper()}): {scan_name}\n(Drag to select range, click 'Confirm' to finish)", fontsize='medium')
                else:
                    ax.set_xlabel("Channel Index", fontsize='medium', fontweight='bold')
                    ax.set_title(f"Channel ROI Selector ({active_det.upper()}): {scan_name}\n(Drag to select range, click 'Confirm' to finish)", fontsize='medium')

                ax.relim()
                ax.autoscale_view()
                update_info_text()
                fig.canvas.draw_idle()

        radio.on_clicked(set_detector)
        ax._radio = radio # Keep reference

    # 4. Confirm Button
    confirmed = False
    def confirm(event):
        nonlocal confirmed
        confirmed = True
        btn.label.set_text("✓ Confirmed!")
        btn.color = 'lime'
        fig.canvas.draw_idle()
        plt.close(fig)
        try:
            fig.canvas.stop_event_loop()
        except Exception:
            pass

    ax_btn = fig.add_axes([0.84, 0.15, 0.14, 0.10])
    btn = Button(ax_btn, 'Confirm ROI', color='lightgreen', hovercolor='palegreen')
    btn.on_clicked(confirm)
    ax._btn = btn # Keep reference

    plt.show()

    e_str = f" ({gain * selected_roi[0] + offset:.1f} eV - {gain * selected_roi[1] + offset:.1f} eV)" if is_calibrated else ""
    print(f"  [ROI Selector] Final Selection ({active_det.upper()}): Channels {selected_roi[0]} to {selected_roi[1]}{e_str}")
    return (selected_roi[0], selected_roi[1])

def grid_interpolate_map(x, y, z, resolution=200, map_roi=None):
    """
    Interpolates scattered (x, y, z) data onto a regular grid for smooth rendering.
    Returns (grid_x, grid_y, grid_z).
    """
    if map_roi:
        x1, x2 = sorted(map_roi[0:2])
        y1, y2 = sorted(map_roi[2:4])
    else:
        x1, x2 = np.min(x), np.max(x)
        y1, y2 = np.min(y), np.max(y)
        
    xi = np.linspace(x1, x2, resolution)
    yi = np.linspace(y1, y2, resolution)
    grid_x, grid_y = np.meshgrid(xi, yi)
    
    # Grid the data
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
    
    return xi, yi, grid_z
def visualize_stitching_overlap(data_packs):
    """
    Plots the coordinates of multiple data packs on a single axis to visualize overlap.
    """
    plt.figure(figsize=(10, 8))
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for i, dp in enumerate(data_packs):
        c = colors[i % len(colors)]
        plt.scatter(dp['x'], dp['y'], s=1, color=c, alpha=0.5, label=dp['scan_name'])
        
        # Draw bounding box
        x1, x2 = np.min(dp['x']), np.max(dp['x'])
        y1, y2 = np.min(dp['y']), np.max(dp['y'])
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=2, ec=c, fc='none')
        plt.gca().add_patch(rect)
        
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title('Map Overlap Visualization')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()
