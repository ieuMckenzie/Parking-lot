"""Simple tkinter GUI for the Parking Lot Scanner."""

import glob
import io
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def discover_video_devices() -> list[str]:
    """Find /dev/video* devices available on Linux."""
    return sorted(glob.glob("/dev/video*"))


def discover_models() -> list[str]:
    """Find .pt model files under the project models/ directory."""
    models_dir = os.path.join(PROJECT_ROOT, "models")
    found = []
    for root, _dirs, files in os.walk(models_dir):
        for f in files:
            if f.endswith(".pt"):
                found.append(os.path.join(root, f))
    return sorted(found)


class LogRedirector(io.TextIOBase):
    """Redirect stdout/stderr writes into a tkinter Text widget (thread-safe)."""

    def __init__(self, text_widget: tk.Text):
        self.text_widget = text_widget

    def write(self, msg: str):
        if msg.strip():
            self.text_widget.after(0, self._append, msg)
        return len(msg)

    def _append(self, msg: str):
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, msg + "\n")
        self.text_widget.see(tk.END)
        self.text_widget.configure(state="disabled")

    def flush(self):
        pass


class ScannerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Parking Lot Scanner")
        self.root.geometry("720x660")
        self.root.minsize(600, 550)

        self.engine = None
        self.engine_thread = None

        self._build_ui()
        self._redirect_output()

    # ── UI Construction ─────────────────────────────────────────────

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # --- Model selection ---
        model_frame = ttk.LabelFrame(self.root, text="YOLO Model")
        model_frame.pack(fill="x", **pad)

        self.model_var = tk.StringVar()
        models = discover_models()
        best = [m for m in models if m.endswith("best.pt")]
        if best:
            self.model_var.set(best[0])
        elif models:
            self.model_var.set(models[0])

        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, values=models, width=70)
        model_combo.pack(side="left", fill="x", expand=True, padx=(8, 4), pady=6)
        ttk.Button(model_frame, text="Browse…", command=self._browse_model).pack(side="right", padx=(0, 8), pady=6)

        # --- Source selection ---
        source_frame = ttk.LabelFrame(self.root, text="Input Source")
        source_frame.pack(fill="x", **pad)

        self.source_type = tk.StringVar(value="photo")

        radio_row = ttk.Frame(source_frame)
        radio_row.pack(fill="x", padx=8, pady=(6, 0))
        for label, val in [("Photo", "photo"), ("Video", "video"), ("USB Camera", "usb"), ("RTSP Feed", "rtsp")]:
            ttk.Radiobutton(radio_row, text=label, variable=self.source_type, value=val,
                            command=self._on_source_type_change).pack(side="left", padx=(0, 16))

        # Dynamic source input area
        self.source_input_frame = ttk.Frame(source_frame)
        self.source_input_frame.pack(fill="x", padx=8, pady=(4, 8))

        # File path (photo/video)
        self.file_path_var = tk.StringVar()
        self.file_row = ttk.Frame(self.source_input_frame)
        self.file_entry = ttk.Entry(self.file_row, textvariable=self.file_path_var, width=60)
        self.file_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.file_browse_btn = ttk.Button(self.file_row, text="Select File…", command=self._browse_source_file)
        self.file_browse_btn.pack(side="right")

        # USB dropdown
        self.usb_var = tk.StringVar()
        self.usb_row = ttk.Frame(self.source_input_frame)
        self.usb_devices = discover_video_devices()
        usb_values = self.usb_devices if self.usb_devices else ["(no devices found)"]
        self.usb_combo = ttk.Combobox(self.usb_row, textvariable=self.usb_var, values=usb_values, width=30, state="readonly")
        self.usb_combo.pack(side="left", padx=(0, 4))
        if self.usb_devices:
            self.usb_combo.current(0)
        ttk.Button(self.usb_row, text="Refresh", command=self._refresh_usb).pack(side="left")

        # RTSP entry
        self.rtsp_var = tk.StringVar(value="rtsp://")
        self.rtsp_row = ttk.Frame(self.source_input_frame)
        ttk.Label(self.rtsp_row, text="URL:").pack(side="left")
        ttk.Entry(self.rtsp_row, textvariable=self.rtsp_var, width=60).pack(side="left", fill="x", expand=True, padx=(4, 0))

        self._on_source_type_change()

        # --- Options ---
        opts_frame = ttk.LabelFrame(self.root, text="Options")
        opts_frame.pack(fill="x", **pad)

        opts_row1 = ttk.Frame(opts_frame)
        opts_row1.pack(fill="x", padx=8, pady=(6, 2))

        ttk.Label(opts_row1, text="Confidence:").pack(side="left")
        self.thresh_var = tk.DoubleVar(value=0.5)
        ttk.Spinbox(opts_row1, from_=0.1, to=1.0, increment=0.05, textvariable=self.thresh_var,
                     width=6).pack(side="left", padx=(4, 16))

        ttk.Label(opts_row1, text="OCR Workers:").pack(side="left")
        self.workers_var = tk.IntVar(value=2)
        ttk.Spinbox(opts_row1, from_=1, to=8, textvariable=self.workers_var,
                     width=4).pack(side="left", padx=(4, 16))

        opts_row2 = ttk.Frame(opts_frame)
        opts_row2.pack(fill="x", padx=8, pady=(2, 8))

        self.sr_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts_row2, text="Super Resolution", variable=self.sr_var).pack(side="left", padx=(0, 16))

        self.gpu_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts_row2, text="GPU", variable=self.gpu_var).pack(side="left", padx=(0, 16))

        self.enhance_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts_row2, text="Enhance Feed", variable=self.enhance_var).pack(side="left", padx=(0, 16))

        self.debug_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts_row2, text="OCR Debug", variable=self.debug_var).pack(side="left")

        # --- Controls ---
        ctrl_frame = ttk.Frame(self.root)
        ctrl_frame.pack(fill="x", **pad)

        self.start_btn = ttk.Button(ctrl_frame, text="Start Scanner", command=self._start)
        self.start_btn.pack(side="left", padx=(8, 4))

        self.stop_btn = ttk.Button(ctrl_frame, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=4)

        self.status_label = ttk.Label(ctrl_frame, text="Idle", foreground="gray")
        self.status_label.pack(side="left", padx=16)

        # --- Log area ---
        log_frame = ttk.LabelFrame(self.root, text="Output")
        log_frame.pack(fill="both", expand=True, **pad)

        self.log_text = tk.Text(log_frame, height=12, state="disabled", wrap="word",
                                bg="#1e1e1e", fg="#d4d4d4", font=("monospace", 10))
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)

    def _redirect_output(self):
        redirector = LogRedirector(self.log_text)
        sys.stdout = redirector
        sys.stderr = redirector

    # ── Source type switching ────────────────────────────────────────

    def _on_source_type_change(self):
        for widget in (self.file_row, self.usb_row, self.rtsp_row):
            widget.pack_forget()

        st = self.source_type.get()
        if st in ("photo", "video"):
            self.file_row.pack(fill="x")
        elif st == "usb":
            self.usb_row.pack(fill="x")
        elif st == "rtsp":
            self.rtsp_row.pack(fill="x")

    # ── Browse dialogs ──────────────────────────────────────────────

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select YOLO model",
            initialdir=os.path.join(PROJECT_ROOT, "models"),
            filetypes=[("PyTorch models", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.model_var.set(path)

    def _browse_source_file(self):
        st = self.source_type.get()
        if st == "photo":
            filetypes = [("Images", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")]
            title = "Select Photo"
        else:
            filetypes = [("Videos", "*.mp4 *.avi *.mkv *.mov *.wmv"), ("All files", "*.*")]
            title = "Select Video"
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if path:
            self.file_path_var.set(path)

    def _refresh_usb(self):
        self.usb_devices = discover_video_devices()
        values = self.usb_devices if self.usb_devices else ["(no devices found)"]
        self.usb_combo.configure(values=values)
        if self.usb_devices:
            self.usb_combo.current(0)

    # ── Start / Stop ────────────────────────────────────────────────

    def _get_source(self) -> str | None:
        st = self.source_type.get()
        if st in ("photo", "video"):
            path = self.file_path_var.get().strip()
            if not path or not os.path.isfile(path):
                print(f"Error: select a valid {st} file.")
                return None
            return path
        elif st == "usb":
            dev = self.usb_var.get()
            if not dev or dev.startswith("("):
                print("Error: no USB camera selected.")
                return None
            return dev
        elif st == "rtsp":
            url = self.rtsp_var.get().strip()
            if not url or url == "rtsp://":
                print("Error: enter an RTSP URL.")
                return None
            return url
        return None

    def _start(self):
        model = self.model_var.get().strip()
        if not model or not os.path.isfile(model):
            print("Error: select a valid YOLO model file.")
            return

        source = self._get_source()
        if source is None:
            return

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text="Starting…", foreground="orange")

        self.engine_thread = threading.Thread(target=self._run_engine, args=(model, source), daemon=True)
        self.engine_thread.start()

    def _run_engine(self, model: str, source: str):
        try:
            from parking_lot.config import ScannerConfig, CameraConfig, SRConfig

            cfg = ScannerConfig(
                model_path=model,
                sources=[source],
                min_thresh=self.thresh_var.get(),
                num_ocr_workers=max(1, self.workers_var.get()),
                ocr_debug=self.debug_var.get(),
                use_gpu=self.gpu_var.get(),
                camera=CameraConfig(feed_enhance=self.enhance_var.get()),
                sr=SRConfig(enabled=self.sr_var.get()),
            )

            from parking_lot.engine.scanner import ScannerEngine

            self.engine = ScannerEngine(cfg)
            self.engine.start()
            self.root.after(0, lambda: self.status_label.configure(text="Running", foreground="green"))

            # Run the blocking display loop (opens cv2 window)
            self.engine.run_display_loop()
        except Exception as e:
            print(f"Engine error: {e}")
        finally:
            self.engine = None
            self.root.after(0, self._on_engine_stopped)

    def _stop(self):
        if self.engine:
            print("Stopping scanner…")
            self.engine.stop()

    def _on_engine_stopped(self):
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="Stopped", foreground="gray")

    # ── Cleanup ─────────────────────────────────────────────────────

    def destroy(self):
        if self.engine:
            self.engine.stop()


def main():
    root = tk.Tk()
    app = ScannerGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.destroy(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
