# Main Program, run this

import warnings

# 24/10 2025 All working but a few duplicated procedures
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pynvml.*")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*CUDA capability.*")
warnings.filterwarnings("ignore", message=".*overflow encountered.*")
from router import CommandRouter
# Also set numpy to ignore specific warnings
import numpy as np
# ... your existing imports ...

from Avatars import CircleAvatarWindow, RectAvatarWindow, RectAvatarWindow2, RadialPulseAvatar, FaceRadialAvatar

# ... rest of your imports ...
np.seterr(over='ignore', under='ignore', invalid='ignore')
from dotenv import load_dotenv

load_dotenv()
from tkinter import ttk
import os, json, threading, time, re, math
from collections import deque
import soundfile as sf
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, messagebox
from io import BytesIO
from PIL import Image, ImageTk
import matplotlib
from status_light_window import StatusLightWindow

# Import the external LaTeX window
from latex_window import LatexWindow
# === Main App Class ===
from app_main import App



# This uses two models and a different Json file and handles images as well
# Look for the Json file  called Json2. You will need two models as per the Json file loaded into ollama
# can read equations off paper and handwriting. Uses qwen_llmSearch2.py
# Can use 'Sleep' and 'Awaken' to mute the microphone but text can still be used. There is also a switch to stop speech.
# So you can run entirely in text mode if you so choose.
# To search the internet you will need a key from Brave Search API https://brave.com/search/api/
# It's free for a fair number of searches but after that you need to pay
# Put that key in the file .env or the program won't work
# Try asking it "what is the latest news in New Zealand and it will find that enws and summarise it.
# Warning, Ai models make mistakes so check and answers
# Tom Moir Oct 2025

matplotlib.rcParams["figure.max_open_warning"] = 0
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tkinter.font as tkfont
from tkinter.scrolledtext import ScrolledText
import base64, tempfile, requests
import queue
import httpx
import trafilatura
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from urllib.parse import urljoin
import re
import winsound
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import cv2  # pip install opencv-python
except Exception:
    cv2 = None

# Optional drag & drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # pip install tkinterdnd2
except Exception:
    DND_FILES = None
    TkinterDnD = None

try:
    import torch

    print("[GPU torch]", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
except Exception:
    pass

try:
    import ctranslate2

    print("[CT2 CUDA types]", ctranslate2.get_supported_compute_types("cuda"))
except Exception:
    pass

# External modules you provide
from audio_io import list_input_devices, VADListener
from asr_whisper import ASR
from qwen_llmSearch2 import QwenLLM
from pydub import AudioSegment

...

# Import the math speech converter
from Speak_Maths import MathSpeechConverter

# Create a global instance
math_speech_converter = MathSpeechConverter()


# ===  ITEM DATACLASS ===
@dataclass
class Item:
    title: str
    url: str
    snippet: str = ""
    pubdate: Optional[str] = None
    summary: Optional[str] = None
    image_urls: List[str] = field(default_factory=list)


# === END DATACLASS ===


# ---------- Config ----------
def load_cfg():
    import os, json
    env_path = os.environ.get("APP_CONFIG")
    if env_path and os.path.exists(env_path):
        path = env_path
    else:
        base = os.path.dirname(os.path.abspath(__file__))
        c1 = os.path.join(base, "config.json")
        c2 = os.path.join(base, "config.example.json")
        c3 = "config.json" if os.path.exists("config.json") else None
        c4 = "config.example.json" if os.path.exists("config.example.json") else None
        path = next((p for p in (c1, c2, c3, c4) if p and os.path.exists(p)), None)

    if not path:
        raise FileNotFoundError("No config.json or config.example.json found")

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    print(f"[cfg] loaded: {os.path.abspath(path)}")
    sp = cfg.get("system_prompt", cfg.get("qwen_system_prompt", ""))
    print(f"[cfg] system_prompt present: {bool(sp)} (len={len(sp) if isinstance(sp, str) else 'n/a'})")
    return cfg


# ---------- TTS cleaner ----------
def clean_for_tts(text: str, speak_math: bool = True) -> str:
    """
    Enhanced TTS cleaner that converts LaTeX math to spoken English.

    Args:
        text: Input text containing LaTeX math
        speak_math: If True, convert math to speech; if False, say "equation"

    Returns:
        Text ready for TTS with math properly spoken
    """
    if not text:
        return ""

    # Use the math speech converter to handle LaTeX math
    cleaned_text = math_speech_converter.make_speakable_text(text, speak_math=speak_math)

    # Additional light cleanup for TTS
    cleaned_text = re.sub(r"[#*_`~>\[\]\(\)-]", "", cleaned_text)
    cleaned_text = re.sub(r":[a-z_]+:", "", cleaned_text)
    cleaned_text = re.sub(r"^[QAqa]:\s*", "", cleaned_text)
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)

    return cleaned_text.strip()

# Add another cleaning method for odd output from Deepseek
def clean_model_output(text: str) -> str:
    """
    Clean model-specific formatting artifacts from AI responses.
    """
    if not text:
        return ""

    cleaned = text

    print(f"🔧 [CLEANER] Input: {repr(text[:100])}")

    # Remove ALL variants of DeepSeek tokens:
    # <|im_end|>
    # <|im_end>|<think>
    # <|im_start|>
    # Any other <|...|> patterns

    # Method 1: Remove everything after any end token pattern
    end_patterns = [
        '<|im_end|>',
        '<|im_end>|<think>',
        '<|end|>',
        '<|endoftext|>'
    ]

    for pattern in end_patterns:
        if pattern in cleaned:
            parts = cleaned.split(pattern)
            cleaned = parts[0].strip()
            print(f"🔧 [CLEANER] Split by pattern: {pattern}")
            break  # Stop after first match

    # Method 2: Remove any remaining individual tokens
    tokens_to_remove = [
        '<|im_start|>', '<|im_end|>', '<|end|>', '<|endoftext|>',
        '<|im_end>|<think>', '<|think|>', '<|system|>', '<|user|>', '<|assistant|>'
    ]

    for token in tokens_to_remove:
        cleaned = cleaned.replace(token, '')

    # Method 3: Aggressive regex for any <|...|> or <|...> patterns
    import re
    cleaned = re.sub(r'<\|[^>]*(?:\|>|>)', '', cleaned)

    # Method 4: Remove LaTeX document wrappers
    if '\\documentclass' in cleaned:
        if '\\begin{document}' in cleaned:
            parts = cleaned.split('\\begin{document}')
            if len(parts) > 1:
                cleaned = parts[1].strip()
        if '\\end{document}' in cleaned:
            cleaned = cleaned.split('\\end{document}')[0].strip()

    # Final cleanup
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()

    print(f"🔧 [CLEANER] Output: {repr(cleaned[:100])}")
    print(f"🔧 [CLEANER] Success: {not any(token in cleaned for token in tokens_to_remove)}")

    return cleaned



def purge_temp_images(folder="out"):
    """
    Remove temporary/snapshot images we create. Keeps non-image files (e.g., last_reply.wav).
    """
    try:
        if not os.path.isdir(folder):
            return
        for name in os.listdir(folder):
            low = name.lower()
            # keep audio; delete our snapshots and temp frames
            if low.startswith("snapshot_") and low.endswith(".png"):
                try:
                    os.remove(os.path.join(folder, name))
                except Exception:
                    pass
            if low in ("live_frame.png", "tmp_frame.png"):
                try:
                    os.remove(os.path.join(folder, name))
                except Exception:
                    pass
    except Exception as e:
        print(f"[startup] purge_temp_images: {e}")


# === LaTeX Window ===

#wus here!
# End Latex Window
# === Echo Engine ===
class EchoEngine:
    """
    Ultra-light 'sci-fi echo': y[n] = dry*x[n] + wet*(x[n] + fb*y[n-D])
    Controls:
      - delay_ms (echo spacing)
      - intensity (maps to feedback & wet)
    """

    def __init__(self):
        self.enabled = False
        self.delay_ms = 144.0
        self.intensity = 0.47  # 0..1
        self.dry = 0.70

    def _coeffs(self):
        # Map intensity -> (feedback, wet) smoothly but safely stable
        fb = min(0.90, max(0.05, 0.15 + 0.8 * self.intensity))
        wet = min(0.95, max(0.05, 0.2 + 0.7 * self.intensity))
        return fb, wet

    def process_array(self, x, sr):
        if not self.enabled:
            return x.astype(np.float32)

        D = max(1, int(sr * self.delay_ms / 1000.0))
        fb, wet = self._coeffs()
        dry = float(self.dry)

        y = np.zeros_like(x, dtype=np.float32)
        # simple IIR echo; use integer delay for speed
        for n in range(len(x)):
            x_n = float(x[n])
            y_n = x_n
            if n - D >= 0:
                y_n += fb * y[n - D]
            y[n] = dry * x_n + wet * y_n

        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 1.0:
            y = y / peak
        return y.astype(np.float32)

    def process_file(self, in_wav, out_wav):
        print(f"[ECHO DEBUG] Processing file: {in_wav} -> {out_wav}")
        print(f"[ECHO DEBUG] Echo enabled: {self.enabled}, delay: {self.delay_ms}, intensity: {self.intensity}")
        x, sr = _read_wav_mono(in_wav)
        y = self.process_array(x, sr)
        _write_wav(out_wav, y, sr)
        print(f"[ECHO DEBUG] File processing complete")
        return out_wav, sr


class EchoWindow(tk.Toplevel):
    """Tiny control panel: Enable, Delay (ms), Intensity."""

    def __init__(self, master, engine: EchoEngine):
        super().__init__(master)
        self.engine = engine
        self.title("Sci-Fi Echo")
        self.geometry("360x180")
        self.protocol("WM_DELETE_WINDOW", self.withdraw)
        self.build_ui()
        self.withdraw()

    def _slider(self, parent, text, vmin, vmax, var, fmt="{:.1f}"):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text=text, width=16).pack(side="left")
        s = ttk.Scale(row, from_=vmin, to=vmax, orient="horizontal", variable=var)
        s.pack(side="left", fill="x", expand=True, padx=8)
        lab = ttk.Label(row, width=8)
        lab.pack(side="right")

        def update(*_):
            lab.config(text=fmt.format(var.get()))

        var.trace_add("write", lambda *_: update())
        update()
        return s

    def build_ui(self):
        e = self.engine
        wrap = ttk.Frame(self)
        wrap.pack(fill="both", expand=True, padx=8, pady=8)

        self.v_enabled = tk.BooleanVar(value=e.enabled)
        ttk.Checkbutton(wrap, text="Enable Echo", variable=self.v_enabled, command=self._apply).pack(anchor="w")

        self.v_delay = tk.DoubleVar(value=e.delay_ms)
        self._slider(wrap, "Delay (ms)", 60.0, 480.0, self.v_delay, "{:.0f}")

        self.v_inten = tk.DoubleVar(value=e.intensity)
        self._slider(wrap, "Intensity", 0.0, 1.0, self.v_inten, "{:.2f}")

        btns = ttk.Frame(wrap)
        btns.pack(fill="x", pady=(4, 0))
        ttk.Button(btns, text="Apply", command=self._apply).pack(side="left")
        ttk.Button(btns, text="Hide", command=self.withdraw).pack(side="right")

        for v in (self.v_delay, self.v_inten):
            v.trace_add("write", lambda *_: self._apply())

    def _apply(self):
        e = self.engine
        e.enabled = bool(self.v_enabled.get())
        e.delay_ms = float(self.v_delay.get())
        e.intensity = float(self.v_inten.get())


# === WEB SEARCH WINDOW CLASS ===
class WebSearchWindow(tk.Toplevel):
    def __init__(self, master, log_fn=None):
        super().__init__(master)
        self.title("Web Search")
        self.geometry("980x740")
        self.minsize(780, 620)
        self.protocol("WM_DELETE_WINDOW", self.withdraw)
        self._log = log_fn or (lambda msg: None)

        self.queue = queue.Queue()
        self.in_progress = False

        # Initialize tracking variables
        self.thumb_cache = {}
        self.image_bytes_cache = {}
        self.thumb_target = {}
        self.thumb_placeholder = {}
        self._all_results = []

        # LaTeX preview window for search results
        self.latex_win = None
        self.latex_auto = tk.BooleanVar(value=True)

        self._build_ui()
        self._build_strip_lights()
        self._tick_strip_lights()
        self._poll_queue()

    def _build_ui(self):
        # Main container
        self.main_container = ttk.Frame(self, padding=10)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=36, pady=36)

        # Query section
        ttk.Label(self.main_container, text="Query:").pack(anchor="w")
        self.txt_in = tk.Text(self.main_container, height=3, wrap="word")
        self.txt_in.pack(fill=tk.X, pady=(2, 8))

        # LaTeX controls for search results
        latex_controls = ttk.Frame(self.main_container)
        latex_controls.pack(fill=tk.X, pady=(0, 8))

        ttk.Checkbutton(
            latex_controls, text="Auto LaTeX preview for search results",
            variable=self.latex_auto
        ).pack(side="left", padx=(0, 10))

        ttk.Button(
            latex_controls, text="Show/Hide LaTeX",
            command=self.toggle_latex
        ).pack(side="left", padx=(0, 10))

        ttk.Button(
            latex_controls, text="Copy Raw LaTeX",
            command=self.copy_raw_latex
        ).pack(side="left")

        self.btn = ttk.Button(self.main_container, text="Search & Summarise", command=self.on_go)
        self.btn.pack(pady=(0, 8), anchor="w")

        # Output section - create a container for results that we can refresh
        self.results_container = ttk.Frame(self.main_container)
        self.results_container.pack(fill=tk.BOTH, expand=True)

        # Create initial results display
        self._create_results_display()

        # Status
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self.main_container, textvariable=self.status).pack(anchor="w", pady=(6, 0))

    def _create_results_display(self):
        """Create or recreate the results display area"""
        # Clear existing results display if it exists
        if hasattr(self, 'results_frame') and self.results_frame:
            try:
                self.results_frame.destroy()
            except:
                pass

        # Create new results frame
        self.results_frame = ttk.Frame(self.results_container)
        self.results_frame.pack(fill=tk.BOTH, expand=True)

        # Output text area
        ttk.Label(self.results_frame, text="Output:").pack(anchor="w")
        self.txt_out = tk.Text(self.results_frame, wrap="word", height=12)
        self.txt_out.pack(fill=tk.BOTH, expand=True)

        # Scrollbar for text output
        sbr = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.txt_out.yview)
        sbr.place(relx=1.0, rely=0, relheight=0.7, anchor="ne")
        self.txt_out.config(yscrollcommand=sbr.set)

        # Images container
        self.images_container = ttk.Frame(self.results_frame)
        self.images_container.pack(fill=tk.X, pady=8)

    def _ensure_latex_window(self):
        """Create LaTeX window if it doesn't exist"""
        if self.latex_win is None or not self.latex_win.winfo_exists():
            # Use the same settings as the main app's LaTeX window
            self.latex_win = LatexWindow(
                self.master,  # Use master (main app) as parent for consistency
                log_fn=self._log,
                text_family="Segoe UI",
                text_size=12,
                math_pt=8
            )

    def toggle_latex(self):
        """Toggle LaTeX window using main app's system"""
        try:
            if hasattr(self, 'main_app') and self.main_app:
                self.main_app.toggle_latex()
            elif hasattr(self, 'master') and hasattr(self.master, 'ensure_latex_window'):
                latex_win = self.master.ensure_latex_window("search")
                if latex_win.state() == "withdrawn":
                    latex_win.show()
                else:
                    latex_win.hide()
        except Exception as e:
            error_msg = f"[search] toggle latex error: {e}"
            if hasattr(self, 'logln'):
                self.logln(error_msg)
            else:
                print(error_msg)

    def copy_raw_latex(self):
        """Copy raw LaTeX source to clipboard with proper document structure"""
        try:
            content = self._last_text or ""

            if content.strip():
                # Create a complete LaTeX document
                latex_document = f"""\\documentclass{{article}}
    \\usepackage{{amsmath}}
    \\usepackage{{amssymb}}
    \\usepackage{{graphicx}}
    \\usepackage{{hyperref}}
    \\begin{{document}}

    {content}

    \\end{{document}}"""

                self.clipboard_clear()
                self.clipboard_append(latex_document)
                self._log("[latex] Complete LaTeX document copied to clipboard")
            else:
                self.clipboard_clear()
                self.clipboard_append("")
                self._log("[latex] No content to copy")

        except Exception as e:
            self._log(f"[latex] copy raw failed: {e}")

    def preview_latex(self, content: str, context="text"):
        """Preview LaTeX content with append/replace option"""
        if not self.latex_auto.get():
            return

        def _go():
            try:
                latex_win = self.ensure_latex_window(context)
                latex_win.show()

                # === CHECK APPEND MODE ===
                if self.latex_append_mode.get():
                    # APPEND MODE - add to existing content
                    latex_win.append_document(content)
                    self.logln(f"[latex] 📝 Appended to {context} window")
                else:
                    # REPLACE MODE - clear and show new content (original behavior)
                    latex_win.show_document(content)
                    self.logln(f"[latex] 🔄 Showing in {context} window (replace mode)")

                self._current_latex_context = context

            except Exception as e:
                self.logln(f"[latex] preview error ({context}): {e}")

        self.master.after(0, _go)

    def _build_strip_lights(self):
        self.strip_canvas = tk.Canvas(self, highlightthickness=0, bg="white")
        self.strip_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        tk.Misc.lower(self.strip_canvas)

        self.strip_items = []
        self.strip_colors = ["#ff3b30", "#ff9500", "#ffcc00", "#34c759", "#5ac8fa", "#007aff", "#af52de"]
        self.strip_step = 0

        margin = 10
        thickness = 40
        seg = 44

        def make_rect(x1, y1, x2, y2):
            rid = self.strip_canvas.create_rectangle(x1, y1, x2, y2, width=0, fill="white")
            self.strip_items.append(rid)

        def rebuild(_evt=None):
            self.strip_canvas.delete("all")
            self.strip_items.clear()
            W = max(1, self.winfo_width())
            H = max(1, self.winfo_height())
            # bottom
            x = margin
            y = H - margin - thickness
            while x + seg <= W - margin:
                make_rect(x, y, x + seg - 3, y + thickness)
                x += seg
            # left
            x = margin
            y = H - margin - thickness
            while y - seg >= margin:
                make_rect(x, y - seg + 3, x + thickness, y)
                y -= seg
            # top
            x = margin + thickness
            y = margin
            while x + seg <= W - margin - thickness:
                make_rect(x, y, x + seg - 3, y + thickness)
                x += seg
            # right
            x = W - margin - thickness
            y = margin + thickness
            while y + seg <= H - margin - thickness:
                make_rect(x, y, x + thickness, y + seg - 3)
                y += seg

        self.bind("<Configure>", rebuild)
        self.rebuild = rebuild
        rebuild()

    def _tick_strip_lights(self):
        if self.in_progress and self.strip_items:
            n = len(self.strip_items)
            self.strip_step = (self.strip_step + 1) % (n * len(self.strip_colors))
            for i, rid in enumerate(self.strip_items):
                idx = (self.strip_step + i) % len(self.strip_colors)
                self.strip_canvas.itemconfig(rid, fill=self.strip_colors[idx])
        else:
            for rid in self.strip_items:
                self.strip_canvas.itemconfig(rid, fill="white")
        self.after(90, self._tick_strip_lights)

    def show(self):
        self.deiconify()
        self.lift()

    def hide(self):
        """Hide the window (called by close button)"""
        try:
            self.withdraw()
        except Exception as e:
            print(f"Error hiding window: {e}")

    def log(self, msg: str):
        self.status.set(msg)
        self.update_idletasks()

    def on_go(self):
        if self.in_progress:
            self.log("Already running…")
            return

        q = self.txt_in.get("1.0", "end").strip()
        if not q:
            messagebox.showinfo("Info", "Please type a query.")
            return

        q = self.normalize_query(q)

        # DON'T recreate the results display - just clear the existing one
        self.txt_out.delete("1.0", "end")

        # Clear the images container
        for widget in self.images_container.winfo_children():
            widget.destroy()

        # Clear tracking variables (keep cache for performance)
        self.image_bytes_cache.clear()
        self.thumb_target.clear()
        self.thumb_placeholder.clear()
        self._all_results.clear()
        # Reset accumulated LaTeX content for new search
        self._accumulated_latex_content = ""

        self.in_progress = True
        self.btn.config(state="disabled")
        self.log("Searching…")

        # Start search in thread
        try:
            threading.Thread(target=self._thread_run, args=(q,), daemon=True).start()
        except Exception as e:
            self.log(f"Failed to start search thread: {e}")
            self.in_progress = False
            self.btn.config(state="normal")

    def _thread_run(self, query: str):
        # === SEARCH PROGRESS AUDIBLE INDICATOR ===
        if hasattr(self.master, 'start_search_progress_indicator'):
            self.master.start_search_progress_indicator()

        try:
            results = self.brave_search(query, 6)
            self.queue.put(("searched", len(results)))

            for idx, it in enumerate(results, 1):
                self.queue.put(("status", f"Processing result {idx}/{len(results)}..."))

                try:
                    html = self.polite_fetch(it.url)
                    if not html:
                        self.log(f"Failed to fetch: {it.url}")
                        self.queue.put(("article", {
                            "title": it.title, "url": it.url, "pubdate": None,
                            "summary": "Fetch failed - could not retrieve content.", "images": []
                        }))
                        continue

                    it.pubdate = self.guess_pubdate(html)
                    it.image_urls = self.extract_images(html, it.url)
                    text = self.extract_readable(html, it.url)

                    if len(text) < 400:
                        it.summary = "Not enough readable text to summarise."
                    else:
                        try:
                            it.summary = self.summarise_with_qwen(text, it.url, it.pubdate)
                            if not it.summary or "failed" in it.summary.lower():
                                it.summary = "Summarization unavailable for this content."
                        except Exception as e:
                            self.log(f"Summarization error: {e}")
                            it.summary = f"Summarization error: {str(e)}"

                    self.queue.put(("article", {
                        "title": it.title, "url": it.url, "pubdate": it.pubdate,
                        "summary": it.summary, "images": it.image_urls
                    }))
                    time.sleep(0.6)

                except Exception as e:
                    self.log(f"Error processing result {idx}: {e}")
                    continue

            self.queue.put(("done", None))
        except Exception as e:
            self.log(f"Search thread error: {e}")
            self.queue.put(("error", str(e)))
        # === STOP ON ERROR ===
        if hasattr(self.master, 'stop_search_progress_indicator'):
            self.master.stop_search_progress_indicator()

    def _create_final_summary(self) -> str:
        """Create a consolidated summary of all search results"""
        if not self._all_results:
            return "No results to summarize."

        # Use the ACCUMULATED content that already has all the equations
        if hasattr(self, '_accumulated_latex_content') and self._accumulated_latex_content:
            final_text = "COMPLETE SEARCH RESULTS WITH EQUATIONS:\n\n" + self._accumulated_latex_content
        else:
            # Fallback to the original summary method
            summary_parts = ["Search Results Summary:"]
            for i, result in enumerate(self._all_results, 1):
                title = result.get('title', 'No title')
                summary = result.get('summary', 'No summary available')

                # Clean up the summary text but preserve equations
                clean_summary = re.sub(r'=+', '', summary)
                clean_summary = re.sub(r'\s+', ' ', clean_summary).strip()
                clean_summary = re.sub(r'[\*#_-]{2,}', '', clean_summary)
                clean_summary = re.sub(r'^(summary|result|findings?):?\s*', '', clean_summary, flags=re.IGNORECASE)

                if len(clean_summary) > 250:
                    clean_summary = clean_summary[:247] + "..."

                summary_parts.append(f"Result {i}: {title}")
                if clean_summary and clean_summary != "No summary available":
                    summary_parts.append(f"{clean_summary}")
                summary_parts.append("")

            final_text = "\n".join(summary_parts)
            final_text = re.sub(r'\n{3,}', '\n\n', final_text)

        # PREVIEW LATEX FOR THE FINAL ACCUMULATED CONTENT
        self.preview_latex(final_text)

        return final_text.strip()

    def _poll_queue(self):
        try:
            while True:
                msg, payload = self.queue.get_nowait()
                try:
                    if msg == "status":
                        self.log(payload)
                    elif msg == "searched":
                        n = payload
                        self.log(f"Found {n} results. Summarising…")
                        self.play_ding()

                    elif msg == "article":
                        if self.in_progress:  # Only process if we're still searching
                            data = payload
                            block = (
                                    f"Title: {data['title']}\n"
                                    f"URL: {data['url']}\n"
                                    + (f"Publish date: {data['pubdate']}\n" if data['pubdate'] else "")
                                    + f"\nSummary:\n{data['summary']}"
                            )
                            self.append_output(block)
                            self._show_images(data.get("images", []), data['title'], data['url'])
                            self._all_results.append(data)

                    elif msg == "done":
                        self.log("Done.")
                        self.in_progress = False
                        self.btn.config(state="normal")
                        self.play_dong()
                        # === STOP PROGRESS INDICATOR ===
                        if hasattr(self.master, 'stop_search_progress_indicator'):
                            self.master.stop_search_progress_indicator()

                        if self._all_results:
                            summary_text = self._create_final_summary()
                            if hasattr(self, 'synthesize_search_results'):
                                self.synthesize_search_results(summary_text)

                    elif msg == "imgbytes":
                        url, data = payload
                        self.image_bytes_cache[url] = data
                        target = self.thumb_target.pop(url, None)
                        if target and self.winfo_exists():
                            self._create_thumb_from_bytes(url, data, target_frame=target)

                    elif msg == "imgfail":
                        url = payload
                        ph = self.thumb_placeholder.pop(url, None)
                        if ph and ph.winfo_exists():
                            ph.config(text="Image blocked or unavailable")

                    elif msg == "error":
                        self.in_progress = False
                        self.btn.config(state="normal")
                        self.log(f"Error: {payload}")
                    # ===  STOP ON ERROR ===
                    if hasattr(self.master, 'stop_search_progress_indicator'):
                        self.master.stop_search_progress_indicator()

                except Exception as e:
                    self.log(f"Error processing queue message {msg}: {e}")

        except queue.Empty:
            pass
        except Exception as e:
            self.log(f"Queue polling error: {e}")
            self.in_progress = False
            self.btn.config(state="normal")
        # ===  STOP ON GENERAL ERROR ===
        if hasattr(self.master, 'stop_search_progress_indicator'):
            self.master.stop_search_progress_indicator()

        if self.winfo_exists():
            self.after(120, self._poll_queue)

    def append_output(self, block: str):
        self.txt_out.insert("end", block + "\n\n")
        self.txt_out.see("end")

        # ACCUMULATE all results for LaTeX preview instead of overwriting
        if not hasattr(self, '_accumulated_latex_content'):
            self._accumulated_latex_content = ""

        # Add this result to the accumulated content
        self._accumulated_latex_content += block + "\n\n"

        # Preview the ACCUMULATED content (all results so far)
        self.preview_latex(self._accumulated_latex_content)

    def _show_images(self, urls: List[str], article_title: str, article_url: str):
        try:
            section = ttk.Frame(self.images_container)
            section.pack(fill=tk.X, pady=6)

            hdr = ttk.Label(section, text=f"Images: {article_title}", cursor="hand2")
            hdr.pack(anchor="w")
            hdr.bind("<Button-1>", lambda e, u=article_url: webbrowser.open(u))

            if not urls:
                ttk.Label(section, text="(No images found.)").pack(anchor="w")
                return

            for i, u in enumerate(urls[:3]):
                try:
                    if u in self.thumb_cache:
                        self._place_thumb(u, self.thumb_cache[u], target_frame=section)
                    elif u in self.image_bytes_cache:
                        self._create_thumb_from_bytes(u, self.image_bytes_cache[u], target_frame=section)
                    else:
                        ph = ttk.Label(section, text=f"Loading image {i + 1}…")
                        ph.pack(side=tk.LEFT, padx=6)
                        self.thumb_target[u] = section
                        self.thumb_placeholder[u] = ph
                        threading.Thread(target=self._thread_fetch_image, args=(u,), daemon=True).start()
                except Exception as e:
                    self.log(f"Error displaying image {i}: {e}")
                    continue
        except Exception as e:
            self.log(f"Error in _show_images: {e}")

    def _thread_fetch_image(self, url: str):
        try:
            headers = {"User-Agent": "LocalAI-ResearchBot/1.0", "Referer": url}
            with httpx.Client(timeout=15.0, headers=headers, follow_redirects=True) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.content
            self.queue.put(("imgbytes", (url, data)))
        except Exception:
            self.queue.put(("imgfail", url))

    def _create_thumb_from_bytes(self, url: str, data: bytes, target_frame=None):
        try:
            img = Image.open(BytesIO(data))
            img.thumbnail((220, 140))
            tkimg = ImageTk.PhotoImage(img)
            self.thumb_cache[url] = tkimg
            ph = self.thumb_placeholder.pop(url, None)
            if ph and ph.winfo_exists():
                ph.destroy()
            self._place_thumb(url, tkimg, target_frame=target_frame)
        except Exception:
            self.queue.put(("imgfail", url))

    def _place_thumb(self, url: str, tkimg: ImageTk.PhotoImage, target_frame=None):
        parent = target_frame if target_frame is not None else self.images_container
        lbl = ttk.Label(parent, image=tkimg)
        lbl.image = tkimg
        lbl.pack(side=tk.LEFT, padx=6, pady=4)
        lbl.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))

    def play_ding(self):
        """Play search start sound"""
        try:
            fs = 16000
            duration = 0.17
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)
            freq = 988
            beep = 0.3 * np.sin(2 * np.pi * freq * t)
            fade_samples = int(0.01 * fs)
            beep[:fade_samples] *= np.linspace(0, 1, fade_samples)
            beep[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            sd.play(beep, fs, blocking=False)
        except Exception as e:
            print(f"[ding] error: {e}")

    def play_dong(self):
        """Play search complete sound"""
        try:
            fs = 16000
            duration = 0.26
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)
            freq = 659
            beep = 0.3 * np.sin(2 * np.pi * freq * t)
            fade_samples = int(0.01 * fs)
            beep[:fade_samples] *= np.linspace(0, 1, fade_samples)
            beep[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            sd.play(beep, fs, blocking=False)
        except Exception as e:
            print(f"[dong] error: {e}")

    def normalize_query(self, q: str) -> str:
        """Add date context to time-related queries"""
        ql = q.lower()
        now = datetime.now()
        if "today" in ql:
            q += " " + now.strftime("%Y-%m-%d")
        if "yesterday" in ql:
            q += " " + (now - timedelta(days=1)).strftime("%Y-%m-%d")
        if "this week" in ql:
            q += " " + now.strftime("week %G-W%V")
        return q

    def destroy(self):
        """Proper cleanup when window is closed"""
        self.in_progress = False
        # Close LaTeX window if open
        if self.latex_win and self.latex_win.winfo_exists():
            try:
                self.latex_win.destroy()
            except:
                pass
        super().destroy()


# === END WEB SEARCH WINDOW ===
#

# === Main App Class ===


# === END SEARCH METHODS ===
# End of App
# Helper functions for EchoEngine
def _read_wav_mono(path):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32), sr


def _write_wav(path, y, sr):
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y.astype(np.float32), sr)


# -------- Run --------
if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.25)
    except Exception:
        pass
    try:
        style = ttk.Style()
        style.theme_use("clam")
    except Exception:
        pass
    app = App(root)
    root.mainloop()
