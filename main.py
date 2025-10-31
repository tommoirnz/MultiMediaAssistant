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

from web_search_window import WebSearchWindow


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
