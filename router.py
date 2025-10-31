# command_router.py
import re
import time
import webbrowser
from typing import Callable, Dict, Any
import tkinter.messagebox as messagebox


class CommandRouter:
    """
    Handles voice and text command routing for the AI assistant.
    Separated from main App class to reduce complexity.
    """

    def __init__(self, app_instance=None):
        self.app = app_instance
        self.sleep_mode = False

        # Command categories
        self.sleep_commands = ["sleep", "go to sleep", "rest mode", "silence", "stop listening"]
        self.wake_commands = ["wake", "wake up", "awaken", "resume", "start listening", "listen again"]

        # Personality commands mapping
        self.personality_commands = {
            "be the butler": "Jeeves the Butler",
            "butler mode": "Jeeves the Butler",
            "butler personality": "Jeeves the Butler",
            "be the teacher": "Mrs. Hardcastle",
            "teacher mode": "Mrs. Hardcastle",
            "school teacher": "Mrs. Hardcastle",
            "strict teacher": "Mrs. Hardcastle",
            "be the scientist": "Dr. Von Knowledge",
            "mad scientist": "Dr. Von Knowledge",
            "scientist mode": "Dr. Von Knowledge",
            "dr von knowledge": "Dr. Von Knowledge",
            "back to normal": "Default",
            "be normal": "Default",
            "normal mode": "Default",
            "be the explorer": "Space Explorer",
            "space explorer mode": "Space Explorer",
            "space explorer personality": "Space Explorer",
            "captain nova mode": "Space Explorer",
            "om shanti": "BK Yogi",
            "aum shanti": "BK Yogi"
        }

        # Mute commands
        self.mute_commands = {
            "mute text": "mute_text",
            "mute text ai": "mute_text",
            "silence text": "mute_text",
            "text ai mute": "mute_text",
            "text mute": "mute_text",
            "mute vision": "mute_vision",
            "mute vision ai": "mute_vision",
            "silence vision": "mute_vision",
            "vision ai mute": "mute_vision",
            "vision mute": "mute_vision",
            "unmute text": "unmute_text",
            "unmute text ai": "unmute_text",
            "enable text audio": "unmute_text",
            "text audio on": "unmute_text",
            "unmute vision": "unmute_vision",
            "unmute vision ai": "unmute_vision",
            "enable vision audio": "unmute_vision",
            "vision audio on": "unmute_vision",
            "toggle text mute": "toggle_text_mute",
            "toggle vision mute": "toggle_vision_mute",
            "text mute toggle": "toggle_text_mute",
            "vision mute toggle": "toggle_vision_mute"
        }

        # Search commands
        self.search_commands = [
            "search for", "search", "look up", "information on", "find information about",
            "web search", "search the web for", "look for", "that's for", "database", "online"
        ]

        # Other command categories
        self.exit_vision_commands = [
            "new topic", "switch to text", "text mode", "forget image", "clear image", "ignore the image",
            "can I speak to Zen", "stop using the image", "no vision", "text only", "go back to text",
            "can I speak to zen", "back to text", "speak with zen", "speak to zen"
        ]

        self.send_to_text_commands = [
            "send", "send to text", "send to zen", "send information", "send that",
            "ok send that", "pass to text", "pass to zen", "transfer to text", "send to test",
            "give to zen", "forward to text", "share with zen", "send that to text",
            "send it to text", "pass it to test", "send this to text", "pass this to text", "ok push"
        ]

        self.stop_commands = [
            "stop", "stop speaking", "stop talking", "be quiet", "shut up",
            "enough", "that's enough", "okay stop", "ok stop", "fuck off"
        ]

        self.close_windows_commands = [
            "close window", "close all windows", "hide windows", "minimize windows",
            "clean up windows", "tidy windows", "close extra windows", "clean desktop"
        ]

        self.debug_commands = [
            "debug vision", "vision status", "vision debug", "show vision state"
        ]

        self.test_vision_commands = [
            "test vision", "vision test", "check vision"
        ]

        self.what_see_commands = [
            "what do you see", "what can you see", "whats happening", "what is happening",
            "whats going on", "what is going on", "describe what you see", "tell me what you see",
            "show me what you see", "what are you seeing", "whats in front of you",
            "describe the scene", "whats around you", "whats in the room"
        ]

        self.camera_commands = {
            "start camera": "start_camera",
            "open camera": "start_camera",
            "turn on camera": "start_camera",
            "camera on": "start_camera",
            "start the camera": "start_camera",
            "enable camera": "start_camera",
            "stop camera": "stop_camera",
            "close camera": "stop_camera",
            "turn off camera": "stop_camera",
            "camera off": "stop_camera",
            "stop the camera": "stop_camera",
            "disable camera": "stop_camera",
            "take a picture": "take_picture",
            "take picture": "take_picture",
            "take a photo": "take_picture",
            "take photo": "take_picture",
            "snapshot": "take_picture",
            "capture photo": "take_picture",
            "capture image": "take_picture",
            "snap a picture": "take_picture"
        }

        self.vision_takeover_commands = [
            "vision mode", "image mode",
            "speak with the image ai", "speak to the image ai", "talk to the image ai",
            "speak with the vision ai", "speak to the vision ai", "talk to the vision ai",
            "use the image ai", "use the vision ai",
            "switch to vision", "switch to image", "speak with the Guardian"
        ]

    def set_app_instance(self, app_instance):
        """Set the main app instance for callbacks"""
        self.app = app_instance

    def filter_whisper_hallucinations(self, text: str) -> str:
        """Filter out common Whisper hallucinations"""
        if not text or not text.strip():
            return ""

        hallucinations = [
            "thanks for watching", "I'm so sorry", "I am so sorry",
            "don't forget to subscribe", "hit the bell icon",
            "see you next time", "Thank you for watching!", "bye everyone",
            "Thank you for watching and see you next time.", "in this video", "before we start"
        ]

        text_lower = text.lower().strip()
        if self.app:
            self.app.logln(f"[asr-filter] Checking: '{text}'")

        # Remove common punctuation for more flexible matching
        text_clean = text_lower.replace('!', '').replace('?', '').replace('.', '').replace(',', '')
        text_clean = text_clean.strip()

        # Check if the clean text matches any hallucination phrases
        for hallucination in hallucinations:
            if text_clean == hallucination:
                if self.app:
                    self.app.logln(f"[asr-filter] ❌ EXACT MATCH FILTERED: '{text}'")
                return ""

        # Also check if it starts with common hallucination phrases
        for hallucination in hallucinations:
            if text_lower.startswith(hallucination):
                if self.app:
                    self.app.logln(f"[asr-filter] ❌ STARTS WITH FILTERED: '{text}'")
                return ""

        if self.app:
            self.app.logln(f"[asr-filter] ✅ PASSED: '{text}'")
        return text

    def normalize_text(self, text: str) -> str:
        """Normalize text for command matching"""
        if not text:
            return ""

        # Light normalization
        norm_map = {
            "what's": "what is",
            "whats": "what is",
            "i'm": "i am",
            "you're": "you are",
            "it's": "it is",
            "that's": "that is",
        }

        normalized = text.lower()
        for k, v in norm_map.items():
            normalized = normalized.replace(k, v)

        # remove punctuation except spaces/alphanumerics
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s{2,}", " ", normalized).strip()

        return normalized

    def matched(self, text: str, patterns: list) -> bool:
        """Check if text matches any pattern in list"""
        return any(p in text for p in patterns)

    def route_command(self, raw_text: str) -> bool:
        """
        Handle voice/typed control phrases robustly.
        Returns True if a command was executed.
        """
        if not self.app:
            return False

        # Filter hallucinations first
        raw_text = self.filter_whisper_hallucinations(raw_text)
        if not raw_text:
            return False

        text = self.normalize_text(raw_text)

        # Check sleep/wake commands FIRST
        if any(cmd in text for cmd in self.sleep_commands) and not self.sleep_mode:
            self.enter_sleep_mode()
            return True

        if any(cmd in text for cmd in self.wake_commands) and self.sleep_mode:
            self.exit_sleep_mode()
            return True

        # If in sleep mode, ignore ALL other voice commands
        if self.sleep_mode:
            self.app.logln("[sleep] 💤 Ignoring voice input while sleeping")
            self.play_sleep_reminder_beep()
            return True

        # Dispatch order: Most specific commands first
        return (
                self._handle_exit_vision(text) or
                self._handle_send_to_text(text) or
                self._handle_stop_commands(text) or
                self._handle_camera_commands(text) or
                self._handle_what_see_commands(text) or
                self._handle_personality_commands(text) or
                self._handle_mute_commands(text) or
                self._handle_search_commands(text, raw_text) or
                self._handle_vision_takeover(text) or
                self._handle_test_vision(text) or
                self._handle_debug_commands(text) or
                self._handle_close_windows(text)
        )

    def _handle_exit_vision(self, text: str) -> bool:
        """Handle exit vision context commands"""
        if self.matched(text, self.exit_vision_commands):
            self.app.logln(f"[command] Exit vision command detected: '{text}'")
            self.app._last_was_vision = False
            self.app._vision_context_until = 0.0
            self.app._vision_turns_left = 0
            try:
                if hasattr(self.app, 'latex_win'):
                    self.app.latex_win.set_scheme("default")
            except Exception:
                pass
            self.app.logln("[vision] image context cleared; back to text mode")
            return True
        return False

    def _handle_send_to_text(self, text: str) -> bool:
        """Handle send to text AI commands"""
        if self.matched(text, self.send_to_text_commands):
            self.app.logln(f"[command] Send to text command detected: '{text}'")
            self.app._pass_vision_to_text_voice()
            return True
        return False

    def _handle_stop_commands(self, text: str) -> bool:
        """Handle stop speaking commands"""
        if self.matched(text, self.stop_commands):
            self.app.logln("[command] Stop command detected - stopping speech")
            self.app.stop_speaking()
            self.app.set_light("idle")
            return True
        return False

    def _handle_camera_commands(self, text: str) -> bool:
        """Handle camera control commands"""
        for command, method_name in self.camera_commands.items():
            if command in text:
                method = getattr(self.app, f"{method_name}_ui", None)
                if method:
                    method()
                    self.app.set_light("idle")
                    return True
        return False

    def _handle_what_see_commands(self, text: str) -> bool:
        """Handle 'what do you see' commands"""
        if self.matched(text, self.what_see_commands):
            self.app.what_do_you_see_ui()
            self.app.set_light("idle")
            return True
        return False

    def _handle_personality_commands(self, text: str) -> bool:
        """Handle personality switching commands"""
        for command, personality in self.personality_commands.items():
            if command in text:
                if hasattr(self.app, 'personalities') and personality in self.app.personalities:
                    current_personality = self.app.personality_var.get()
                    if current_personality == personality:
                        self.app.speak_search_status(f"Already in {personality} mode")
                    else:
                        self.app.personality_var.set(personality)
                        self.app.apply_personality(personality)
                        if personality == "Default":
                            self.app.speak_search_status("Returning to normal mode")
                        else:
                            self.app.speak_search_status(f"Activating {personality} personality")
                    return True
        return False

    def _handle_mute_commands(self, text: str) -> bool:
        """Handle mute/unmute commands"""
        for command, method_name in self.mute_commands.items():
            if command in text:
                method = getattr(self.app, method_name, None)
                if method:
                    method()
                    return True
        return False

    def _handle_search_commands(self, text: str, raw_text: str) -> bool:
        """Handle search commands with query extraction"""
        for cmd in self.search_commands:
            if cmd in text:
                # Extract the actual query by removing the command phrase
                query = text.replace(cmd, "").strip()

                # Handle cases like "search for cats" vs "search cats"
                if not query:  # If command was at the end, try alternative parsing
                    words = text.split()
                    cmd_words = cmd.split()
                    if len(words) > len(cmd_words):
                        query = " ".join(words[len(cmd_words):])

                # Add date enhancement for flight queries
                if any(flight_word in query.lower() for flight_word in ['flight', 'fly', 'airline', 'airfare']):
                    from datetime import datetime, timedelta
                    today = datetime.now()

                    # Handle "next monday" type queries
                    if "next monday" in query.lower():
                        days_ahead = 7 - today.weekday()  # Monday is 0
                        if days_ahead <= 0:  # If today is Monday or after
                            days_ahead += 7
                        next_monday = today + timedelta(days=days_ahead)
                        query += f" {next_monday.strftime('%Y-%m-%d')}"
                        self.app.logln(f"[search] Enhanced flight query with date: {next_monday.strftime('%Y-%m-%d')}")

                    # Add current year if not present
                    current_year = today.year
                    if str(current_year) not in query and any(word in query.lower() for word in
                                                              ['january', 'february', 'march', 'april', 'may', 'june',
                                                               'july', 'august', 'september', 'october', 'november',
                                                               'december']):
                        query += f" {current_year}"
                        self.app.logln(f"[search] Enhanced flight query with year: {current_year}")

                if query:
                    self.app.logln(f"[search] Voice search: {query}")
                    self.app.speak_search_status(f"Searching for {query}")
                    # Ensure search window exists and is visible
                    self.app.toggle_search_window(ensure_visible=True)

                    # Wait a moment for window to appear, then set query and search
                    def do_voice_search():
                        try:
                            if (self.app.search_win and
                                    self.app.search_win.winfo_exists() and
                                    not self.app.search_win.in_progress):

                                # Clear any previous query and set new one
                                self.app.search_win.txt_in.delete("1.0", "end")
                                self.app.search_win.txt_in.insert("1.0", query)

                                # Start the search
                                self.app.search_win.on_go()
                            else:
                                self.app.logln("[search] Search window not ready for voice command")
                        except Exception as e:
                            self.app.logln(f"[search] Voice search error: {e}")

                    # Give the window time to appear before starting search
                    if hasattr(self.app, 'master'):
                        self.app.master.after(500, do_voice_search)
                    return True
        return False

    def _handle_vision_takeover(self, text: str) -> bool:
        """Handle vision mode takeover commands"""
        if self.matched(text, self.vision_takeover_commands):
            self.app._ensure_image_window()
            try:
                if self.app._img_win and self.app._img_win.winfo_exists():
                    self.app._img_win.deiconify()
                    self.app._img_win.lift()
                    has_image = bool(getattr(self.app._img_win, "_img_path", None))
                    if not has_image:
                        self.app._img_win.start_camera()
            except Exception:
                pass

            self.app._last_was_vision = True
            self.app._vision_turns_left = 5  # Default turns
            self.app._vision_context_until = time.monotonic() + 300  # 5 minutes

            try:
                if hasattr(self.app, 'latex_win'):
                    self.app.latex_win.set_scheme("vision")
            except Exception:
                pass

            self.app.logln(f"[vision] takeover requested — turns={self.app._vision_turns_left}")
            self.app.set_light("idle")
            return True
        return False

    def _handle_test_vision(self, text: str) -> bool:
        """Handle vision test commands"""
        if self.matched(text, self.test_vision_commands):
            self.app.logln("[vision-test] Running vision system test...")
            if hasattr(self.app, 'debug_vision_state'):
                self.app.debug_vision_state()
            # Test if we have a current image
            if hasattr(self.app, '_last_image_path') and self.app._last_image_path and os.path.exists(
                    self.app._last_image_path):
                self.app.logln(f"[vision-test] Current image: {os.path.basename(self.app._last_image_path)}")
                self.app.ask_vision(self.app._last_image_path, "Describe this image briefly for testing.")
            else:
                self.app.logln("[vision-test] No current image available")
            return True
        return False

    def _handle_debug_commands(self, text: str) -> bool:
        """Handle debug commands"""
        if self.matched(text, self.debug_commands):
            if hasattr(self.app, 'debug_vision_state'):
                self.app.debug_vision_state()
            self.app.set_light("idle")
            return True
        return False

    def _handle_close_windows(self, text: str) -> bool:
        """Handle close windows commands"""
        if self.matched(text, self.close_windows_commands):
            self.app.logln(f"[command] Close windows command detected: '{text}'")
            if hasattr(self.app, 'close_all_windows'):
                self.app.close_all_windows()
            return True
        return False

    def enter_sleep_mode(self):
        """Enter sleep mode"""
        if not self.sleep_mode:
            self.sleep_mode = True
            if self.app:
                self.app.set_light("idle")
                self.app.logln("[sleep] 💤 Sleep mode activated - ignoring voice input")
                self.play_sleep_chime()
                if hasattr(self.app, 'close_all_windows'):
                    self.app.close_all_windows()
                try:
                    self.app.master.title("Always Listening — Qwen (SLEEPING)")
                except:
                    pass

    def exit_sleep_mode(self):
        """Exit sleep mode"""
        if self.sleep_mode:
            self.sleep_mode = False
            if self.app:
                self.app.set_light("listening")
                self.app.logln("[sleep] 🔔 Awake mode activated - listening for voice")
                self.play_wake_chime()
                try:
                    self.app.master.title("Always Listening — Qwen (local)")
                except:
                    pass

    def play_sleep_chime(self):
        """Play sleep confirmation chime"""
        try:
            import numpy as np
            import sounddevice as sd

            fs = 16000
            duration = 0.3
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)
            # Descending tone for sleep
            freq = np.linspace(660, 220, len(t))
            beep = 0.2 * np.sin(2 * np.pi * freq * t)
            fade = int(0.02 * fs)
            beep[:fade] *= np.linspace(0, 1, fade)
            beep[-fade:] *= np.linspace(1, 0, fade)
            sd.play(beep, fs, blocking=False)
        except Exception as e:
            if self.app:
                self.app.logln(f"[sleep-chime] {e}")

    def play_wake_chime(self):
        """Play wake confirmation chime"""
        try:
            import numpy as np
            import sounddevice as sd

            fs = 16000
            duration = 0.25
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)
            # Ascending tone for wake
            freq = np.linspace(220, 660, len(t))
            beep = 0.2 * np.sin(2 * np.pi * freq * t)
            fade = int(0.01 * fs)
            beep[:fade] *= np.linspace(0, 1, fade)
            beep[-fade:] *= np.linspace(1, 0, fade)
            sd.play(beep, fs, blocking=False)
        except Exception as e:
            if self.app:
                self.app.logln(f"[wake-chime] {e}")

    def play_sleep_reminder_beep(self):
        """Play noticeable beep to indicate sleeping mode"""
        try:
            import numpy as np
            import sounddevice as sd

            fs = 16000
            duration = 0.25
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)

            # More noticeable tone
            freq1 = 440
            freq2 = 330

            # Create a two-tone beep
            beep1 = 0.3 * np.sin(2 * np.pi * freq1 * t[:len(t) // 2])
            beep2 = 0.3 * np.sin(2 * np.pi * freq2 * t[len(t) // 2:])
            beep = np.concatenate([beep1, beep2])

            # Smooth fade in/out
            fade = int(0.02 * fs)
            beep[:fade] *= np.linspace(0, 1, fade)
            beep[-fade:] *= np.linspace(1, 0, fade)

            sd.play(beep, fs, blocking=False)
            if self.app:
                self.app.logln("[sleep] 💤 (sleep reminder beep)")

        except Exception as e:
            if self.app:
                self.app.logln(f"[sleep-reminder] error: {e}")
