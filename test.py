import os
os.environ["OMP_NUM_THREADS"] = "1"

import tkinter as tk
import socket
import threading
import joblib
import time
import pygame

from pathlib import Path
from collections import deque


# ============================================================
# CONFIG
# ============================================================

PORT = 5005
TIMEOUT = 2.5
SMOOTH = 7


# ============================================================
# APP
# ============================================================

class SignApp:

    def __init__(self, root):

        self.root = root
        self.root.title("🖐️ Sign Language AI System")
        self.root.geometry("1200x750")
        self.root.configure(bg="white")

        # ========================================================
        # AUDIO INIT
        # ========================================================

        pygame.mixer.init()

        # ========================================================
        # MODELS
        # ========================================================

        # LEFT MODEL (11 features)
        self.left_model = joblib.load("CombinedGesture_model.pkl")
        self.left_scaler = joblib.load("Modelscaler.pkl")
        self.left_encoder = joblib.load("label_encoder.pkl")

        # ARABIC MODEL (22 features)
        self.ar_model = joblib.load("ArabicGloves_XGBoost_Model.pkl")
        self.ar_scaler = joblib.load("ArabicGloves_Scaler.pkl")
        self.ar_encoder = joblib.load("ArabicGloves_LabelEncoder.pkl")

        try:
            self.ar_pca = joblib.load("ArabicGloves_PCA.pkl")
        except:
            self.ar_pca = None

        print("✅ Models loaded")

        # ========================================================
        # AUDIO PATHS
        # ========================================================

        BASE_DIR = Path(".")

        self.LETTERS_CHOSEN_DIR = (
            BASE_DIR /
            "generated_audio" /
            "edge_tts_mapping_v2" /
            "chosen"
        )

        self.NUMBERS_CHOSEN_DIR = (
            BASE_DIR /
            "generated_audio" /
            "edge_tts_numbers_mapping_v1" /
            "chosen"
        )

        # ========================================================
        # AUDIO MAPS
        # ========================================================

        self.setup_audio_map()

        # ========================================================
        # PREVENT REPEATING AUDIO
        # ========================================================

        self.last_spoken = ""
        self.last_speak_time = 0

        # ========================================================
        # DATA
        # ========================================================

        self.left = [0] * 11
        self.right = [0] * 11

        self.lt = 0
        self.rt = 0

        self.lock = threading.Lock()
        self.buffer = deque(maxlen=SMOOTH)

        # ========================================================
        # UI
        # ========================================================

        self.setup_ui()

        # ========================================================
        # UDP
        # ========================================================

        self.start_udp()

        # ========================================================
        # LOOP
        # ========================================================

        threading.Thread(target=self.loop, daemon=True).start()

        self.show_page("left")

    # ============================================================
    # AUDIO MAP
    # ============================================================

    def setup_audio_map(self):

        self.LETTER_AUDIO_MAP = {

            "ا": "letter_ا_alif.mp3",
            "أ": "letter_ا_alif.mp3",

            "ب": "letter_ب_cut_0.45.mp3",
            "ت": "letter_ت_teh.mp3",
            "ث": "comp_seh_ث_ث_ه.mp3",
            "ج": "letter_ج_geem.mp3",
            "ح": "letter_ح_ha.mp3",
            "خ": "letter_خ_kha.mp3",
            "د": "comp_dal_د_دال_bang.mp3",
            "ذ": "comp_zal_ذ_ذال_bang.mp3",
            "ر": "letter_ر_ra.mp3",
            "ز": "comp_zay_ز_zeen_latin_long.mp3",
            "س": "comp_seen_س_seen_split_cut.mp3",
            "ش": "letter_ش_sheen.mp3",
            "ص": "comp_sad_ص_saad_long_sukoon.mp3",
            "ض": "comp_dad_ض_daad_long_sukoon.mp3",
            "ط": "comp_ta_ط_ط_ه.mp3",
            "ظ": "letter_ظ_dha.mp3",
            "ع": "letter_ع_ein.mp3",
            "غ": "letter_غ_ghein.mp3",
            "ف": "comp_faa_ف_ف_ه.mp3",
            "ق": "comp_qaf_ق_qaf_split_cut_v2.mp3",
            "ك": "comp_kaf_ك_كاف_sukoon.mp3",
            "ل": "comp_lam_ل_lam_latin.mp3",
            "م": "letter_م_meem.mp3",
            "ن": "letter_ن_noon.mp3",
            "ه": "comp_ha_ه_ه_ه.mp3",
            "و": "comp_waw_و_واو_diac.mp3",
            "ي": "letter_ي_yeh.mp3",
        }

        self.NUMBER_AUDIO_MAP = {

            "صفر": "num_00_v02.mp3",
            "0": "num_00_v02.mp3",

            "واحد": "num_01.mp3",
            "1": "num_01.mp3",

            "اثنين": "num_02_atneen_bang_cut_0_59.mp3",
            "2": "num_02_atneen_bang_cut_0_59.mp3",

            "ثلاثة": "num_03.mp3",
            "3": "num_03.mp3",

            "أربعة": "num_04.mp3",
            "4": "num_04.mp3",

            "خمسة": "num_05.mp3",
            "5": "num_05.mp3",

            "ستة": "num_06.mp3",
            "6": "num_06.mp3",

            "سبعة": "num_07.mp3",
            "7": "num_07.mp3",

            "ثمانية": "num_08.mp3",
            "8": "num_08.mp3",

            "تسعة": "num_09.mp3",
            "9": "num_09.mp3",

            "عشرة": "num_10_asharah_sukoon.mp3",
            "10": "num_10_asharah_sukoon.mp3",
        }

    # ============================================================
    # PLAY AUDIO
    # ============================================================

    def speak_prediction(self, label):

        now = time.time()

        # prevent repeating same audio rapidly
        if label == self.last_spoken and now - self.last_speak_time < 2:
            return

        self.last_spoken = label
        self.last_speak_time = now

        audio_path = None

        # letters
        if label in self.LETTER_AUDIO_MAP:

            audio_path = (
                self.LETTERS_CHOSEN_DIR /
                self.LETTER_AUDIO_MAP[label]
            )

        # numbers
        elif label in self.NUMBER_AUDIO_MAP:

            audio_path = (
                self.NUMBERS_CHOSEN_DIR /
                self.NUMBER_AUDIO_MAP[label]
            )

        if audio_path and audio_path.exists():

            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.load(str(audio_path))
                pygame.mixer.music.play()

                print(f"🔊 Playing: {audio_path.name}")

            except Exception as e:
                print("Audio Error:", e)

    # ============================================================
    # UI SETUP
    # ============================================================

    def setup_ui(self):

        self.pages = {}

        # ========================================================
        # PAGE 1
        # ========================================================

        p1 = tk.Frame(self.root, bg="white")
        self.pages["left"] = p1

        tk.Label(
            p1,
            text="LEFT HAND MODEL",
            font=("Arial", 22, "bold"),
            bg="white",
            fg="#0077b6"
        ).pack(pady=20)

        self.left_label = tk.Label(
            p1,
            text="---",
            font=("Arial", 40),
            bg="white",
            fg="#0077b6"
        )

        self.left_label.pack()

        # ========================================================
        # PAGE 2
        # ========================================================

        p2 = tk.Frame(self.root, bg="white")
        self.pages["arabic"] = p2

        tk.Label(
            p2,
            text="ARABIC MODEL (BOTH HANDS)",
            font=("Arial", 22, "bold"),
            bg="white",
            fg="#023e8a"
        ).pack(pady=20)

        self.ar_label = tk.Label(
            p2,
            text="---",
            font=("Arial", 40),
            bg="white",
            fg="#023e8a"
        )

        self.ar_label.pack()

        # ========================================================
        # PAGE 3
        # ========================================================

        p3 = tk.Frame(self.root, bg="white")
        self.pages["multi"] = p3

        tk.Label(
            p3,
            text="MULTI MODEL FUSION",
            font=("Arial", 22, "bold"),
            bg="white",
            fg="#0096c7"
        ).pack(pady=20)

        self.multi_label = tk.Label(
            p3,
            text="---",
            font=("Arial", 40),
            bg="white",
            fg="#0096c7"
        )

        self.multi_label.pack()

        # ========================================================
        # NAVIGATION
        # ========================================================

        nav = tk.Frame(self.root, bg="#caf0f8")
        nav.pack(side=tk.BOTTOM, fill=tk.X)

        tk.Button(
            nav,
            text="Left",
            bg="#90e0ef",
            command=lambda: self.show_page("left")
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)

        tk.Button(
            nav,
            text="Arabic",
            bg="#90e0ef",
            command=lambda: self.show_page("arabic")
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)

        tk.Button(
            nav,
            text="Fusion",
            bg="#90e0ef",
            command=lambda: self.show_page("multi")
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)

    # ============================================================
    # PAGE SWITCH
    # ============================================================

    def show_page(self, name):

        for p in self.pages.values():
            p.pack_forget()

        self.pages[name].pack(expand=True, fill=tk.BOTH)

        self.page = name

    # ============================================================
    # UDP RECEIVER
    # ============================================================

    def start_udp(self):

        def run():

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("0.0.0.0", PORT))
            sock.setblocking(False)

            print(f"✅ UDP listening on port {PORT}")

            while True:

                try:

                    data, _ = sock.recvfrom(4096)

                    msg = data.decode(errors="ignore").split(",")

                    if len(msg) != 12:
                        continue

                    hand = msg[0].upper()

                    values = list(map(float, msg[1:12]))

                    with self.lock:

                        if hand == "LEFT":

                            self.left = values
                            self.lt = time.time()

                        elif hand == "RIGHT":

                            self.right = values
                            self.rt = time.time()

                except:
                    pass

        threading.Thread(target=run, daemon=True).start()

    # ============================================================
    # MAIN LOOP
    # ============================================================

    def loop(self):

        while True:

            time.sleep(0.03)

            with self.lock:

                left = self.left.copy()
                right = self.right.copy()

                lt = self.lt
                rt = self.rt

            now = time.time()

            left_ok = now - lt < TIMEOUT
            right_ok = now - rt < TIMEOUT

            # =====================================================
            # LEFT MODEL
            # =====================================================

            if self.page == "left":

                if left_ok:

                    X = self.left_scaler.transform([left])

                    pred = self.left_model.predict(X)[0]

                    label = self.left_encoder.inverse_transform([pred])[0]

                    self.left_label.config(text=label)

                    self.speak_prediction(label)

            # =====================================================
            # ARABIC MODEL
            # =====================================================

            elif self.page == "arabic":

                if left_ok or right_ok:

                    if left_ok and right_ok:

                        full = left + right

                    elif left_ok:

                        full = left + [0] * 11

                    else:

                        full = [0] * 11 + right

                    X = self.ar_scaler.transform([full])

                    if self.ar_pca:
                        X = self.ar_pca.transform(X)

                    pred = self.ar_model.predict(X)[0]

                    label = self.ar_encoder.inverse_transform([pred])[0]

                    self.ar_label.config(text=label)

                    self.speak_prediction(label)

            # =====================================================
            # FUSION MODEL
            # =====================================================

            elif self.page == "multi":

                results = []

                if left_ok:

                    X1 = self.left_scaler.transform([left])

                    p1 = self.left_model.predict(X1)[0]

                    results.append(
                        self.left_encoder.inverse_transform([p1])[0]
                    )

                if left_ok or right_ok:

                    if left_ok and right_ok:
                        full = left + right

                    elif left_ok:
                        full = left + [0] * 11

                    else:
                        full = [0] * 11 + right

                    X2 = self.ar_scaler.transform([full])

                    if self.ar_pca:
                        X2 = self.ar_pca.transform(X2)

                    p2 = self.ar_model.predict(X2)[0]

                    results.append(
                        self.ar_encoder.inverse_transform([p2])[0]
                    )

                if results:

                    final = max(set(results), key=results.count)

                    self.multi_label.config(text=final)

                    self.speak_prediction(final)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = SignApp(root)

    root.mainloop()