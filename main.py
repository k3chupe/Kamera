import cv2
import tkinter as tk
from tkinter import Label, Button, Frame, Scale, HORIZONTAL
from PIL import Image, ImageTk
import datetime
import os
import numpy as np

class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # 1. INICJALIZACJA Z DIRECTSHOW (Kluczowe dla Windows)
        # To pozwala na lepszą kontrolę sprzętową
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # Ustawienia domyślne rozdzielczości
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Zmienne stanu
        self.mode = "normal"
        self.is_recording = False
        self.out = None
        self.prev_frame = None

        if not os.path.exists("galeria"):
            os.makedirs("galeria")

        # 2. GUI
        self.video_label = Label(window)
        self.video_label.pack(pady=10)

        # --- PANEL STEROWANIA SPRZĘTOWEGO (HARDWARE) ---
        self.hw_frame = Frame(window, bd=2, relief=tk.GROOVE, bg="#dddddd")
        self.hw_frame.pack(fill=tk.X, padx=10, pady=5)
        
        Label(self.hw_frame, text="STEROWANIE SPRZĘTOWE (HARDWARE)", bg="#dddddd", font=("Arial", 10, "bold")).pack(pady=2)

        # PRZYCISK "ATOMOWY" - Otwiera okno sterownika Windows
        self.btn_settings = Button(self.hw_frame, text="OTWÓRZ SYSTEMOWY PANEL KAMERY", 
                                   command=self.open_camera_settings, bg="#2196F3", fg="white", font=("Arial", 9, "bold"))
        self.btn_settings.pack(pady=5, fill=tk.X, padx=20)
        Label(self.hw_frame, text="(Najpewniejszy sposób na zablokowanie ekspozycji)", bg="#dddddd", font=("Arial", 8)).pack()

        # Suwak Ekspozycji (Programowa próba sterowania)
        self.slider_frame = Frame(self.hw_frame, bg="#dddddd")
        self.slider_frame.pack(pady=5)
        
        Label(self.slider_frame, text="Lub spróbuj wymusić suwakiem (Exposure):", bg="#dddddd").pack()
        # Zakres -13 (bardzo ciemno) do -1 (bardzo jasno). 
        # Wartości dodatnie często nie działają w DSHOW.
        self.exposure_slider = Scale(self.slider_frame, from_=-13, to=-1, resolution=1,
                                     orient=HORIZONTAL, length=300, command=self.force_exposure, bg="#dddddd")
        self.exposure_slider.set(-6)
        self.exposure_slider.pack()
        # ----------------------------------------------------

        # Panel przycisków nagrywania/zdjęć
        self.controls_frame = Frame(window)
        self.controls_frame.pack(fill=tk.X, pady=5)

        self.btn_snapshot = Button(self.controls_frame, text="Zrób Zdjęcie", width=15, command=self.take_snapshot, bg="#4CAF50", fg="white")
        self.btn_snapshot.pack(side=tk.LEFT, padx=10)

        self.btn_record = Button(self.controls_frame, text="Nagraj Wideo: STOP", width=20, command=self.toggle_recording, bg="#f44336", fg="white")
        self.btn_record.pack(side=tk.LEFT, padx=10)

        # Panel efektów
        self.effects_frame = Frame(window)
        self.effects_frame.pack(fill=tk.X, pady=10)
        
        self.btn_normal = Button(self.effects_frame, text="Normalny", command=lambda: self.set_mode("normal"))
        self.btn_normal.pack(side=tk.LEFT, padx=2)
        
        self.btn_motion = Button(self.effects_frame, text="Wykrywanie Ruchu", command=lambda: self.set_mode("motion"))
        self.btn_motion.pack(side=tk.LEFT, padx=2)
        
        self.btn_anaglyph = Button(self.effects_frame, text="Anaglif 3D", command=lambda: self.set_mode("anaglyph"))
        self.btn_anaglyph.pack(side=tk.LEFT, padx=2)

        self.status_label = Label(window, text="Status: Gotowy", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.delay = 15
        self.update()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    # --- FUNKCJE SPRZĘTOWE ---
    def open_camera_settings(self):
        # To jest "MAGICZNA" komenda.
        # Wymusza na systemie Windows otwarcie natywnego okna konfiguracji sterownika.
        self.cap.set(cv2.CAP_PROP_SETTINGS, 1)
        self.status_label.config(text="Otwarto panel sterownika. Zmień tam ustawienia.")

    def force_exposure(self, val):
        # Próba "siłowego" wyłączenia Auto i ustawienia wartości
        value = int(val)
        
        # KROK 1: Próba wyłączenia Auto Exposure (różne flagi dla różnych kamer)
        # 1 = Manual (zazwyczaj)
        # 0.25 = Manual (często w sterownikach DirectShow)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
        
        # KROK 2: Ustawienie wartości
        self.cap.set(cv2.CAP_PROP_EXPOSURE, value)
        print(f"Próba ustawienia ekspozycji: {value}")
    # -------------------------

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            processed_frame = self.process_frame(frame)

            if self.is_recording and self.out is not None:
                self.out.write(processed_frame)

            if len(processed_frame.shape) == 2:
                img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
            self.video_label.config(image=self.photo)
            
            self.prev_frame = frame.copy()

        self.window.after(self.delay, self.update)

    def process_frame(self, frame):
        if self.mode == "normal":
            return frame
        elif self.mode == "motion":
            if self.prev_frame is None: return frame
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray_current, gray_prev)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            return thresh
        elif self.mode == "anaglyph":
            if self.prev_frame is None: return frame
            h, w, _ = frame.shape
            anaglyph = np.zeros((h, w, 3), dtype=np.uint8)
            anaglyph[:,:,0] = frame[:,:,0]
            anaglyph[:,:,1] = frame[:,:,1]
            anaglyph[:,:,2] = self.prev_frame[:,:,2]
            return anaglyph
        return frame

    def set_mode(self, mode):
        self.mode = mode
        self.status_label.config(text=f"Tryb: {mode}")

    def take_snapshot(self):
        ret, frame = self.cap.read()
        if ret:
            processed = self.process_frame(frame)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"galeria/foto_{ts}.jpg", processed)
            self.status_label.config(text="Zapisano zdjęcie")

    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.btn_record.config(text="STOP", bg="orange")
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            is_color = (self.mode != "motion")
            self.out = cv2.VideoWriter(f"galeria/video_{ts}.avi", fourcc, 20.0, (640,480), is_color)
        else:
            self.is_recording = False
            self.btn_record.config(text="Nagraj", bg="#f44336")
            if self.out: self.out.release()
            self.out = None

    def on_closing(self):
        self.cap.release()
        if self.out: self.out.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root, "Kamera Lab - Pełna Kontrola Sprzętowa")