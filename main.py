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

        # Używamy DirectShow (DSHOW) - to najlepszy backend dla Windows
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # Ustawienia rozdzielczości
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Zmienne
        self.mode = "normal"
        self.is_recording = False
        self.out = None
        self.prev_frame = None

        if not os.path.exists("galeria"):
            os.makedirs("galeria")

        # --- GUI ---
        self.video_label = Label(window)
        self.video_label.pack(pady=10)

        # PANEL STEROWANIA EKSPOZYCJĄ
        self.hw_frame = Frame(window, bd=2, relief=tk.GROOVE, bg="#f0f0f0")
        self.hw_frame.pack(fill=tk.X, padx=10, pady=5)
        Label(self.hw_frame, text="KONTROLA EKSPOZYCJI", bg="#f0f0f0", font=("Arial", 10, "bold")).pack()

        # Przycisk: WYMUŚ TRYB MANUALNY
        self.btn_manual = Button(self.hw_frame, text="1. Kliknij: ZABLOKUJ AUTO-EKSPOZYCJĘ", 
                                 command=self.disable_auto_exposure, bg="#FF9800", fg="white")
        self.btn_manual.pack(fill=tk.X, padx=20, pady=5)

        # Suwak
        Label(self.hw_frame, text="2. Ustaw Jasność (Exposure):", bg="#f0f0f0").pack()
        # Zakres dla większości kamer Logitech/Microsoft/Laptop to od -13 do -1
        # -13 = Bardzo jasno (długi czas)
        # -1  = Bardzo ciemno (krótki czas)
        self.exposure_slider = Scale(self.hw_frame, from_=-13, to=-1, resolution=1,
                                     orient=HORIZONTAL, length=300, command=self.set_exposure, bg="#f0f0f0")
        self.exposure_slider.set(-5) # Wartość startowa
        self.exposure_slider.pack(pady=5)

        # Pozostałe przyciski
        self.controls = Frame(window)
        self.controls.pack(fill=tk.X, pady=5)
        Button(self.controls, text="Zrób Zdjęcie", command=self.take_snapshot, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=10)
        self.btn_record = Button(self.controls, text="Nagraj: STOP", command=self.toggle_recording, bg="#f44336", fg="white")
        self.btn_record.pack(side=tk.LEFT, padx=10)

        # Efekty
        self.effects = Frame(window)
        self.effects.pack(fill=tk.X, pady=10)
        Button(self.effects, text="Normalny", command=lambda: self.set_mode("normal")).pack(side=tk.LEFT)
        Button(self.effects, text="Ruch", command=lambda: self.set_mode("motion")).pack(side=tk.LEFT)
        Button(self.effects, text="3D", command=lambda: self.set_mode("anaglyph")).pack(side=tk.LEFT)

        self.status = Label(window, text="Status: Gotowy", relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        self.delay = 15
        self.update()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    # --- KLUCZOWE FUNKCJE NAPRAWCZE ---
    def disable_auto_exposure(self):
        """Próbuje wszystkich znanych kodów, aby wyłączyć automat"""
        print("Próba wyłączenia Auto-Ekspozycji...")
        
        # Kod 0.25 to 'Manual' dla wielu sterowników DirectShow w Windows
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
        
        # Niektóre kamery wolą 1.0 jako Manual
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
        
        # Dla pewności ustawiamy jakąś wartość startową
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)
        
        self.status.config(text="Status: Wysłano komendę MANUAL. Sprawdź suwak.")
        print("Wysłano komendy blokady Auto.")

    def set_exposure(self, val):
        """Wysyła wartość bezpośrednio do sterownika"""
        value = float(val)
        # Najpierw upewniamy się, że Auto jest wyłączone (kod 0.25)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
        
        # Ustawiamy wartość
        result = self.cap.set(cv2.CAP_PROP_EXPOSURE, value)
        
        if result:
            print(f"Ustawiono ekspozycję: {value}")
        else:
            print(f"Kamera zignorowała ustawienie: {value}")

    # --- RESZTA LOGIKI (BEZ ZMIAN) ---
    def update(self):
        ret, frame = self.cap.read()
        if ret:
            processed = self.process_frame(frame)
            if self.is_recording and self.out:
                self.out.write(processed)
            
            # Konwersja kolorów
            if len(processed.shape) == 2:
                img = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.video_label.config(image=self.photo)
            self.prev_frame = frame.copy()
        self.window.after(self.delay, self.update)

    def process_frame(self, frame):
        if self.mode == "motion" and self.prev_frame is not None:
            gray_cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            return cv2.threshold(cv2.absdiff(gray_cur, gray_prev), 25, 255, cv2.THRESH_BINARY)[1]
        elif self.mode == "anaglyph" and self.prev_frame is not None:
            res = np.zeros_like(frame)
            res[:,:,0] = frame[:,:,0] # B
            res[:,:,1] = frame[:,:,1] # G
            res[:,:,2] = self.prev_frame[:,:,2] # R
            return res
        return frame

    def take_snapshot(self):
        ret, frame = self.cap.read()
        if ret:
            ts = datetime.datetime.now().strftime("%H%M%S")
            cv2.imwrite(f"galeria/foto_{ts}.jpg", self.process_frame(frame))
            self.status.config(text="Zapisano zdjęcie")

    def set_mode(self, mode):
        self.mode = mode

    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.btn_record.config(text="STOP", bg="orange")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            ts = datetime.datetime.now().strftime("%H%M%S")
            is_color = (self.mode != "motion")
            self.out = cv2.VideoWriter(f"galeria/vid_{ts}.avi", fourcc, 20.0, (640,480), is_color)
        else:
            self.is_recording = False
            self.btn_record.config(text="Nagraj", bg="#f44336")
            if self.out: self.out.release()

    def on_closing(self):
        self.cap.release()
        if self.out: self.out.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root, "Camera Control Fixed")