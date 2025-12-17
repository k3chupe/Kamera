import cv2
import tkinter as tk
from tkinter import Label, Button, Frame, Scale, HORIZONTAL, Checkbutton, IntVar
from PIL import Image, ImageTk
import datetime
import os
import numpy as np

class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # 1. INICJALIZACJA KAMERY
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Zmienne stanu
        self.mode = "normal"
        self.is_recording = False
        
        # Obiekty do zapisu wideo
        self.out = None       # AVI
        self.out_mp4 = None   # MP4 (DODANE)
        
        self.prev_frame = None

        if not os.path.exists("galeria"):
            os.makedirs("galeria")

        # 2. GUI
        self.video_label = Label(window)
        self.video_label.pack(pady=10)

        # --- PANEL STEROWANIA SPRZĘTOWEGO ---
        self.hw_frame = Frame(window, bd=2, relief=tk.GROOVE)
        self.hw_frame.pack(fill=tk.X, padx=10, pady=5)
        Label(self.hw_frame, text="USTAWIENIA SPRZĘTOWE KAMERY").pack()

        # Checkbox: Auto Ekspozycja
        self.auto_exposure_var = IntVar(value=1) # Domyślnie włączone
        self.cb_auto = Checkbutton(self.hw_frame, text="Auto Ekspozycja", 
                                   variable=self.auto_exposure_var, command=self.toggle_auto_exposure)
        self.cb_auto.pack()

        # Suwak Ekspozycji (Hardware)
        Label(self.hw_frame, text="Czas naświetlania (Exposure):").pack()
        self.exposure_slider = Scale(self.hw_frame, from_=-13, to=0, 
                                     orient=HORIZONTAL, length=300, command=self.set_exposure)
        self.exposure_slider.set(-5) 
        self.exposure_slider.pack(pady=5)
        # ------------------------------------

        # Panel przycisków
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
    def toggle_auto_exposure(self):
        state = self.auto_exposure_var.get()
        if state == 1:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) 
            print("Tryb Auto włączony")
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
            print("Tryb Manualny włączony")
            val = self.exposure_slider.get()
            self.cap.set(cv2.CAP_PROP_EXPOSURE, val)

    def set_exposure(self, val):
        if self.auto_exposure_var.get() == 0:
            value = int(val)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, value)
            print(f"Ustawiono Hardware Exposure: {value}")

    # -------------------------

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            processed_frame = self.process_frame(frame)

            # Zapisywanie klatek jeśli trwa nagrywanie
            if self.is_recording:
                # 1. Zapis AVI
                if self.out is not None:
                    self.out.write(processed_frame)
                # 2. Zapis MP4 (DODANE)
                if self.out_mp4 is not None:
                    self.out_mp4.write(processed_frame)

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
            
            # KROK 1: Różnica na pełnym obrazie kolorowym (3 kanały)
            # Wykrywa zmiany koloru, nawet jeśli jasność jest podobna
            diff_bgr = cv2.absdiff(frame, self.prev_frame)
            
            # KROK 2: Konwersja różnicy na odcienie szarości
            diff_gray = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2GRAY)
            
            # KROK 3: Technika KWADRATOWA (Squaring)
            # Zamieniamy na float, żeby móc podnosić do kwadratu bez błędów
            diff_float = diff_gray.astype(np.float32)
            
            # Wzór: (różnica ^ 2) / 255
            # Matematycznie "miażdży" małe różnice (szum) do zera,
            # a duże różnice (ruch) zostawia wyraźne.
            squared = (diff_float ** 2) / 255.0
            
            # Zamiana z powrotem na liczby całkowite (0-255)
            motion_img = squared.astype(np.uint8)
            
            # KROK 4: Odcięcie szumów z zachowaniem odcieni (THRESH_TOZERO)
            # Wszystko poniżej wartości 10 staje się idealnie czarne (0).
            # Wszystko powyżej 10 ZACHOWUJE swoją jasność (nie robi się 255).
            _, clean_motion = cv2.threshold(motion_img, 10, 255, cv2.THRESH_TOZERO)
            
            # KROK 5: Wzmocnienie wizualne
            # Ponieważ potęgowanie przyciemnia obraz, mnożymy wynik x3, 
            # żeby ruch był lepiej widoczny dla ludzkiego oka.
            clean_motion = cv2.multiply(clean_motion, 3)
            
            return clean_motion

        elif self.mode == "anaglyph":
            if self.prev_frame is None: return frame
            # Tworzenie obrazu 3D
            anaglyph = np.zeros_like(frame)
            anaglyph[:,:,0] = frame[:,:,0] # Kanał Blue z obecnej klatki
            anaglyph[:,:,1] = frame[:,:,1] # Kanał Green z obecnej klatki
            anaglyph[:,:,2] = self.prev_frame[:,:,2] # Kanał Red z POPRZEDNIEJ klatki
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
            print("Zapisano")

    def toggle_recording(self):
        if not self.is_recording:
            # START NAGRYWANIA
            self.is_recording = True
            self.btn_record.config(text="STOP (AVI+MP4)", bg="orange")
            
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            is_color = (self.mode != "motion")
            
            # 1. Konfiguracja AVI
            fourcc_avi = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(f"galeria/video_{ts}.avi", fourcc_avi, 20.0, (640,480), is_color)
            
            # 2. Konfiguracja MP4 (DODANE)
            fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
            self.out_mp4 = cv2.VideoWriter(f"galeria/video_{ts}.mp4", fourcc_mp4, 20.0, (640,480), is_color)
            
            print(f"Start nagrywania: {ts}")
            
        else:
            # STOP NAGRYWANIA
            self.is_recording = False
            self.btn_record.config(text="Nagraj Wideo", bg="#f44336")
            
            if self.out:
                self.out.release()
                self.out = None
                
            if self.out_mp4:
                self.out_mp4.release()
                self.out_mp4 = None
                
            print("Stop nagrywania")

    def on_closing(self):
        self.cap.release()
        if self.out: self.out.release()
        if self.out_mp4: self.out_mp4.release() # (DODANE)
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root, "Kamera Laboratorium - Sterowanie Sprzętowe + MP4")