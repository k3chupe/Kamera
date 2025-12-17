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
        # UWAGA: Dodałem 'cv2.CAP_DSHOW' - to tryb DirectShow w Windows, 
        # który pozwala na lepszy dostęp do ustawień sprzętowych kamery.

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
        # W Windows zazwyczaj wartości są ujemne potęgi 2 (np. -5 to jasno, -10 to ciemno)
        # Ale zakres zależy od kamery. Ustawiam bezpieczny zakres -13 do -1.
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
        # 1 = Auto, 0 = Manual (w teorii)
        # W OpenCV często: 0.25 to Manual, 0.75 to Auto (dziwne mapowanie DirectShow)
        state = self.auto_exposure_var.get()
        
        if state == 1:
            # Włącz Auto
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # 3 często oznacza Auto w Windows
            print("Tryb Auto włączony")
        else:
            # Wyłącz Auto (Przejdź na Manual)
            # Często trzeba ustawić 1 (Manual) albo 0.25 w zależności od sterownika
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
            print("Tryb Manualny włączony")
            
            # Po przejściu na manual, odśwież wartość z suwaka
            val = self.exposure_slider.get()
            self.cap.set(cv2.CAP_PROP_EXPOSURE, val)

    def set_exposure(self, val):
        # Działa tylko jeśli Auto jest wyłączone!
        if self.auto_exposure_var.get() == 0:
            value = int(val)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, value)
            print(f"Ustawiono Hardware Exposure: {value}")

    # -------------------------

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            # Tutaj NIE MA już cyfrowego rozjaśniania (convertScaleAbs)
            # Obraz przychodzi z kamery już jasny/ciemny sprzętowo
            
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

    # Reszta funkcji bez zmian (take_snapshot, toggle_recording, set_mode...)
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

    def on_closing(self):
        self.cap.release()
        if self.out: self.out.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root, "Kamera Laboratorium - Sterowanie Sprzętowe")