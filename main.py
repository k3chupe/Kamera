import cv2
import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk
import datetime
import os
import numpy as np

class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # 1. INICJALIZACJA KAMERY
        self.cap = cv2.VideoCapture(0) # 0 to domyślna kamera USB
        
        # Ustawienia rozdzielczości (opcjonalne, dla płynności)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Zmienne stanu
        self.mode = "normal" # tryby: normal, motion, anaglyph
        self.is_recording = False
        self.out = None # obiekt do zapisu wideo
        self.prev_frame = None # potrzebne do wykrywania ruchu i anaglifów

        # Utwórz folder na wyniki, jeśli nie istnieje
        if not os.path.exists("galeria"):
            os.makedirs("galeria")

        # 2. GUI - INTERFEJS UŻYTKOWNIKA
        # Główny panel wideo
        self.video_label = Label(window)
        self.video_label.pack(pady=10)

        # Panel przycisków
        self.controls_frame = Frame(window)
        self.controls_frame.pack(fill=tk.X, pady=5)

        # Przyciski
        self.btn_snapshot = Button(self.controls_frame, text="Zrób Zdjęcie", width=15, command=self.take_snapshot, bg="#4CAF50", fg="white")
        self.btn_snapshot.pack(side=tk.LEFT, padx=5)

        self.btn_record = Button(self.controls_frame, text="Nagraj Wideo: STOP", width=20, command=self.toggle_recording, bg="#f44336", fg="white")
        self.btn_record.pack(side=tk.LEFT, padx=5)

        # Panel wyboru efektów
        self.effects_frame = Frame(window)
        self.effects_frame.pack(fill=tk.X, pady=5)
        
        Label(self.effects_frame, text="Wybierz tryb:").pack(side=tk.LEFT, padx=5)
        
        self.btn_normal = Button(self.effects_frame, text="Normalny", command=lambda: self.set_mode("normal"))
        self.btn_normal.pack(side=tk.LEFT, padx=2)
        
        self.btn_motion = Button(self.effects_frame, text="Wykrywanie Ruchu", command=lambda: self.set_mode("motion"))
        self.btn_motion.pack(side=tk.LEFT, padx=2)
        
        self.btn_anaglyph = Button(self.effects_frame, text="Anaglif 3D", command=lambda: self.set_mode("anaglyph"))
        self.btn_anaglyph.pack(side=tk.LEFT, padx=2)

        # Etykieta statusu
        self.status_label = Label(window, text="Status: Gotowy", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Start pętli odczytu wideo
        self.delay = 15 # ms (odświeżanie ok. 60 FPS)
        self.update()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    # 3. GŁÓWNA PĘTLA PROGRAMU
    def update(self):
        ret, frame = self.cap.read()
        if ret:
            # Obróć, jeśli kamera pokazuje lustrzane odbicie (opcjonalne)
            # frame = cv2.flip(frame, 1)

            # Przetwarzanie obrazu w zależności od trybu
            processed_frame = self.process_frame(frame)

            # Nagrywanie wideo (jeśli włączone)
            if self.is_recording and self.out is not None:
                self.out.write(processed_frame)

            # Konwersja do wyświetlenia w Tkinter (BGR -> RGB)
            # Uwaga: w trybie ruchu obraz może być czarno-biały, trzeba to obsłużyć
            if len(processed_frame.shape) == 2: # Obraz czarno-biały (GRAY)
                img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)
            else: # Obraz kolorowy (BGR)
                img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
            self.video_label.config(image=self.photo)

            # Zapisz obecną klatkę jako 'poprzednią' dla następnego obiegu pętli
            self.prev_frame = frame.copy()

        self.window.after(self.delay, self.update)

    # 4. LOGIKA PRZETWARZANIA OBRAZU (SENS ZADANIA)
    def process_frame(self, frame):
        if self.mode == "normal":
            return frame
        
        elif self.mode == "motion":
            # Wykrywanie ruchu: Różnica między obecną a poprzednią klatką
            if self.prev_frame is None:
                return frame
            
            # Konwersja na odcienie szarości
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            
            # Obliczenie różnicy bezwzględnej
            diff = cv2.absdiff(gray_current, gray_prev)
            
            # Progowanie (zwiększenie kontrastu ruchu)
            # Piksele o różnicy < 25 stają się czarne, > 25 białe
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            return thresh

        elif self.mode == "anaglyph":
            # Symulacja 3D: Przesunięcie czasowe
            # Lewe oko = poprzednia klatka, Prawe oko = obecna klatka
            if self.prev_frame is None:
                return frame
            
            height, width, _ = frame.shape
            anaglyph = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Matrix anaglifu (prosty):
            # Kanał R (czerwony) z klatki poprzedniej
            # Kanały G i B (cyjan) z klatki obecnej
            anaglyph[:, :, 0] = frame[:, :, 0]       # Blue z obecnej
            anaglyph[:, :, 1] = frame[:, :, 1]       # Green z obecnej
            anaglyph[:, :, 2] = self.prev_frame[:, :, 2] # Red z poprzedniej (tworzy przesunięcie)
            
            return anaglyph
        
        return frame

    # 5. OBSŁUGA PRZYCISKÓW
    def set_mode(self, mode):
        self.mode = mode
        self.status_label.config(text=f"Status: Tryb {mode}")

    def take_snapshot(self):
        # Pobieramy aktualnie przetworzoną klatkę
        ret, frame = self.cap.read()
        if ret:
            processed = self.process_frame(frame)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"galeria/foto_{timestamp}_{self.mode}.jpg"
            cv2.imwrite(filename, processed)
            print(f"Zapisano zdjęcie: {filename}")
            self.status_label.config(text=f"Zapisano: {filename}")

    def toggle_recording(self):
        if not self.is_recording:
            # START NAGRYWANIA
            self.is_recording = True
            self.btn_record.config(text="Nagraj Wideo: NAGRYWANIE", bg="orange")
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"galeria/wideo_{timestamp}.avi"
            
            # Kodek dla Windows (XVID lub MJPG)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            
            # Sprawdzamy czy nagrywamy w kolorze czy B&W (dla Motion Detect)
            is_color = True
            if self.mode == "motion":
                is_color = False
            
            self.out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480), is_color)
            self.status_label.config(text=f"Nagrywanie: {filename}")
        else:
            # STOP NAGRYWANIA
            self.is_recording = False
            self.btn_record.config(text="Nagraj Wideo: STOP", bg="#f44336")
            if self.out:
                self.out.release()
                self.out = None
            self.status_label.config(text="Nagrywanie zakończone")

    def on_closing(self):
        # Sprzątanie po zamknięciu
        if self.cap.isOpened():
            self.cap.release()
        if self.out:
            self.out.release()
        self.window.destroy()

# Uruchomienie aplikacji
if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root, "Kamera Laboratorium - Python OpenCV")