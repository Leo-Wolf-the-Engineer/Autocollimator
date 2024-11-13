import cv2
import numpy as np
from pypylon import pylon
from scipy.optimize import curve_fit


# Funktion zur Definition der Gaußschen Kurve
def gaussian(x, amp, mean, stddev):
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


# Verbindung zur Kamera herstellen
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# Kamera konfigurieren
#camera.PixelFormat = "BGR8"
#camera.ExposureTimeAbs.Value = 20.0  # Belichtungszeit auf 20 Mikrosekunden setzen
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# Frame-Größe ermitteln
grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
frame = grabResult.Array
height, width, _ = frame.shape
grabResult.Release()

# Hauptschleife zur Verarbeitung der Frames
while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        frame = grabResult.Array

        # Extrahieren der blauen Pixel
        blue_channel = frame[:, :, 0]

        # Summieren der Intensitäten in X- und Y-Richtung
        intensity_x = np.sum(blue_channel, axis=0)
        intensity_y = np.sum(blue_channel, axis=1)

        # X-Richtung fitten
        x = np.arange(width)
        popt_x, _ = curve_fit(gaussian, x, intensity_x, p0=[np.max(intensity_x), np.argmax(intensity_x), 10])

        # Y-Richtung fitten
        y = np.arange(height)
        popt_y, _ = curve_fit(gaussian, y, intensity_y, p0=[np.max(intensity_y), np.argmax(intensity_y), 10])

        # Ermittelten Werte ausgeben
        #print(f"Frame: {grabResult.GetFrameNumber()}")
        print(f"X-Richtung: Amplitude={popt_x[0]:.2f}, Mittelwert={popt_x[1]:.2f}, Standardabweichung={popt_x[2]:.2f}")
        print(f"Y-Richtung: Amplitude={popt_y[0]:.2f}, Mittelwert={popt_y[1]:.2f}, Standardabweichung={popt_y[2]:.2f}")

        # Aktuellen Frame plotten
        plot_current_frame(frame)

    grabResult.Release()

camera.StopGrabbing()
camera.Close()