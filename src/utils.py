import matplotlib.pyplot as plt
import cv2

# Funktion zum Plotten des aktuellen Frames und der Intensitätsverteilungen
def plot_current_frame(axs, frame, intensity_x, intensity_y):
    # Aktualisieren des Plots des aktuellen Frames
    axs[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Aktueller Frame")
    axs[0].axis('off')

    # Aktualisieren des Plots der Intensitätsverteilung in X-Richtung
    axs[1].cla()
    axs[1].plot(intensity_x)
    axs[1].set_title("Intensitätsverteilung in X-Richtung")
    axs[1].set_xlabel("Pixel")
    axs[1].set_ylabel("Intensität")

    # Aktualisieren des Plots der Intensitätsverteilung in Y-Richtung
    axs[2].cla()
    axs[2].plot(intensity_y)
    axs[2].set_title("Intensitätsverteilung in Y-Richtung")
    axs[2].set_xlabel("Pixel")
    axs[2].set_ylabel("Intensität")

    plt.draw()
    plt.pause(0.005)  # Kurze Pause von 5ms einlegen, um den Plot zu aktualisieren

# Plot initialisieren
def initialize_plot():
    plt.ion()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    return fig, axs