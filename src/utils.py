from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QScreen
from datetime import datetime
import numpy as np
import sys

# Save the whole window as an image
def save_window_as_image(window,app):
    screen = app.primaryScreen()
    screenshot = screen.grabWindow(window.winId())
    filename = datetime.now().strftime(f"{window.windowTitle().replace(' ', '_').lower()}_%Y%m%d_%H%M%S.png")
    screenshot.save(filename, 'png')
