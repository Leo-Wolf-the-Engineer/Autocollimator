import sys
from PyQt5 import QtWidgets
from win_live import AutocollimatorLiveWindow
from data_storage import ContinousDataStorage

# launch the live window
app = QtWidgets.QApplication([])
image_frame_storage = []
data_storage = ContinousDataStorage(1)
autocollimator_live_window = AutocollimatorLiveWindow(app, image_frame_storage, data_storage)
autocollimator_live_window.win.show()
app.exec_()