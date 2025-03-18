#unittest for win_live.py
from PyQt5 import QtWidgets
import unittest

from win_live import AutocollimatorLiveWindowThread

class TestAutocollimatorLiveWindowThread(unittest.TestCase):
    def test_init(self):
        app = QtWidgets.QApplication([])
        thread = AutocollimatorLiveWindowThread()
        self.assertEqual(thread.isRunning(), False)
        thread.start()
        self.assertEqual(thread.isRunning(), True)
        thread.quit()
        app.quit()


