import os
import sys
import time
import numpy as np
import tensorflow as tf
import pickle
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QFrame
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg

print("Starting EEG Classifier Application")

class PlotManager:
    def __init__(self, layout):
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1a1a2e')
        self.plot_widget.setTitle("EEG Signal", color="w", size="15pt")
        self.plot_widget.setLabel('left', 'Amplitude', color='white', size=15)
        self.plot_widget.setLabel('bottom', 'Samples', color='white', size=15)
        self.plot_widget.addLegend()
        self.line = self.plot_widget.plot(pen=pg.mkPen(color='r', width=2), name="EEG Signal")
        layout.addWidget(self.plot_widget)

    def update_plot(self, eeg_signal):
        self.line.setData(eeg_signal)

def update_eeg_plot(plot_manager, eeg_state_label, prediction_label, eeg_state_container):
    global count
    
    if count == 0:
        time.sleep(3)
        eeg_state_label.setText("Connected")
        eeg_state_container.setStyleSheet("background-color: #39FF14; border-radius: 10px;")
    
    if count >= X.shape[0]:
        print("Reached end of data")
        return

    eeg_signal = X[count][0, :]
    plot_manager.update_plot(eeg_signal)

    pred = model.predict(np.expand_dims(X[count], axis=0))
    pred = np.argmax(pred, axis=1)
    prediction = label_encoder.classes_[pred][0]
    
    prediction_label.setText(prediction)
    count += 1

def main():
    # Ensure the offscreen platform is set for headless environments
    # os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["DISPLAY"] = ":99"

    # Initialize QApplication before any other GUI components
    app = QApplication(sys.argv)

    # Load models and data
    global model, label_encoder, X, count
    with open('label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)
    
    X = np.load('X.npy')
    model = tf.keras.models.load_model('eegModel.h5')
    count = 0

    # Main GUI setup
    window = QWidget()
    main_layout = QVBoxLayout(window)

    # PlotManager to handle EEG signal plotting
    plot_manager = PlotManager(main_layout)

    # Bottom container layout
    bottom_container = QVBoxLayout()
    main_layout.addLayout(bottom_container)

    # EEG State Container
    eeg_state_container = QFrame()
    eeg_state_container.setStyleSheet("background-color: #72AEE6; border-radius: 10px;")
    eeg_state_container.setFixedSize(300, 100)

    eeg_state_label = QLabel("Idle", eeg_state_container)
    eeg_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    eeg_state_label.setStyleSheet("color: white; font-size: 40px; font-weight: bold;")

    eeg_state_layout = QVBoxLayout(eeg_state_container)
    eeg_state_layout.addWidget(eeg_state_label, alignment=Qt.AlignmentFlag.AlignCenter)
    bottom_container.addWidget(eeg_state_container, alignment=Qt.AlignmentFlag.AlignHCenter)

    # Prediction Container
    prediction_container = QFrame()
    prediction_container.setStyleSheet("background-color: #1d2951; border-radius: 10px;")
    prediction_container.setFixedSize(300, 100)

    prediction_label = QLabel("Prediction", prediction_container)
    prediction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    prediction_label.setStyleSheet("color: white; font-size: 40px; font-weight: bold;")

    prediction_layout = QVBoxLayout(prediction_container)
    prediction_layout.addWidget(prediction_label, alignment=Qt.AlignmentFlag.AlignCenter)
    bottom_container.addWidget(prediction_container, alignment=Qt.AlignmentFlag.AlignHCenter)

    # Finalize and display the window
    window.setLayout(main_layout)
    window.setWindowTitle("EEG Classifier")
    window.show()

    # Timer to update EEG plot
    timer = QTimer()
    timer.timeout.connect(lambda: update_eeg_plot(plot_manager, eeg_state_label, prediction_label, eeg_state_container))
    timer.start(1000)

    # Start application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
