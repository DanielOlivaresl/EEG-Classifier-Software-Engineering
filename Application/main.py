from PySide6.QtCore import QObject, Signal, Slot, QUrl
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import QApplication
import sys
import threading
import time
import pickle
import numpy as np
from keras.models import load_model



#We load in the encoder and the data 
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

X = np.load('X.npy')

model = load_model('eegModel.h5')


count =0




# BackendConnector class for updating text in QML
class BackendConnector(QObject):
    updateTextSignal = Signal(str)
    updateEEGStateSignal = Signal(str,str)
    @Slot(str)
    def updateText(self, text):
        self.updateTextSignal.emit(text)

    @Slot(str,str)
    def updateEEgState(self,text,color):
        self.updateEEGStateSignal.emit(text,color)




# Function to update text in QML (called from Python)
def update_text_in_qml(root, new_text):
    text2_object = root.findChild(QObject, "text2")  # Use objectName here
    if text2_object:
        text2_object.setProperty("text", new_text)
    else:
        print("Error: Could not find 'text2' object")


def updateEEGTextState(root, text,color):
    stateObject = root.findChild(QObject,"text1")
    containerObject = root.findChild(QObject,"rectangle1")
    if stateObject:
        stateObject.setProperty("text", text)
        containerObject.setProperty("color",color)
    else:
        print("Error, could not fint object")
    
        



#Function that will run on a separate thread to update the EEG state
def simulateEEGConection(backend):
    backend.updateEEGStateSignal.emit("Idle","#72AEE6")
    time.sleep(2)
    while True:
        
        backend.updateEEGStateSignal.emit("Connected", "#39FF14")
        time.sleep(3)


# Function that runs in a separate thread to continuously update text
def change_text_after_delay(backend):
    global count  # Use the global variable count
    time.sleep(2)  # Wait for 3 seconds
    
    while True:
        time.sleep(0.5)  
        
        if count >= len(X):
            break

        pred = model.predict(np.expand_dims(X[count],axis=0))
        pred = np.argmax(pred,axis=1)
        prediction = label_encoder.classes_[pred]
        
        
        
        
        
        
        backend.updateText(f"{prediction[0]}")
        count += 1  # Increment the count for the next update

# Main function to set up the application
def main():
    app = QApplication(sys.argv)
    engine = QQmlApplicationEngine()

    # Correct path to your QML file
    qml_file_path = "C:/Users/Danie/OneDrive/Desktop/Escuela/IA/Semestre 6/IS/EEG-Classifier-Softwate-Engineering/Application/App.qml"
    engine.load(QUrl.fromLocalFile(qml_file_path))

    # Check if QML file was loaded successfully
    if not engine.rootObjects():
        print(f"Error: Failed to load QML file from {qml_file_path}")
        sys.exit(-1)
    else:
        print("QML file loaded successfully")

    # Get the root object from the QML
    root = engine.rootObjects()[0]

    # Create backend connector object
    backend = BackendConnector()

    # Connect the Python signal to the QML text2 object
    backend.updateTextSignal.connect(lambda new_text: update_text_in_qml(root, new_text))
    
    #Connect the Python signal to the EEG state object
    backend.updateEEGStateSignal.connect(lambda eegStateText, colorText: updateEEGTextState(root,eegStateText,colorText))
    

    # Start the update thread
    threading.Thread(target=change_text_after_delay, args=(backend,), daemon=True).start()

    #We now start the eeg state update thread
    threading.Thread(target=simulateEEGConection,args=(backend,),daemon=True).start()


    # Execute the application
    sys.exit(app.exec())

# Entry point
if __name__ == '__main__':
    main()
