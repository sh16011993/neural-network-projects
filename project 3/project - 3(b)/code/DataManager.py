import pickle
import numpy as np
import DataProcessor as dp
import DataAnalyzer as da
import DataReader as dr

def save_data(file_name, data):
    file = open(file_name + ".obj", 'wb')
    pickle.dump(data, file)
    
def load_data(file_name):
    file = open(file_name + ".obj", 'rb') 
    data = pickle.load(file)
    return data
    
def generate_windowed_data(length, slide, fs):
    train_arm, train_wrist, train_detect = dr.get_train_data()
    train_arm = np.vstack(train_arm)
    train_wrist = np.vstack(train_wrist)
    train_detect = np.hstack(train_detect)

    _, _, train_detect_window = dp.apply_window(length, slide, train_arm, train_wrist, train_detect)
    train_arm_window = da.apply_window_features(train_arm, length, slide, fs)
    train_wrist_window = da.apply_window_features(train_wrist, length, slide, fs)
    
    print("Training: Full Arm Shape: " + str(np.shape(train_arm)))
    print("Training: Full Wrist Shape: " + str(np.shape(train_wrist)))
    print("Training: Full Detection Shape: " + str(np.shape(train_detect)))
    print("Training: Windowed Arm Shape: " + str(np.shape(train_arm_window)))
    print("Training: Windowed Wrist Shape: " + str(np.shape(train_wrist_window)))
    print("Training: Windowed Detect Shape: " + str(np.shape(train_detect_window)))
    
    
    val_arm, val_wrist, val_detect = dr.get_val_data()
    val_arm = np.vstack(val_arm)
    val_wrist = np.vstack(val_wrist)
    val_detect = np.hstack(val_detect)
    
    _, _, val_detect_window = dp.apply_window(length, slide, val_arm, val_wrist, val_detect)
    val_arm_window = da.apply_window_features(val_arm, length, slide, fs)
    val_wrist_window = da.apply_window_features(val_wrist, length, slide, fs)
    
    print("Validation: Windowed Arm Shape: " + str(np.shape(val_arm_window)))
    print("Validation: Windowed Wrist Shape: " + str(np.shape(val_wrist_window)))
    print("Validation: Windowed Detect Shape: " + str(np.shape(val_detect_window)))
    
    data = {"val_arm":val_arm_window, 
            "val_wrist":val_wrist_window, 
            "val_detect":val_detect_window,
            "train_arm":train_arm_window,
            "train_wrist":train_wrist_window,
            "train_detect":train_detect_window}
    
    return data

def save_windowed_data(file_name, length, slide, fs):
    data = generate_windowed_data(length, slide, fs)
    save_data(file_name, data)
    
def load_windowed_data(length, slide):
    file_name = "Data_" + str(length) + "_" + str(slide)
    data = load_data(file_name)
    return data
    

'''lengths = [100, 150, 200, 250, 300, 350, 400]
slides = [10, 25, 50]
for slide in slides:
    for length in lengths:
        #slide = 25
        file_name = "Data_" + str(length) + "_" + str(slide)
        save_windowed_data(file_name, length, slide, 50)'''