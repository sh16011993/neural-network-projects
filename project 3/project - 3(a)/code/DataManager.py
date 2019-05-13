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
    arms, wrists, detections = dr.get_all_data()
    
    #Generate training set
    train_arm = arms[0]
    train_wrist = wrists[0]
    train_detect = detections[0]
    for i in range(1,5):
        train_arm = np.concatenate((train_arm, arms[i]), axis=0)
        train_wrist = np.concatenate((train_wrist, wrists[i]), axis=0)
        train_detect = np.concatenate((train_detect, detections[i]), axis=0)

    _, _, train_detect_window = dp.apply_window(length, slide, train_arm, train_wrist, train_detect)
    train_arm_window = da.apply_window_features(train_arm, length, slide, fs)
    train_wrist_window = da.apply_window_features(train_wrist, length, slide, fs)
    
    print("Training: Full Arm Shape: " + str(np.shape(train_arm)))
    print("Training: Full Wrist Shape: " + str(np.shape(train_wrist)))
    print("Training: Full Detection Shape: " + str(np.shape(train_detect)))
    print("Training: Windowed Arm Shape: " + str(np.shape(train_arm_window)))
    print("Training: Windowed Wrist Shape: " + str(np.shape(train_wrist_window)))
    print("Training: Windowed Detect Shape: " + str(np.shape(train_detect_window)))
    
    #Generate test set
    test_arm, test_wrist, _, test_detect = dr.get_data(6)
    test_detect = test_detect.astype(int)
    test_detect = test_detect.ravel()
    _, _, test_detect_window = dp.apply_window(length, slide, test_arm, test_wrist, test_detect)
    test_arm_window = da.apply_window_features(test_arm, length, slide, fs)
    test_wrist_window = da.apply_window_features(test_wrist, length, slide, fs)
    
    print("Testing: Windowed Arm Shape: " + str(np.shape(test_arm_window)))
    print("Testing: Windowed Wrist Shape: " + str(np.shape(test_wrist_window)))
    print("Testing: Windowed Detect Shape: " + str(np.shape(test_detect_window)))
    
    data = {"test_arm":test_arm_window, 
            "test_wrist":test_wrist_window, 
            "test_detect":test_detect,
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

for length in lengths:
    slide = 25
    file_name = "Data_" + str(length) + "_" + str(slide)
    save_windowed_data(file_name, length, slide, 50)'''
