import numpy as np

DATA_PATH = './AllData/'

files = ["armIMU", "wristIMU", "time", "detection"]

def get_data(number):
    path = DATA_PATH
    armIMU = []
    wristIMU = []
    time = []
    detection = []

    data = [armIMU, wristIMU, time, detection]
    for i in range(len(files)):
        file_name = path + files[i] + str(number) + ".txt"
        print("Reading: " + str(file_name))
        data[i] = read_lines(file_name)
    return tuple(data)
    
def read_lines(file):
    #Returns text file lines as a numpy array
    text_data = []
    float_data = []

    #Read in data as text
    with open(file) as textFile:
        for line in textFile:
            text_data.append(line)
            
    #Convert data to list of floats
    for line in text_data:
        line = line.split()
        line = [float(number) for number in line]
        float_data.append(line)
    return np.array(float_data)
    
def get_train_data():
    arms = []
    wrists = []
    detections = []
    for i in range(1,9):
        armIMU, wristIMU, time, detection = get_data(i)
        detection = detection.astype(int)
        detection = detection.ravel()
        
        arms.append(armIMU)
        wrists.append(wristIMU)
        detections.append(detection)
    return arms, wrists, detections
    
def get_val_data():
    arms = []
    wrists = []
    detections = []
    for i in range(9,11):
        armIMU, wristIMU, time, detection = get_data(i)
        detection = detection.astype(int)
        detection = detection.ravel()
        
        arms.append(armIMU)
        wrists.append(wristIMU)
        detections.append(detection)
    return arms, wrists, detections

def get_lstm_data():
    arms, wrists, detections = get_train_data()
    train_arms = arms[0]
    train_wrists = wrists[0]
    train_detections = detections[0]

    #Append multiple lists of training data
    for i in range(1, len(arms)):
        train_arms = np.concatenate((train_arms, arms[i]), axis=0)
        train_wrists = np.concatenate((train_wrists, wrists[i]), axis=0)
        train_detections = np.concatenate((train_detections, detections[i]), axis=0)
    
    train_arms_wrists = np.concatenate((train_arms, train_wrists), axis=1)
    
    arms, wrists, detections = get_val_data()
    val_arms = arms[0]
    val_wrists = wrists[0]
    val_detections = detections[0]
    
    #Append multiple lists of validation data
    for i in range(1, len(arms)):
        val_arms = np.concatenate((val_arms, arms[i]), axis=0)
        val_wrists = np.concatenate((val_wrists, wrists[i]), axis=0)
        val_detections = np.concatenate((val_detections, detections[i]), axis=0)
    
    val_arms_wrists = np.concatenate((val_arms, val_wrists), axis=1)
    
    return train_arms_wrists, train_detections, val_arms_wrists, val_detections

def window_data(length, slide, data):
    windowed_data = []
    
    index = 0
    while (index + length) <= len(data):
        windowed_data.append(data[index:index+length])
        index = index+slide
    
    windowed_data = np.dstack(windowed_data)
    windowed_data = np.rollaxis(windowed_data, -1)
    return windowed_data
    
#def window_detections(length, slide, data):
#    windowed_data = []
#    
#    index = 0
#    while (index + length) <= len(data):
#        windowed_data.append(data[index:index+length])
#        index = index+slide
#    
#    windowed_data = np.array(windowed_data)
#    return windowed_data

def window_detections(length, slide, data):
    windowed_data = []
    
    index = 0
    while (index + length) <= len(data):
        current = data[index:index+length]
        val = int(np.round(np.mean(current)))
        windowed_data.append(val)
        index = index+slide
    
    windowed_data = np.array(windowed_data)
    return windowed_data
