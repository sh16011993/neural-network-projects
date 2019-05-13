import numpy as np

def get_data(data_path, number, test):
    path = data_path
    armIMU = []
    wristIMU = []
    time = []
    detection = []

    if test: 
        files = ["armIMU", "wristIMU", "time"]
        data = [armIMU, wristIMU, time]
        for i in range(len(files)):
            file_name = path + "Session0" + str(number) + "/" + files[i] + ".txt"
            print("Reading: " + str(file_name))
            data[i] = read_lines(file_name)
    else:
        files = ["armIMU", "wristIMU", "time", "detection"]
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
    
def get_all_data(path, test = False):
    arms = []
    wrists = []
    detections = []
    if test:
        for i in range(1,5):
            armIMU, wristIMU, time = get_data(path, i, test)
            arms.append(armIMU)
            wrists.append(wristIMU)
        return arms, wrists
    else:
        for i in range(1,6):
            armIMU, wristIMU, time, detection = get_data(path, i, test)
            detection = detection.astype(int)
            detection = detection.ravel()
			
            arms.append(armIMU)
            wrists.append(wristIMU)
            detections.append(detection)
        return arms, wrists, detections
