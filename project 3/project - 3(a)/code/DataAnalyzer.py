from scipy.stats import kurtosis, skew
import numpy as np

def get_features(x, fs):    
    f1 = np.mean(x, axis=0)
    cov = np.cov(np.transpose(x))
    f2 = np.append(list(cov[0, 0:3]), list(cov[1, 1:3]))
    f2 = np.append(list(f2), list(cov[2, 2:3]))
    f3 = np.array([skew(x[:, 0]), skew(x[:, 1]), skew(x[:, 2])])
    f4 = np.array([kurtosis(x[:, 0], fisher=False), kurtosis(x[:, 1], fisher=False), kurtosis(x[:, 2], fisher=False)])
    
    fs = 50
    f5 = np.zeros(3)
    f6 = np.zeros(3)
    
    for i in range(3):
        g = abs(np.fft.fft(x[:,i]))
        g = g[0:round(len(g)/2)]
        g[0] = 0
        v = max(g)
        idx = np.argmax(g)
        f5[i] = v
        f6[i] = fs/(2*len(g))*(idx)
        
    features = np.concatenate((f1, f2, f3, f4, f5, f6))
    return features


def apply_window_features(imu, length, slide, fs, test = False):
    acc = imu[:, 0:3]
    gyro = imu[:, 3:6]

    acc_features = []
    gyro_features = []
    
    index = 0
    while (index+length-1) < len(imu):
        acc_x = acc[index:(index+length), :]
        gyro_x = gyro[index:(index+length), :]
        acc_feat = get_features(acc_x, fs)
        gyro_feat = get_features(gyro_x, fs)
        acc_features.append(acc_feat)
        gyro_features.append(gyro_feat)
        index += slide
    if test:
        acc_x = acc[index:, :]
        gyro_x = gyro[index:, :]
        acc_feat = get_features(acc_x, fs)
        gyro_feat = get_features(gyro_x, fs)
        acc_features.append(acc_feat)
        gyro_features.append(gyro_feat)
	  
    #print("Features generated using sliding window")
    return np.concatenate((np.array(acc_features), np.array(gyro_features)),axis=1)