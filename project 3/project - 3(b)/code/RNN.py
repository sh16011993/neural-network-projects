import DataReader as dr
import DataProcessor as dp
import DataAnalyzer as da
import DataManager as dm
import numpy as np
	
# lstm model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import load_model

import matplotlib.pyplot as plt

def create_lstm(neurons, layers, timesteps, dimension, batch_size):
    
    model = Sequential()
    for i in range(layers):
        #If current LSTM layer is not the last, return third dimension 
        return_sequence = True
        if (i+1) >= layers:
            return_sequence = False
        print(return_sequence)
        
        #If current LSTM layer is first, set input shape
        if i == 0:
            model.add(LSTM(neurons, 
                           return_sequences=return_sequence, 
                           input_shape=(timesteps, dimension)))
        else:
            model.add(LSTM(neurons,
                           return_sequences=return_sequence))
    
    model.add(Dropout(0.5))
    #model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    return model
    
def load_lstm():
    print("Not implemented yet")

def fit_lstm(model, train_x, train_y, val_x, val_y, epochs, batch_size, verbose):
    model.fit(train_x, 
              train_y,
              batch_size=batch_size, 
              epochs=epochs,
              verbose=verbose,
              validation_data=(val_x, val_y))
    
    return model
    
def test_lstm(model, test_x):
    y_pred = model.predict(test_x)
    return y_pred
    
def get_features(length, slide):
    data = dm.load_windowed_data(length, slide)
    train_arm = data["train_arm"]
    train_wrist = data["train_wrist"]
    train_detect = data["train_detect"]
    val_arm = data["val_arm"]
    val_wrist = data["val_wrist"]
    val_detect = data["val_detect"]

    train_arm_wrist = np.concatenate((train_arm, train_wrist), axis=1)
    val_arm_wrist = np.concatenate((val_arm, val_wrist), axis=1)
    
    _, _, val_detect_full = dr.get_val_data()
    val_detect_full = np.hstack(val_detect_full)
    
    return train_arm_wrist, train_detect, val_arm_wrist, val_detect, val_detect_full

def run(train_x1, train_y1, val_x1, val_y1):  
    timesteps = 1
    slide = 1
    
    train_x = dr.window_data(timesteps, slide, train_x1)
    train_y = dr.window_detections(timesteps, slide, train_y1)
    val_x = dr.window_data(timesteps, slide, val_x1)
    val_y = dr.window_detections(timesteps, slide, val_y1)
    
    #Initial parameters
    neurons = 200
    layers = 2
    batch_size = 256
    epochs = 4
    verbose=True
    dimension = train_x.shape[2]

    model = create_lstm(neurons=neurons, 
                        layers=layers, 
                        timesteps=timesteps, 
                        dimension=dimension, 
                        batch_size=batch_size)
    
    
    shorten_train = -1*(train_x.shape[0] % batch_size)
    shorten_val = -1*(val_x.shape[0] % batch_size)
    
    model = fit_lstm(model=model, 
                     train_x=train_x[:shorten_train], 
                     train_y=train_y[:shorten_train], 
                     val_x=val_x[:shorten_val], 
                     val_y=val_y[:shorten_val], 
                     epochs=epochs,
                     batch_size=batch_size,
                     verbose=verbose)
    
    return model, train_x, train_y, val_x, val_y

def plot_300(predict, detect):
    print('Plotting')
    x = np.arange(1, 1501)
    plt.plot(x, detect[16000:17500], label = 'detections')
    plt.plot(x, predict[16000:17500], label = 'predictions')
    plt.xlabel('time')
    plt.legend(loc='upper left', fontsize=14)
    plt.title('Plotting Predictions vs Detections', fontsize=14)
    plt.savefig('proj3b_plt.png')
    plt.show()	
	
length = 300
slide = 50

train_x1, train_y1, val_x1, val_y1, val_y_full = get_features(length, slide)
#train_x1, train_y1, val_x1, val_y1 = dr.get_lstm_data()

model, train_x, train_y, val_x, val_y = run(train_x1, train_y1, val_x1, val_y1)
model.save('RNN_model.h5')
y_pred = test_lstm(model, val_x)
y_pred = y_pred.ravel()
real_pred = dp.expand_predictions_2(val_y_full, length, slide, y_pred)
real_pred = np.round(real_pred)
real_pred = real_pred.astype(int)
plot_300(real_pred, val_y_full)
tp,fp,tn,fn = da.get_confusion_metrics(val_y_full, real_pred)
accuracy = (tp+tn)/(tp+fp+tn+fn)
recall = tp/(tp+fn)
precision = tp/(tp+fp)
f1 = 2*tp/(2*tp+fp+fn)

print("Accuracy: " + str(accuracy))
print("Recall: " + str(recall))
print("Precision: " + str(precision))
print("F1 Score: " + str(f1))