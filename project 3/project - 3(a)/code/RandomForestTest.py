import DataReader as dr
import DataProcessor as dp
import DataAnalyzer as da
import DataManager as dm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.svm import SVC  
from sklearn.neural_network import MLPClassifier
import sys
import matplotlib.pyplot as plt

def get_accuracy_windowed(classifier, parameters, length, slide, arms, wrists, detections):
    #Assemble training data
    armIMU = arms[0]
    wristIMU = wrists[0]
    detection = detections[0]
    for i in range(1,5):
        armIMU = np.concatenate((armIMU, arms[i]), axis=0)
        wristIMU = np.concatenate((wristIMU, wrists[i]), axis=0)
        detection = np.concatenate((detection, detections[i]), axis=0)
        
    train_arm, train_wrist, train_detect = dp.apply_window(length, slide, armIMU, wristIMU, detection)
    print("Window applied to training data")
    
    #Assemble validation data
    arm, wrist, time, detect = dr.get_data(6)
    detect = detect.astype(int)
    detect = detect.ravel()
    val_arm, val_wrist, val_detect = dp.apply_window(length, slide, arm, wrist, detect)
    print("Window applied to validation data")

    train_arm_wrist = np.concatenate((train_arm, train_wrist), axis=1)
    val_arm_wrist = np.concatenate((val_arm, val_wrist), axis=1)
    
    accuracies = perform_classification(classifier, 
                                        parameters, 
                                        slide, 
                                        train_arm_wrist, 
                                        train_detect, 
                                        val_arm_wrist, 
                                        val_detect)
    
    return accuracies

def get_accuracy_features(classifier, parameters, length, slide, path):
    data = dm.load_windowed_data(length, slide)
    train_arm = data["train_arm"]
    train_wrist = data["train_wrist"]
    train_detect = data["train_detect"]
    val_arm = data["test_arm"]
    val_wrist = data["test_wrist"]
    #val_detect = data["test_detect"]

    train_arm_wrist = np.concatenate((train_arm, train_wrist), axis=1)
    val_arm_wrist = np.concatenate((val_arm, val_wrist), axis=1)
    
    _, _, _, val_detect = dr.get_data(path, 6, False)
    val_detect = val_detect.astype(int)
    val_detect = val_detect.ravel()
    
    accuracies = perform_classification(classifier, parameters, slide, train_arm_wrist, train_detect, val_arm_wrist, val_detect, length)
    
    return accuracies
   	
def perform_classification(clf, parameters, slide, train_arm_wrist, train_detect, val_arm_wrist, val_detect, length):
    accuracies = []
    parameters = parameters[clf]

    for parameter in parameters:
        #Train random forest classifier
        classifier = RandomForestClassifier(n_estimators = parameter)
        
        if clf is "knn":
            #Train k-nearest neighbors classifier
            classifier = KNeighborsClassifier(n_neighbors=parameter) 
        
        elif clf is "svm":
            #Train SVM classifier
            classifier = SVC(kernel=parameter)
            if parameter.isdigit():
                    classifier = SVC(kernel='poly', degree=int(parameter))
        
        elif clf is "mlp":
            #Train MLP classifier
            classifier = MLPClassifier(hidden_layer_sizes = parameter, max_iter = 1000)
		
        #Fit classifier to training data
        classifier.fit(train_arm_wrist, train_detect)
        
        predictions = classifier.predict(val_arm_wrist)
        predictions = dp.expand_predictions_2(val_detect, length, slide, predictions)
        
        plot_300(predictions, val_detect)
		
        errors = abs(predictions - val_detect)
        print('Mean Absolute Error: ' + str(np.mean(errors)))
        accuracies.append(1-np.mean(errors))
    return accuracies   
	
def plot_300(predict, detect):
    print('Plotting')
    x = np.arange(1, 3001)
    plt.plot(x, detect[11000:14000], label = 'detections')
    plt.plot(x, predict[11000:14000], label = 'predictions')
    plt.xlabel('time')
    plt.legend(loc='upper left', fontsize=14)
    plt.title('Plotting Predictions vs Detections', fontsize=14)
    plt.savefig('proj3_plt.png')
    plt.show()
	
def write_predictions(classifier, estimators, length, slide, sessions_count, test_data_path):
	## First combining the training and validation data
	# Loading train data
    data = dm.load_windowed_data(length, slide)
    train_arm = data["train_arm"]
    train_wrist = data["train_wrist"]
    train_detect = data["train_detect"]
	# Loading Validation data
    val_arm = data["test_arm"]
    val_wrist = data["test_wrist"]
    val_detect = data["test_detect"]
    true_val_detect = []
    index = 0
    while(index+length-1 < len(val_detect)):
        true_val_detect.append(round(np.mean(val_detect[index:(index+length)])))
        index += slide
	
    train_arm_wrist = np.concatenate((train_arm, train_wrist), axis=1)
    val_arm_wrist = np.concatenate((val_arm, val_wrist), axis=1)
    
    final_train_arm_wrist = np.concatenate((train_arm_wrist, val_arm_wrist), axis = 0)
    final_train_detect = np.concatenate((train_detect, true_val_detect), axis = 0)
	
	## Specify Classifier and its parameters
    classifier = RandomForestClassifier(n_estimators = estimators)
	## Fit classifier to training data
    classifier.fit(final_train_arm_wrist, final_train_detect)
	## Make predictions on the test data
    for i in range(sessions_count):
        arms, wrists, _ = dr.get_data(test_data_path, i+1, True)
        test_arm = da.apply_window_features(arms, length, slide, 50, True)
        test_wrist = da.apply_window_features(wrists, length, slide, 50, True)
        test_arm_wrist = np.concatenate((test_arm, test_wrist), axis=1)
        predictions = classifier.predict(test_arm_wrist)
        final_predictions = dp.expand_predictions_2(arms, length, slide, predictions, True)
		 
        print("Length of Predictions for Session " + str(i+1) + ": " + str(len(final_predictions)))
		
		## Writing the predictions to a file
        with open(test_data_path + "Session0" + str(i+1) + "/" + "prediction" +  ".txt", "w") as mf:
            for val in final_predictions:
                mf.write("%s\n" % val)

test_param = (sys.argv[1])
if test_param == 'False':
	data_path = 'C:/Users/sshek/Documents/My Documents/Spring 2019 Courses/Deep Learning and Neural Nets/Projects/Project 3(a)/ECE542Project3A-master/AllData/'
	#arms, wrists, detections = dr.get_all_data(data_path, False)
	lengths = [100, 150, 200, 250, 300, 350, 400]
	#slides = [0.1, 0.25]
	rf_estimators = [100, 150, 200, 250, 300, 350, 400]
	knn_neighbors = [50, 100, 150, 200, 300, 400, 500]
	svm_kernels = ['linear', 'rbf', 'sigmoid']
	mlp_hidden_layers = [2*(100,), 3*(100,), 5*(100,), 10*(100,)] 
	svm_polys = [2,3,4,5,6,7,8,9,10,15,20]
	for poly in svm_polys:
		svm_kernels.append(str(poly))
	classifiers = ['rf', 'knn', 'svm']

	parameters = {'rf':rf_estimators,
				  'knn':knn_neighbors,
				  'mlp': mlp_hidden_layers,
				  'svm':svm_kernels}

	accuracies = []
	for length in lengths:
		slide = 50
		print('Slide is: ', slide)
		acc_row = get_accuracy_features("knn", parameters, length, slide)
		accuracies.append(acc_row)
	print(accuracies)
	acc = np.array(accuracies)
	
	# ## Final Model
	# rf_estimators = [150]
	# get_accuracy_features("rf", parameters, 300, 50, data_path)
else:
	test_data_path = "C:/Users/sshek/Documents/My Documents/Spring 2019 Courses/Deep Learning and Neural Nets/Projects/Project 3(a)/ECE542Project3A-master/Test Data/"
	## Final Testing Model [Best Model]
	### Random Forest (Slide -> 50, window -> 300, estimators -> 150)
	#test_accuracy = 
	write_predictions("rf", 150, 300, 50, 4, test_data_path)
