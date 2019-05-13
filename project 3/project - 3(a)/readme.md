Project3a - Body-Rocking Behavior Recognition

Wearable sensors have been shown to be effective for promoting self-awareness, wellness and re-education. For this project, you will develop a detector for Body-rocking behavior from a blind subject (also commonly observed in individuals with autism) using inertial measurements from a wearable system. Two sensors, one of the wrist and another on the arm (as illustrated on the right), were used to record motion using accelerometers and gyroscopes. The data will consists of sessions about 1-2 hours long each with annotations of when the behavior was observed. Your goal will be to train a classifier for the detections of these events.
This project will have two parts: In the first phase, you are welcome to try any machine learning approach for detection (e.g.,
Random Forest, SVM, MLP, etc). This will serve as your baseline for
the second part of the project. We will provide you a test set for which you will turn in predictions in a format to be specified soon. The results from all the teams will be shared to the class in order to give you an idea of what to expect for performance. For the second phase, you will be expected to implement a recurring neural network (e.g., LSTM or CNN-LSTM) for learning features from the data and improving on your results from the first part of this project. The groups with the best performance on this second phase will receive extra credit.
Deliverables:
1.	Predictions for the provided test set
2.	Technical report
For your report, you should follow the following guidelines:
1.	The length should not be more than 3 pages long. No introduction or abstract are needed. Your report should have three sections.
2.	Section 1 - Methodology: Include a description of your approach including citations to papers describing the methodology, and references to toolboxes (and specific functions) used for the implementation.
3.	Section 2 – Model Training and Hyperparameter Selection: Provide details about your procedure for hyper-parameter tuning.
4.	Section 3 – Evaluation: Perform an evaluation of your method including error metrics. We will be giving you a test set for which you will give us your prediction for evaluation.
5.	Include several plots showing results of your inference and the groundtruth.
Remember that you are welcome try some complex models for prediction, but this first round of the project is only meant to provide a baseline so classical machine learning techniques are acceptable.
 
Some References:
•	L. Sadouk, T. Gadi, and E. H. Essoufi, “A novel deep learning approach for recognizing stereotypical motor movements within and across subjects on the autism spectrum disorder,” in Comp. Int. and Neurosc., 2018.
•	N. M. Rad, S. M. Kia, C. Zarbo, T. van Laarhoven, G. Jurman, P. Venuti, E. Marchiori, and C. Furlanello, “Deep learning for automatic stereotypical motor movement detection using wearable sensors in autism spectrum disorders,”SignalProcessing, vol. 144, pp. 180–191, 2018.
•	U. Grossekathofer, N. V. Manyakov, V. Mihajlovic, G. Pandina, A. Skalkin, S. Ness, A. Bangerter, and M. S. Goodwin, “Automated detection of stereotypical motor movements in autism spectrum disorder using recurrence quantification analysis,” in Front. Neuroinform., 2017
•	M. S. Goodwin, M. Haghighi, Q. Tang, M. Akakaya, D. Erdogmus, and S. S. Intille, “Moving towards a real-time system for automatically recognizing stereotypical motor movements in individuals on the autism spectrum using wireless accelerometry,” in Ubicomp, 2014.