import numpy as np

def apply_window(length, slide, arm, wrist, detection, test = False):
    index = 0
    new_arm = []
    new_wrist = []
    new_detect = []
    
    while True:
        arm_row = arm[index]
        wrist_row = wrist[index]
        for i in range(1,length):
            arm_row = np.concatenate((arm_row, arm[index+i]))
            wrist_row = np.concatenate((wrist_row, wrist[index+i]))
        new_arm.append(arm_row)
        new_wrist.append(wrist_row)
        if not test:
            new_detect.append(round(np.mean(detection[index:(index+length)])))
        index += slide
        if index+length-1 >= len(arm):
            break
    new_arm = np.array(new_arm)
    new_wrist = np.array(new_wrist)
    if not test:
        new_detect = np.array(new_detect)
    if not test:
        return new_arm, new_wrist, new_detect
    else:
        return new_arm, new_wrist

def expand_predictions(old_detection, slide, pred):
    new_pred = []
    for i in range(len(pred)):
        val = pred[i]
        for j in range(slide):
            new_pred.append(val)
    remainder = len(old_detection) - len(new_pred)
    for i in range(remainder):
        new_pred.append(pred[-1])
    return np.array(new_pred)

def expand_predictions_2(old_detection, length, slide, pred, test = False):
    new_pred = []

    #Initialize with first (length-slide) values to predictions are added to the end of the window
    val = pred[0]
    for i in range(length-slide):
        new_pred.append(val)
    if test:
        for i in range(len(pred)-1):
            val = pred[i]
            for j in range(slide):
                new_pred.append(val)
        remainder = len(old_detection) - len(new_pred)
        for i in range(remainder):
            new_pred.append(pred[-1])
    else:
        for i in range(len(pred)):
            val = pred[i]
            for j in range(slide):
                new_pred.append(val)
        remainder = len(old_detection) - len(new_pred)
        for i in range(remainder):
            new_pred.append(pred[-1])
			
    return np.array(new_pred)