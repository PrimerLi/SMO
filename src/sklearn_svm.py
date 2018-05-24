import pandas as pd 
import numpy as np
from sklearn import svm

'''
    Confusion matrix:
                        Actual N  |   Actual P
     Prediction N         TN      |      FN
     ------------------------------------------------
     Prediction P         FP      |     TP
'''
def generateConfusionMatrix(prediction, trueLabel):
    assert(len(prediction) == len(trueLabel))
    labelSet = set(trueLabel)
    negativeLabel = 0
    positiveLabel = 1
    assert(negativeLabel in labelSet)
    assert(positiveLabel in labelSet)
    assert(len(labelSet) == 2)
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0
    result = np.zeros((2,2))
    for i in range(len(prediction)):
        if (prediction[i] == 0): # prediction is negative
            if (trueLabel[i] == negativeLabel):
                trueNegative += 1
            else:
                falseNegative += 1
        else: # positive prediction
            if (trueLabel[i] == negativeLabel):
                falsePositive += 1
            else:
                truePositive += 1
    result[0, 0] = trueNegative
    result[0, 1] = falseNegative
    result[1, 0] = falsePositive
    result[1, 1] = truePositive
    return result

def to_string(confusionMatrix):
    return "\n".join(map(lambda row: "  ".join(map(str, row)), confusionMatrix))

def getPrecision(confusionMatrix):
    (row, col) = confusionMatrix.shape
    assert(row == 2 and col == 2)
    truePositive = confusionMatrix[1,1]
    falsePositive = confusionMatrix[1,0]
    eps = 1.0e-10
    return float(truePositive + eps)/(truePositive + falsePositive + eps)

def getRecall(confusionMatrix):
    (row, col) = confusionMatrix.shape
    assert(row == 2 and col == 2)
    truePositive = confusionMatrix[1,1]
    falseNegative = confusionMatrix[0,1]
    eps = 1.0e-10
    return float(truePositive + eps)/float(truePositive + falseNegative + eps)

def getFPR(confusionMatrix): # get the fasle positive rate
    (row, col) = confusionMatrix.shape
    assert(row == 2 and col == 2)
    trueNegative = confusionMatrix[0,0]
    falsePositive = confusionMatrix[1,0]
    eps = 1.0e-10
    return float(falsePositive + eps)/float(falsePositive + trueNegative + eps)

def get_readable_results(confusionMatrix):
    (row, col) = confusionMatrix.shape
    assert(row == col and row == 2)
    true_positive = confusionMatrix[1,1]
    false_positive = confusionMatrix[1,0]
    false_negative = confusionMatrix[0,1]
    eps = 1.0e-10
    capture_rate = (true_positive + eps)/(true_positive + false_negative + eps)
    incorrect_slay_rate = (false_positive + eps)/(false_positive + true_positive + eps)
    print "confusion matrix: "
    print to_string(confusionMatrix)
    print "Capture rate: ", capture_rate
    print "Incorrect slay rate: ", incorrect_slay_rate
    return capture_rate, incorrect_slay_rate

def main():
    import os
    import sys

    trainFileName = "train.csv"
    testFileName = "test.csv"
    assert(os.path.exists(trainFileName))
    assert(os.path.exists(testFileName))
    train_df = pd.read_csv(trainFileName)
    test_df = pd.read_csv(testFileName)
    keys = train_df.keys()
    features = keys[0:-1]
    label = keys[-1]
    X_train = train_df[features]
    y_train = train_df[label]
    X_test = test_df[features]
    y_test = test_df[label]
    print "Training the model ... "
    C = 1.0
    svc = svm.SVC(kernel='linear', C=C, decision_function_shape='ovr').fit(X_train, y_train)
    print "Model training finished. "
    print "Making predictions ... "
    predictions = svc.predict(X_test)
    print "Predictions made. "
    labels = y_test
    ofile = open("predictions_labels.csv", "w")
    ofile.write("prediction,label\n")
    for i in range(len(predictions)):
        ofile.write(str(predictions[i]) + "," + str(labels[i]) + "\n")
    ofile.close()

    confusionMatrix = generateConfusionMatrix(predictions, labels)
    get_readable_results(confusionMatrix)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
