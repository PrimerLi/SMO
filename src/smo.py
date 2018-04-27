import numpy as np
import os
import random

class Sample:
    def __init__(self, x, label):
        self.x = x
        self.label = label
    def __str__(self):
        return ",".join(map(lambda ele: str(ele), self.x)) + "," + str(label)

class DataSet:
    def __init__(self, samples):
        self.samples = samples
        self.numberOfSamples = len(samples)
    def __str__(self):
        return "Number of samples: " + str(self.numberOfSamples)

def sign(x):
    if (x > 0):
        return 1
    else:
        return -1

def to_sample(line):
    tempArray = line.split(",")
    x = map(lambda ele: float(ele), tempArray[0:-1])
    label = sign(int(tempArray[-1]))
    return Sample(x, label)

def read(inputFileName):
    import os
    assert(os.path.exists(inputFileName))
    ifile = open(inputFileName, "r")
    lines = []
    for (index, string) in enumerate(ifile):
        if (index == 0):
            header = string.strip("\n")
        else:
            lines.append(string.strip("\n"))
    ifile.close()
    samples = map(lambda line: to_sample(line), lines)
    dataSet = DataSet(samples)
    return dataSet

def is_equal(alpha_old, alpha_new):
    assert(len(alpha_old) == len(alpha_new))
    eps = 1.0e-5
    return np.linalg.norm(alpha_old - alpha_new) < eps

class Interval:
    def __init__(self, left, right):
        assert(left <= right)
        self.left = left
        self.right = right
    def __str__(self):
        return "[" + str(self.left) + ", " + str(self.right) + "]"

def within_interval(number, interval):
    if (number < interval.left or number > interval.right):
        return False
    else:
        return True

def get_alpha_1_limit(alpha_1, alpha_2, y_1, y_2, C):
    assert(C > 0)
    if (y_1*y_2 == -1):
        k = alpha_1 - alpha_2
        assert(True or within_interval(k, Interval(-C, C)))
        return Interval(max(0, k), C + min(0, k))
    elif(y_1*y_2 == 1):
        k = alpha_1 + alpha_2
        assert(True or within_interval(k, Interval(0, 2*C)))
        return Interval(max(0, k-C), min(C, k))

def get_alpha_2_limit(alpha_1, alpha_2, y_1, y_2, C):
    assert(C > 0)
    if (y_1*y_2 == -1):
        k = alpha_1 - alpha_2
        assert(within_interval(k, Interval(-C, C)))
        return Interval(max(-k, 0), C + min(-k, 0))
    elif(y_1*y_2 == 1):
        k = alpha_1 + alpha_2
        assert(within_interval(k, Interval(0, 2*C)))
        return Interval(max(0, k-C), min(C, k))
    else:
        print "Label error. "
        sys.exit(-1)

def clip(alpha, interval): 
    if (within_interval(alpha, interval)):
        return alpha
    elif(alpha < interval.left):
        return interval.left
    else:
        return interval.right

verbose = 2

def on_boundary(alpha, C):
    assert(C > 0)
    eps = 1.0e-6
    if (alpha > eps and C - alpha > eps):
        return True
    return False

def get_lines(inputFileName):
    assert(os.path.exists(inputFileName))
    ifile = open(inputFileName, "r")
    lines = []
    for (index, string) in enumerate(ifile):
        if (index == 0):
            header = string.strip("\n")
        else:
            lines.append(string.strip("\n"))
    ifile.close()
    return header, lines

'''
    Confusion matrix:
                        Actual N  |   Actual P
     Prediction N         TN      |      FN
     ------------------------------------------------
     Prediction P         FP      |     TP
'''            
def get_confusion_matrix(predictions, labels):
    assert(len(predictions) == len(labels))
    true_negative = 0.0
    false_negative = 0.0
    false_positive = 0.0
    true_positive = 0.0
    for i in range(len(predictions)):
        if (predictions[i] > 0):
            if(labels[i] > 0):
                true_positive += 1.
            else:
                false_positive += 1.
        else:
            if (labels[i] > 0):
                 false_negative += 1.
            else:
                 true_negative += 1.
    print true_negative, false_negative
    print false_positive, true_positive
    ofile = open("confusion_matrix.txt", "w")
    ofile.write(str(true_negative) + "  " + str(false_negative) + "\n")
    ofile.write(str(false_positive) + "  " + str(true_positive))
    ofile.close()
    eps = 1.0e-10
    print "Capture-rate = ", (true_positive + eps)/(true_positive + false_positive + eps)
    print "Incorrect slay rate = ", (false_positive + eps)/(false_positive + true_positive) 

class SVM:
    def __init__(self, inputFileName, C):#inputFileName contains the train data. 
        assert(os.path.exists(inputFileName))
        dataSet = read(inputFileName)
        self.X = map(lambda sample: np.asarray(sample.x), dataSet.samples)
        self.y = np.asarray(map(lambda sample: sample.label, dataSet.samples))
        self.numberOfFeatures = len(self.X[0])
        self.numberOfSamples = len(self.y)
        self.alpha = np.zeros(self.numberOfSamples)
        self.C = C
        self.beta = np.zeros(self.numberOfFeatures)
        self.beta_0 = 0.0
    def get_beta(self):
        self.beta = np.zeros(self.numberOfFeatures)
        for i in range(self.numberOfSamples):
            self.beta += self.alpha[i]*self.y[i]*self.X[i]
        return self.beta
    def sweep(self):
        eps = 1.0e-16
        for first_index in range(self.numberOfSamples):
            for second_index in range(self.numberOfSamples):
                i = random.randint(0, self.numberOfSamples-1)
                j = random.randint(0, self.numberOfSamples-1)
                if (i == j):
                    continue
                alpha_1 = self.alpha[i]
                alpha_2 = self.alpha[j]
                x_1 = self.X[i]
                x_2 = self.X[j]
                y_1 = self.y[i]
                y_2 = self.y[j]
                s = y_1*y_2
                interval_2 = get_alpha_2_limit(alpha_1, alpha_2, y_1, y_2, self.C)
                self.beta = self.get_beta()
                alpha_2_star = alpha_2 + y_2*((self.beta.dot(x_1) - y_1) - (self.beta.dot(x_2) - y_2))/np.dot(x_1 - x_2, x_1 - x_2)
                if (within_interval(alpha_2_star, interval_2)):
                    alpha_2_new = alpha_2_star
                elif(alpha_2_star < interval_2.left):
                    alpha_2_new = interval_2.left
                else:
                    alpha_2_new = interval_2.right
                alpha_1_new = alpha_1 - s*(alpha_2_new - alpha_2)
                assert(abs(alpha_1_new + s*alpha_2_new - (alpha_1 + s*alpha_2)) < eps)
                self.alpha[i] = alpha_1_new
                self.alpha[j] = alpha_2_new
                if(verbose >= 4):
                    print "alpha_1_old: ", alpha_1, "alpha_1_new: ", alpha_1_new
                    print "alpha_2_old: ", alpha_2, "alpha_2_new: ", alpha_2_new
    def train(self):
        iterationMax = 1000
        eps = 1.0e-10
        for i in range(iterationMax):
            alpha_old = np.asarray(map(lambda ele: ele, self.alpha))
            self.sweep()
            alpha_new = np.asarray(map(lambda ele: ele, self.alpha))
            error = np.linalg.norm(alpha_new - alpha_old)
            print "i = ", i+1, ", total = ", iterationMax, ", error = ", error
            if (verbose >= 3):
                print "alpha_old: "
                print alpha_old
                print "alpha_new:"
                print alpha_new
            if (error < eps):
                break
        self.beta = self.get_beta()
        assert(abs(self.alpha.dot(self.y)) < eps)
        if (verbose >= 2):
            print "alpha:"
            print self.alpha
            print "beta: "
            print self.beta
        boundary_indices = []
        for i in range(len(self.alpha)):
            if (on_boundary(self.alpha[i], self.C)):
                boundary_indices.append(i)
        beta_0_values = []
        for index in boundary_indices:
            beta_0_values.append(self.y[index] - self.beta.dot(self.X[index]))
        if (verbose >= 2):
            print "beta_0 values: "
            print beta_0_values
        self.beta_0 = np.mean(np.asarray(beta_0_values))
        if (verbose >= 2):
            print "beta_0 = ", self.beta_0
    def test(self, testFileName):
        assert(os.path.exists(testFileName))
        header, lines = get_lines(testFileName)
        labels = map(lambda line: int(line.split(",")[-1]), lines)
        feature_vectors = map(lambda line: np.asarray(map(lambda x: float(x), line.split(",")[0:-1])), lines)
        predictions = map(lambda vector: self.beta.dot(vector) + self.beta_0, feature_vectors)
        ofile = open("predictions_labels.csv", "w")
        ofile.write("prediction,label\n")
        for i in range(len(predictions)):
            ofile.write(str(predictions[i]) + "," + str(labels[i]) + "\n")
        ofile.close()
        get_confusion_matrix(predictions, labels)

def print_file(header, lines, outputFileName):
    ofile = open(outputFileName, "w")
    ofile.write(header + "\n")
    for i in range(len(lines)):
        ofile.write(lines[i] + "\n")
    ofile.close()

def cross_validation(inputFileName, trainRatio, C):
    assert(C > 0)
    assert(os.path.exists(inputFileName))
    assert(trainRatio > 0 and trainRatio < 1)
    header, lines = get_lines(inputFileName)
    trainNumber = int(len(lines)*trainRatio)
    print_file(header, lines[0:trainNumber], "train.csv")
    print_file(header, lines[trainNumber:], "test.csv")
    svm = SVM("train.csv", C)
    svm.train()
    svm.test("test.csv")

def main():
    import sys
    if (len(sys.argv) != 3):
        print "inputFileName = sys.argv[1], trainRatio = sys.argv[2]. "
        return -1

    C = 1000
    inputFileName = sys.argv[1]
    trainRatio = float(sys.argv[2])
    cross_validation(inputFileName, trainRatio, C)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
