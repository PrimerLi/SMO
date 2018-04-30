import pandas as pd
import os

def getDF(inputFileName):
    assert(os.path.exists(inputFileName))
    df = pd.read_csv(inputFileName)
    return df

def convert_to_svm_label(label):
    if (label > 0):
        return 1
    else:
        return -1

def change_label(inputFileName, outputFileName):
    df = getDF(inputFileName)
    assert("label" in df.keys())
    labels = df['label']
    df["label"] = map(convert_to_svm_label, labels)
    df.to_csv(outputFileName, index = False)

def main():
    import sys
    if (len(sys.argv) != 2):
        print "inputFileName = sys.argv[1]. "
        return -1

    inputFileName = sys.argv[1]
    change_label(inputFileName, inputFileName.replace(".csv", "_svm.csv"))
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
