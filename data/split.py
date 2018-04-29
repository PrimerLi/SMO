def split(inputFileName, trainRatio):
    import os
    assert(os.path.exists(inputFileName))
    assert(trainRatio > 0 and trainRatio < 1)

    lines = []
    ifile = open(inputFileName, "r")
    for (index, string) in enumerate(ifile):
        if (index == 0):
            header = string.strip("\n")
        else:
            lines.append(string.strip("\n"))
    ifile.close()

    trainNumber = int(trainRatio*len(lines))

    ofile = open("train.csv", "w")
    ofile.write(header + "\n")
    for i in range(trainNumber):
        ofile.write(lines[i] + "\n")
    ofile.close()

    ofile = open("test.csv", "w")
    ofile.write(header + "\n")
    for i in range(trainNumber, len(lines)):
        ofile.write(lines[i] + "\n")
    ofile.close()

def main():
    import sys

    if (len(sys.argv) != 3):
        print "inputFileName = sys.argv[1], trainRatio = sys.argv[2]. "
        return -1

    inputFileName = sys.argv[1]
    trainRatio = float(sys.argv[2])
    split(inputFileName, trainRatio)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
