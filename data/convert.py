def convert(inputFileName, outputFileName):
    import os
    assert(os.path.exists(inputFileName))
    lines = []
    ifile = open(inputFileName, "r")
    for (index, string) in enumerate(ifile):
        lines.append(string.strip("\n"))
    ifile.close()
    ofile = open(outputFileName, "w")
    for i in range(len(lines)):
        ofile.write("  ".join(lines[i].split(",")[0:-1]) + "\n")
    ofile.close()

def main():
    import os
    import sys
    if (len(sys.argv) != 3):
        print "inputFileName = sys.argv[1], outputFileName = sys.argv[2]. "
        return -1
    
    inputFileName = sys.argv[1]
    outputFileName = sys.argv[2]
    convert(inputFileName, outputFileName)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
