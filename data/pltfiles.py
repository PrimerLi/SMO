#!/usr/bin/env python

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def readFile(fileName):
    import sys
    try:
        ifile = open(fileName, "r")
    except:
        sys.exit(fileName + " does not exist. ")
    x = []
    y = []
    for (index, string) in enumerate(ifile):
        string = string.replace(",", " ")
        a = string.split()
        if (is_float(a[0])):
            x.append(float(a[0]))
            y.append(float(a[1]))
        else:
            continue
    return x, y

def main():
    import sys
    import os
    import matplotlib.pyplot as plt

    if (len(sys.argv) == 1):
        print "Input file name as arguments. "
        return -1
    else:
        colors = ["bo", 'ro', 'co', 'mo', 'go']
        if (len(sys.argv) > len(colors)+1):
            print "Too many files to plot. "
            return -1
        
        fileNames = []
        for i in range(1, len(sys.argv)):
            fileNames.append(sys.argv[i])
        for i in range(len(fileNames)):
            if (not os.path.exists(fileNames[i])):
                print fileNames[i] + " does not exist."
                return -1
        
        xLists = []
        yLists = []
        for i in range(len(fileNames)):
            x, y = readFile(fileNames[i])
            xLists.append(x)
            yLists.append(y)

        xmin = min(xLists[0])
        xmax = max(xLists[0])
        ymin = min(yLists[0])
        ymax = max(yLists[0])

        for i in range(len(xLists)):
            min_temp = min(xLists[i])
            max_temp = max(xLists[i])
            if (xmin > min_temp):
                xmin = min_temp
            if (xmax < max_temp):
                xmax = max_temp
        for i in range(len(yLists)):
            min_temp = min(yLists[i])
            max_temp = max(yLists[i])
            if (ymin > min_temp):
                ymin = min_temp
            if (ymax < max_temp):
                ymax = max_temp

        xdiff = xmax - xmin
        ydiff = ymax - ymin
        ymin = ymin - 0.1*ydiff
        ymax = ymax + 0.1*ydiff
        xmin = xmin - 0.05*xdiff
        xmax = xmax + 0.05*xdiff

        if (len(xLists) == 1):
            plt.plot(xLists[0], yLists[0], colors[0])
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.grid()
            plt.show()
        elif(len(xLists) == 2):
            plt.plot(xLists[0], yLists[0], colors[0], xLists[1], yLists[1], colors[1])
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.grid()
            plt.show()
        elif(len(xLists) == 3):
            plt.plot(xLists[0], yLists[0], colors[0], xLists[1], yLists[1], colors[1], xLists[2], yLists[2], colors[2])
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.grid()
            plt.show()
        elif(len(xLists) == 4):
            plt.plot(xLists[0], yLists[0], colors[0], xLists[1], yLists[1], colors[1], xLists[2], yLists[2], colors[2], xLists[3], yLists[3], colors[3])
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.grid()
            plt.show()
        elif(len(xLists) == 5):
            plt.plot(xLists[0], yLists[0], colors[0], xLists[1], yLists[1], colors[1], xLists[2], yLists[2], colors[2], xLists[3], yLists[3], colors[3], xLists[4], yLists[4], colors[4])
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.grid()
            plt.show()

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
