#!/usr/bin/env python

import random

class Point:
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label
    def __str__(self):
        return ",".join(map(lambda ele: str(ele), [self.x, self.y, self.label]))

def printPoints(points, outputFileName):
    ofile = open(outputFileName, "w")
    for i in range(len(points)):
        ofile.write(points[i].__str__() + "\n")
    ofile.close()

def shuffle(input_list):
    size = len(input_list)
    for i in range(len(input_list)):
        random_index = random.randint(0, size-1)
        temp = input_list[i]
        input_list[i] = input_list[random_index]
        input_list[random_index] = temp
    return input_list

def main():
    import os
    import sys

    if (len(sys.argv) != 3):
        print "n = sys.argv[1], p = sys.argv[2]. "
        return -1

    negativePoints = []
    positivePoints = []
    numberOfNegatives = int(sys.argv[1])
    numberOfPositives = int(sys.argv[2])

    for i in range(numberOfNegatives):
        randomX = random.uniform(-1, 1.5)
        randomY = random.uniform(-1, 1.5)
        point = Point(randomX, randomY, -1)
        negativePoints.append(point)
    for i in range(numberOfPositives):
        randomX = 3 + random.uniform(-1.1, 2)
        randomY = 3 + random.uniform(-1.1, 1)
        point = Point(randomX, randomY, 1)
        positivePoints.append(point)

    printPoints(negativePoints, "negativePoints.txt")
    printPoints(positivePoints, "positivePoints.txt")
    points = negativePoints + positivePoints
    points = shuffle(points)
    ofile = open("data.csv", "w")
    ofile.write("x,y,label\n")
    for i in range(len(points)):
        ofile.write(str(points[i]) + "\n")
    ofile.close()

    #os.system("./pltfiles.py negativePoints.txt positivePoints.txt")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
