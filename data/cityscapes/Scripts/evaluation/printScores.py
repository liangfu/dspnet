#!/usr/bin/env python

import json
import sys
sys.path.insert(0,"../../../../dataset")
from cs_labels import labels

def printScores():
    jsonfile = "../../evaluationResults/resultPixelLevelSemanticLabeling.json"
    data = None
    with open(jsonfile) as fp:
        data = json.load(fp)
    D = data["classScores"]
    # print labels
    # print D
    for label in labels:
        if label.trainId>=0 and label.trainId<255:
            print "%s &" % (label.name,),
    print "mAP"
    for label in labels:
        if label.trainId>=0 and label.trainId<255:
            print "%.1f &" % (D[label.name]*100.,),
    print "%.1f" % (data["averageScoreClasses"]*100.,)

if __name__=="__main__":
    printScores()
