from riptide.pipelines import Candidate
import numpy as np
import datetime, os, sys, fnmatch
from FeatureExtractor import FeatureExtractor


class DataProcessor:

    def __init__(self):

        self.h5Regex = "*.h5"
        self.FeatureStore = []

    def storeFeature(self,features,candname):

        allFeatures = str(",".join(map(str, features)))
        entry1 = allFeatures + ",%" + candname
        entry2 = entry1.replace("nan", "0")  # Remove NaNs since these cause error for ML tools like WEKA
        entry3 = entry2.replace("inf", "0")  # Remove infinity values since these cause error for ML tools like WEKA
        self.FeatureStore.append(entry3)

    def process(self, directory, output, feature_type):

        start = datetime.datetime.now()

        for root, subFolders, filenames in os.walk(directory):

            for filename in fnmatch.filter(filenames, self.h5Regex):

                candfile = os.path.join(root, filename)
                cand = Candidate.load_hdf5(str(candfile))

                fe = FeatureExtractor()

                features = fe.getfeatures(cand, feature_type)
                features.append("?")

                self.storeFeature(features,candfile)

        outputText = ""

        for f in self.FeatureStore:

            outputText += f + "\n"

        outputFile = open(output, 'a')
        outputFile.write(str(outputText))
        outputFile.close()

        end = datetime.datetime.now()
        print 'Processing time = ',str(end-start)


