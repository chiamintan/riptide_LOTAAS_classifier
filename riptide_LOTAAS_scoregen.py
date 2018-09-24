import argparse
import numpy as np
import os
import sys
from DataProcessor import DataProcessor

class riptideLOTAASscoregen:

    def main(self, argv=None):

        print ''
        print 'riptide candidates classifier score generator.'
        print 'The script calculates the features from the .h5 candidate plots and write them down in arff format.'
        print 'Features based on Lyon et al 2016 and Tan et al 2018 are used on the profile, subints, DM curve and period plot'
        print ''

        #Input arguments for the score generator.
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', type=str, dest='dir', nargs=1, default='', help='directory where the h5 files are', required=True)
        parser.add_argument('-f', type=str, dest='out', nargs=1, default='', help='path to the file to store the arff file', required=True)
        parser.add_argument('-t', type=int, dest='feature_type', nargs=1, default=0, help='the set of features to use', required=True)
        args = parser.parse_args()

        self.dir = args.dir[0]
        self.out = args.out[0]
        self.feature_type = args.feature_type[0]

        #Check for input directory.
        if (os.path.isdir(self.dir)):
            pass
        else:
            print 'Input directory does not exists'
            sys.exit()

        #Check for output file. If it doesn't exists, create a new one.
        if (os.path.isfile(self.out)):
            open(self.out, 'w').close()
        else:
            try:
                output = open(self.out, 'w')  # First try to create file.
                output.close()
            except IOError:
                pass
            if (not os.path.isfile(self.out)):
                print 'output file does not exists'
                sys.exit()

        if (self.feature_type < 1 or self.feature_type > 3):
            print 'feature type out of range'
            sys.exit()

        #Write out the format for arff files
        outputFile = open(self.out, 'a')
        outputFile.write("@relation Feature_Type_"+str(self.feature_type)+"\n")
        if self.feature_type == 1:
            print 'Standard features used'
            for i in range(1,21):
                outputFile.write("@attribute Feature_"+str(i)+" numeric"+"\n")
        elif self.feature_type == 2:
            print 'Profile and subints are gated at 25% around maximum'
            for i in range(1,21):
                outputFile.write("@attribute Feature_"+str(i)+" numeric"+"\n")
        elif self.feature_type == 3:
            print 'Profile and subints are rebinned to 256 bins'
            for i in range(1,21):
                outputFile.write("@attribute Feature_"+str(i)+" numeric"+"\n")


        outputFile.write("@attribute class {0,1,2}"+"\n")
        outputFile.write("@data"+"\n")
        outputFile.close()

        #Let's get our features
        dp = DataProcessor()
        dp.process(self.dir, self.out, self.feature_type)

if __name__ == '__main__':
    riptideLOTAASscoregen().main()