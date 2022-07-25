#! /usr/bin/env python3

"""
Development Version: Python 3.5.1
Author: Benjamin Cordier
Description: Module For Performance 
Assessment of Classification Task
License: BSD 3 Clause
--
Copyright 2018 Benjamin Cordier
Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, 
this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, 
this list of conditions and the following disclaimer in the documentation 
and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its 
contributors may be used to endorse or promote products derived from this 
software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
"""

# Import Modules
import math
from collections import OrderedDict

class Performance(object):

    #
    # Metric Function Definitions
    #

    __metrics        = {
        "statistics"    : {
            "accuracy"      : lambda tp, tn, fp, fn: (tp + tn) / (tp + tn + fp + fn) if (tp + tn) > 0 else 0.0,
            "f1score"       : lambda tp, tn, fp, fn: (2 * tp) / ((2 * tp) + (fp + fn)) if tp > 0 else 0.0,
            "sensitivity"   : lambda tp, tn, fp, fn: tp / (tp + fn) if tp > 0 else 0.0,
            "specificity"   : lambda tp, tn, fp, fn: tn / (tn + fp) if tn > 0 else 0.0,
            "precision"     : lambda tp, tn, fp, fn: tp / (tp + fp) if tp > 0 else 0.0,
            "recall"        : lambda tp, tn, fp, fn: tp / (tp + fn) if tp > 0 else 0.0,
            "tpr"           : lambda tp, tn, fp, fn: tp / (tp + fn) if tp > 0 else 0.0,
            "tnr"           : lambda tp, tn, fp, fn: tn / (tn + fp) if tn > 0 else 0.0,
            "fpr"           : lambda tp, tn, fp, fn: fp / (fp + tn) if fp > 0 else 0.0,
            "fnr"           : lambda tp, tn, fp, fn: fn / (fn + tp) if fn > 0 else 0.0,
            "ppv"           : lambda tp, tn, fp, fn: tp / (tp + fp) if tp > 0 else 0.0,
            "npv"           : lambda tp, tn, fp, fn: tn / (tn + fn) if tn > 0 else 0.0,
            "mcc"           : lambda tp, tn, fp, fn: ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp * tn * fn * fp) > 0 else 0.0,
            "j-statistic"   : lambda tp, tn, fp, fn: (tp / (tp + fn)) + (tn / (tn + fp)) - 1 if (tp > 0 or (tp + fn) > 0) or (tn > 0 or (tn + fp) > 0) else -1.0
        },
        "counts"        : {
            "tp"            : lambda tp, tn, fp, fn: tp,
            "tn"            : lambda tp, tn, fp, fn: tn,
            "fp"            : lambda tp, tn, fp, fn: fp,
            "fn"            : lambda tp, tn, fp, fn: fn,
            "pos"           : lambda tp, tn, fp, fn: tp + fn,
            "neg"           : lambda tp, tn, fp, fn: tn + fp,
            "prop"          : lambda tp, tn, fp, fn: (tp + fn) / (tp + tn + fp + fn)
        }
    }

    #
    # Initialization
    #

    def __init__(self, actual = [], predicted = [], __metrics = __metrics):
        # Set Reference to Defaults
        self.metrics   = __metrics
        self.actual    = []
        self.predicted = []
        # Call Update Function
        self.update(actual, predicted)

    #
    # Update Function
    #

    def update (self, actual = [], predicted = []):
        # Type Check Inputs, Allow For Update of Actual and Predicted Simultaneously or Individually
        self.actual         = actual        if type(actual)     is list and len(actual)    > 0    else self.actual
        self.predicted      = predicted     if type(predicted)  is list and len(predicted) > 0    else self.predicted
        assert len(self.actual) == len(self.predicted), "Actual and predicted lists must be equal in length"
        assert len(self.actual) > 0, "Actual and predicted lists should have a length greater than 0"
        # Additional References
        self.classes        = sorted(set(self.actual))
        self.matrix         = [ [ 0 for _ in self.classes ] for _ in self.classes ]
        self.classToIndex   = { key: i for i, key in enumerate(self.classes) }
        self.indexToClass   = { i: key for i, key in enumerate(self.classes) }
        # Generate Confusion Matrix
        for p, a in zip(self.predicted, self.actual):
            self.matrix[self.classToIndex[p]][self.classToIndex[a]] += 1
        # Matrix Sum
        self.n              = sum([ sum(row) for row in self.matrix ]) 
        # Matrix as Proportions (Normalized)
        self.normed         = [ row for row in map(lambda i: list(map(lambda j: j / self.n, i)), self.matrix) ]
        # Generate Statistics Data Structure
        self.results        = OrderedDict(((c, {"counts" : OrderedDict(), "stats" : OrderedDict()}) for c in self.classes))
        # Compute Counts & Statistics
        for i in range(len(self.classes)):
            row = sum(self.matrix[i][:])
            col = sum([row[i] for row in self.matrix]) # Can't Access Matrix Col Using matrix[:][i] With Vanilla Python
            tp  = self.matrix[i][i]
            fp  = row - tp
            fn  = col - tp
            tn  = self.n - row - col + tp
            # Populate Counts Dictionary
            for count, func in self.metrics["counts"].items():
                self.results[self.indexToClass[i]]["counts"][count] = self.metrics["counts"][count](tp, tn, fp, fn)
            # Populate Statistics Dictionary
            for stat, func in self.metrics["statistics"].items():
                self.results[self.indexToClass[i]]["stats"][stat]   = self.metrics["statistics"][stat](tp, tn, fp, fn)
        return self

    #
    # Getter Methods
    #

    # Get Class Map
    def getClasses (self):
        return self.classes

    # Get Class Counts
    def getClassBalance (self):
        return { self.indexToClass[i] : self.results[self.indexToClass[i]]["counts"]["pos"] for i, _ in enumerate(self.classes) }

    # Get Class Proportions
    def getClassProportions (self):
        return { self.indexToClass[i] : self.results[self.indexToClass[i]]["counts"]["prop"] for i, _ in enumerate(self.classes) }

    # Get Available Keys For Counts & Statistics
    def getAvailable (self):
        return {"stats" : list(self.metrics["statistics"].keys()), "counts" : list(self.metrics["counts"].keys()) }

    # Get Statistic For All Classes
    def getStatistic (self, statistic = "accuracy"):
        return statistic, { self.indexToClass[i]: self.results[self.indexToClass[i]]["stats"][statistic] for i, _ in enumerate(self.classes) }

    # Get Counts For All Classes
    def getCount (self, count = "tp"):
        return count, { self.indexToClass[i]: self.results[self.indexToClass[i]]["counts"][count] for i, _ in enumerate(self.classes) }

    # Get All Statistics For All Classes
    def getStatistics (self):
        return { self.indexToClass[i]: self.results[self.indexToClass[i]]["stats"] for i, _ in enumerate(self.classes) }

    # Get All Counts For All Classes
    def getCounts (self):
        return { self.indexToClass[i]: self.results[self.indexToClass[i]]["counts"] for i, _ in enumerate(self.classes) }

    # Get Statistic By Specified Class
    def getStatisticByClass (self, c, statistic = "accuracy"):
        return statistic, self.results[c]["stats"][statistic]

    # Get Count By Specified Class
    def getCountByClass (self, c, count = "tp"):
        return count, self.results[c]["counts"][count]

    # Get All Counts & Statistics
    def getAll (self):
        return self.results

    # Get Confusion Matrix
    def getConfusionMatrix(self, normalized = False):
        if normalized:
            return self.normed
        else:
            return self.matrix

    #
    # Print Functions
    #

    # Print Summary Statistics
    def summarize (self):
        for i, c in enumerate(self.classes):
            print("=" * 30)
            print("%s" % str(c))
            print("-- Counts")
            for key, val in sorted(self.results[self.indexToClass[i]]["counts"].items(), key = lambda item: (len(item[0]), item[0])):
                print("   %s: %s" % (key.ljust(16), str(val).ljust(8)))
            print("\n-- Statistics")
            for key, val in sorted(self.results[self.indexToClass[i]]["stats"].items(), key = lambda item: (len(item[0]), item[0])):
                print("   %s: %s" % (key.ljust(16), ("%0.4f%%" % (val * 100)).ljust(8)))
        print("=" * 30)

    # Print Confusion Matrix
    def tabulate (self, normalized = False):
        minlen = max([len(str(c)) for c, n in self.getClassBalance().items()])
        cellwidth = minlen if minlen > 7 else 7
        print("=" * (cellwidth * (len(self.classes) + 2)))
        if normalized:
            print("        %s\n" % " ".join([("%sᴬ" % c).ljust(cellwidth)[0:cellwidth] for c in self.classes]))
            for c, row in zip(self.classes, self.normed):
                print("%s %s\n" % (("%sᴾ" % c).ljust(cellwidth)[0:cellwidth], " ".join([("%0.2f%%" % (val * 100)).ljust(cellwidth)[0:cellwidth] for val in row])))
        else:
            print("        %s\n" % " ".join([("%sᴬ" % c).ljust(cellwidth)[0:cellwidth] for c in self.classes]))
            for c, row in zip(self.classes, self.matrix):
                print("%s %s\n" % (("%sᴾ" % c).ljust(cellwidth)[0:cellwidth], " ".join([str(val).ljust(cellwidth)[0:cellwidth] for val in row])))
        print("Note: classᴾ = Predicted, classᴬ = Actual")
        print("=" * (cellwidth * (len(self.classes) + 2)))

# Example Usage
if __name__ == "__main__":
    
    # Actual & Predicted Classes
    actual      = ["A", "B", "C", "C", "B", "C", "C", "B", "A", "A", "B", "A", "B", "C", "A", "B", "C"]
    predicted   = ["A", "B", "B", "C", "A", "C", "A", "B", "C", "A", "B", "B", "B", "C", "A", "A", "C"]

    # Initialize Performance Class
    performance = Performance(actual, predicted)
    
    # Print Statistical Summary
    performance.summarize()
    
    # Print Confusion Matrix
    performance.tabulate()
    
    # Print Normalized Confusion Matrix
    performance.tabulate(normalized = True)
    
else:
    pass