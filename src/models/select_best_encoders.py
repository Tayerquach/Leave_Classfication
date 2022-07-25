

import os
import sys
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np
from src.helper.helpers import __features__, __encoder_performances_file__, __best_encoder_file__, num_folds

def select_best_encoders(performamces_path):
    performances = pd.read_csv(performamces_path)
    best_encoders = pd.DataFrame(columns=performances.columns)
    i = 0
    for fold in range(1, num_folds + 1):
        for feature in __features__:
            encoders = performances[np.logical_and(performances['feature'] == feature, performances['fold'] == fold)]
            best_encoder = encoders.loc[encoders['val_acc'].idxmax()]
            best_encoders.loc[i] = best_encoder
            i += 1
    
    return best_encoders

if __name__ == "__main__":
    best_encoders = select_best_encoders(__encoder_performances_file__)
    best_encoders.to_csv(__best_encoder_file__, index = False)
    print("Successfully!")

    