# -*- coding: utf-8 -*-
# @author: Boi Mai Quach <quachmaiboi@gmail.com>
################################################

import os
import cv2
import sys
# Bring your packages onto the path
sys.path.insert(0, os.getcwd())
from preprocessing import Preprocessing
from src.helper.helpers import __label_files__, progressBar, breakpoints, __index_dataset__
import numpy as np
import pandas as pd
import pickle
from time import time

def main(input_filepath, output_filepath):
    """ 
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    
    Parameters
    ----------
        input_filepath: str
            A string representing the path of the image to be read.
        output_filepath: str
            A string representing the path of the image to be saved.
        
    Returns
    -------
        None
    """
 
    if not os.path.exists(output_filepath):
        os.mkdir(output_filepath)
    
    files = os.listdir(input_filepath)
    files = [f for f in files if f[-4:] == ".jpg"]

    df = pd.read_csv(__index_dataset__)
    end_progress = len(files)
    labels_name = []
    labels = []

    for i, file_name in enumerate(files):
        file_path = os.path.join(input_filepath, file_name)
        processed_path = os.path.join(output_filepath, file_name)

        #### Create dataset
        # Read image 
        image = cv2.imread(file_path)
        processed_image = run_preprocessing.rotate_img(image)
        progressBar(i+1, end_progress)
        
        #Save image
        cv2.imwrite(processed_path, processed_image)

        #Create labels
        
        leaf_name = df[df["Filename"] == file_name]["Leaf"].values[0]
        image_num = int(file_name.split(".")[0])
        flag = False
        i = 0
        for i in range(0, len(breakpoints), 2):
            if (image_num >= breakpoints[i]) and (image_num <= breakpoints[i+1]):
                flag = True
                break
        if flag:
            label = int(i/2)
            labels.append(label)
            labels_name.append(leaf_name)

    mapping_names = dict((set(list(zip(labels, labels_name)))))
        
    # Save labels
    np.save(__label_files__['labels'], labels)
    with open(__label_files__['mapping_names'], 'wb') as f:
        pickle.dump(mapping_names, f)
        
        
if __name__ == "__main__":
    input_filepath = "data/raw/Leaves"
    output_filepath = "data/processed/dataset/"
    run_preprocessing = Preprocessing()
    tt = time()
    main(input_filepath, output_filepath)
    print()
    print("Running time: ", time() - tt)