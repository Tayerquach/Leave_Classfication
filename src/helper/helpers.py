# -*- coding: utf-8 -*-
# @author: Boi Mai Quach <quachmaiboi@gmail.com>
################################################

# from pyexpat import features
import sys
from sklearn.model_selection import train_test_split
# Set seed value
seed_value = 23
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

import pickle

breakpoints = [1001,1059,
        1060,1122,
        1552,1616,
        1123,1194,
        1195,1267,
        1268,1323,
        1324,1385,
        1386,1437,
        1497,1551,
        1438,1496,
        2001,2050,
        2051,2113,
        2114,2165,
        2166,2230,
        2231,2290,
        2291,2346,
        2347,2423,
        2424,2485,
        2486,2546,
        2547,2612,
        2616,2675,
        3001,3055,
        3056,3110,
        3111,3175,
        3176,3229,
        3230,3281,
        3282,3334,
        3335,3389,
        3390,3446,
        3447,3510,
        3511,3563,
        3566,3621]

name_classes = ["pubescent bamboo", "Chinese horse chestnut", "Anhui Barberry", "Chinese redbud", "true indigo",\
        "Japanese maple", "Nanmu", "castor aralia", "Chinese cinnamon", "goldenrain tree",\
        "Big-fruited Holly", "Japanese cheesewood", "wintersweet", "camphortree", "Japan Arrowwood",\
        "sweet osmanthus", "deodar", "ginkgo-maidenhair tree", "Crape myrtle, Crepe myrtle", "oleander",\
        "yew plum pine", "Japanese Flowering Cherry", "Glossy Privet", "Chinese Toon", "peach",\
        "Ford Woodlotus", "trident maple", "Beale's barberry", "southern magnolia", "Canadian poplar",\
        "Chinese tulip tree", "tangerine"]

__features__ = ['image', 'vein', 'xyprojection', 'color', 'texture', 'fourier', 'shape']

__feature_shape__ = {
	'image': [300,300,3],	
	'vein': [300,300],	
	'xyprojection': [60,],
	'color': [36,],
	'texture': [13,],
	'fourier': [40,],
	'shape': [38,],
}

__feature_files__ = {
    'image': "data/features/images.npy",
	'vein': "data/features/vein.npy",
	'xyprojection': "data/features/xyprojection.npy",
	'color': "data/features/color.npy",
	'texture': "data/features/texture.npy",
	'fourier': "data/features/fourier.npy",
	'shape': "data/features/shape.npy",
}

__result_files__ ={
	'prediction': "data/interim/predicted_data.npy",
	'true': "data/interim/actual_data.npy",
	'false': "data/interim/false_data.npy",
	'kfold_val_acc': "data/interim/kfold_val_acc.npy",
	'kfold_test_acc': "data/interim/kfold_test_acc.npy",
}

__visualisation_files__ = {
	'result_table': "reports/kfold_accuracy.csv", 
	'confusion_matrix': "reports/figures/confusion_matrix.png",
	'misclassified_images': "reports/figures/misclassified_leave_prediction.png",
}


__label_files__ ={
	'labels': "data/processed/labels/labels.npy",
	'mapping_names': "data/processed/labels/mapping_names.pkl"
} 

__normalizing_features__ = ['color', 'texture', 'fourier', 'shape', "combine"]

__index_dataset__ = "data/external/prod_dataset_indexed.csv"

__index_kfold__ = "data/external/Dataset_10FoldCV_indexed.csv"

__model_file__ = "ENCODER-{}-l2rate{}-dropout{}-fold{}.h5"
__prod_encoder_file__ = "PROD_ENCODER-{}-l2rate{}-dropout{}.h5"

__models_folder__ = "models/Dataset_10FoldCV_indexed_models"
__prod_encoders_folder__ = "models/prod_models/encoders"
__prod_decoders_folder__ = "models/prod_models/decoders"

__encoder_performances_file__ = "data/interim/Dataset_10FoldCV_indexed_encoders_performances.csv"

__prod_encoder_performance_file__ = "data/interim/prod_encoders_performances.csv"

__best_encoder_file__ = "data/interim/train_encoders_best.csv"

__decoder_file__ = "DECODER-fold{}.pickle"
__prod_decoder_file__ = "PROD_DECODER.pickle"

__decoder_performances_file__ = "data/interim/Dataset_10FoldCV_indexed_decoders_performances.csv"

prod_parameters = {
    "image": (0.0001, 0.5),
    "vein" : (0.001, 0.5),
    "xyprojection": (0.1, 0.5),
    "color": (0.01, 0.5),
    "texture": (0.01, 0.5),
    "fourier": (0.1, 0.5),
    "shape": (0.001, 0.5)
    }

num_folds = 10


def progressBar(value, endvalue, bar_length=20):

    """ Runs the loading bar to observe the process
    """

    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rPercent: [{0}] {1}% {2}/{3} ".format(arrow + spaces, int(round(percent * 100)), value, endvalue))
    sys.stdout.flush()

def load_features(feature):
    if type(feature) == list:
        return [np.load(__feature_files__[f]) for f in feature]
    return np.load(__feature_files__[feature])

def load_labels():
    return np.load(__label_files__["labels"])

def load_mapping_name():
	with open(__label_files__["mapping_names"], 'rb') as f:
		mapping_names = pickle.load(f)
	return mapping_names


def normalize_feature_data(feature, X_train, X_valid, X_test):
    """normalize data
    any feature in __normalizing_features__ is normalized, otherwise kept intact
    """
    if type(feature) == list:
        for i, f in enumerate(feature):
            
            if f in __normalizing_features__:
                stds = np.std(X_train[i], axis=0)
                stds[stds==0.0] = 1.0
                means = np.mean(X_train[i], axis=0)
                X_train[i] = (X_train[i]-means)/stds
                X_valid[i] = (X_valid[i]-means)/stds
                X_test[i] = (X_test[i]-means)/stds
    else:
        if feature in __normalizing_features__:
            stds = np.std(X_train, axis=0)
            stds[stds==0.0] = 1.0
            means = np.mean(X_train, axis=0)
            X_train = (X_train-means)/stds
            X_valid = (X_valid-means)/stds
            X_test = (X_test-means)/stds
            
    return X_train, X_valid, X_test

def split_train_test_valid(feature, Kfold, fold, X, y):
    """split dataset X, y into train, valid, test sets based on kfold and fold
    """
    fold = "Fold_" + str(fold)
    
    train_index = Kfold[fold] == "Train"
    valid_index = Kfold[fold] == "Valid"
    test_index = Kfold[fold] == "Test"

    if type(feature) == list:
        X_train = [x[train_index] for x in X]
        X_valid = [x[valid_index] for x in X]
        X_test = [x[test_index] for x in X]
    else:
        X_train = X[train_index]
        X_valid = X[valid_index]
        X_test = X[test_index]

    ## normalize handcrafted features
    X_train, X_valid, X_test = normalize_feature_data(feature, X_train, X_valid, X_test)

    y_train = y[train_index]
    y_valid = y[valid_index]
    y_test = y[test_index]

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

def normalize_arr(arr):
	stds = np.std(arr)
	if stds == 0.0:
		stds = 1.0
	means = np.mean(arr)
	norm_arr = (arr-means)/stds
	return norm_arr

def normalize_feature(X, features):
	if len(features) > 1:
		for idx, f in enumerate(features):          
			if f in __normalizing_features__:
				feature_arr = X[idx].copy()
				new_feature_arr = []
				for item in feature_arr:
					norm_arr = normalize_arr(item)
					new_feature_arr.append(norm_arr)
				temp = np.asarray(new_feature_arr)
				X[idx] = temp

	else:
		if features[0] in __normalizing_features__:
			feature_arr = X.copy()
			new_feature_arr = []
			for item in feature_arr:
				norm_arr = normalize_arr(item)
				new_feature_arr.append(norm_arr)
			temp = np.asarray(new_feature_arr)
			X = temp
		
	return X



def split_train_test_valid_prod(feature, X, y):
	X_train = []
	X_test  = []
	X_valid = []

	if len(feature) > 1:
		for i in range(len(feature)):
			train, test, y_train, y_test = train_test_split(X[i], y, test_size=0.2, random_state=seed_value)
			test, valid, y_test, y_valid = train_test_split(test, y_test, test_size=0.5, random_state=seed_value)
			X_train.append(train)
			X_test.append(test)
			X_valid.append(valid)
	else:
		X_train, X_test, y_train, y_test = train_test_split(X[0], y, test_size=0.2, random_state=seed_value)
		X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=seed_value)

	## normalize handcrafted features
	X_train = normalize_feature(X_train, feature)
	X_valid = normalize_feature(X_valid, feature)
	X_test = normalize_feature(X_test, feature)
	# X_train, X_valid, X_test = normalize_feature_data(feature, X_train, X_valid, X_test)

	return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

	
