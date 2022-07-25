import os
import sys
sys.path.insert(0, os.getcwd())
from src.helper.helpers import prod_parameters, normalize_feature, load_features, __features__, load_labels, load_mapping_name, __prod_encoders_folder__, __prod_decoder_file__, __prod_decoders_folder__, __prod_encoder_file__
from src.models.model import EncoderExtractor
import pickle
import cv2
import numpy as np
import pandas as pd
from src.data.preprocessing import Preprocessing
from src.features.build_features import get_features
import warnings
warnings.filterwarnings('ignore')

def get_image_from_name(name_leaf):
    label_file = "data/external/prod_dataset_indexed.csv"
    df_label = pd.read_csv(label_file)
    db = df_label[df_label["Leaf"] == name_leaf]
    leaf_path = db["Filepath"].values[0]
    img = cv2.imread(leaf_path)

    return img

def predict_image(image_paths, encoder_paths, decoder_path):  
      
    y_names = load_mapping_name()
    run_preprocessing = Preprocessing()
    num_paths = len(image_paths)
    
    #Preprocessing
    f_images = np.empty((num_paths,300,300,3), dtype=np.uint8)
    f_veins = np.empty((num_paths,300,300), dtype=np.uint8)
    f_colors = np.empty((num_paths,36), dtype=np.float32)
    f_shapes = np.empty((num_paths,38), dtype=np.float32)
    f_textures = np.empty((num_paths,13), dtype=np.float32)
    f_fouriers = np.empty((num_paths,40), dtype=np.float32)
    f_xyprojections = np.empty((num_paths, 60), dtype=np.float32)
    for i, path in enumerate(image_paths):
        image = cv2.imread(path)
        processed_image = run_preprocessing.rotate_img(image)
        resized_image = cv2.resize(processed_image, (300,300))
        #Extract features
        vein, xyprojection, color, texture, fourier, shape = get_features(resized_image)
        f_images[i] = resized_image
        f_veins[i] = vein
        f_xyprojections[i] = xyprojection
        f_colors[i] = color
        f_textures[i] = texture
        f_fouriers[i] = fourier
        f_shapes[i] = shape
        X = [f_images, f_veins, f_xyprojections, f_colors, f_textures, f_fouriers, f_shapes]
        norm_X = normalize_feature(X, __features__)
    
    #Encoder
    extractor = EncoderExtractor()
    extractor.load_encoders(encoder_paths)
    norm_X = extractor.extract(norm_X)
    
    #Decoder
    prod_decoder = pickle.load(open(decoder_path, 'rb'))
    
    #Prediction
    y_pred = prod_decoder.predict(norm_X)
    y_pred_names = [y_names[pred] for pred in y_pred]

    return y_pred_names

