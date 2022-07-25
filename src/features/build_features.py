# -*- coding: utf-8 -*-
# @author: Boi Mai Quach <quachmaiboi@gmail.com>
################################################

import os
import sys
from time import time
# Bring your packages onto the path
sys.path.insert(0, os.getcwd())
from src.data.preprocessing import Preprocessing
from src.helper.helpers import __feature_files__, progressBar

import cv2
import numpy as np
from scipy.stats import kurtosis, skew
import mahotas as mt
from src.features.elliptic_fourier_descriptions import EllipticFourierDescriptors


"""
    Handfeature extraction 
    Color, Shape, Vein, Texture, Fourier Descriptor, XY_projection, Image
"""

run_preprocessing = Preprocessing()

def get_color_features(leaf, mask):
    """
    Color feature extraction
    The four descriptive statistics (mean, variance, skewnessm and kurtosis) are extracted 
    based on three color space: RGB, HSV, and HSL

    Parameters
    ----------
    leaf: numpy.ndarray
        A color image
    mask: numpy.ndarray
        Binary mask and contour
    
    Returns
    -------
    color_features: list
        All statistical values from 3 color spaces.

    """

    bgr = leaf.copy()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)

    mask = mask > 0
    blue = bgr[:,:,0][mask]
    green = bgr[:,:,1][mask]
    red = bgr[:,:,2][mask]
    hsv_hue = hsv[:,:,0][mask]
    hsv_sat = hsv[:,:,1][mask]
    hsv_val = hsv[:,:,2][mask]
    hls_hue = hls[:,:,0][mask]
    hls_lig = hls[:,:,1][mask]
    hls_sat = hls[:,:,2][mask]

    channels = [blue, green, red, hsv_hue, hsv_sat, hsv_val, hls_hue, hls_lig, hls_sat]
    means = list(map(np.mean, channels))
    stds = list(map(np.std, channels))
    skews = list(map(skew, channels))
    kurtosisses = list(map(kurtosis, channels))
    color_features = means + stds + skews + kurtosisses

    return color_features

def get_shape_features(contour):
    """
    Shape feature extraction
    The feature groups includes geometric features, morphological, and moment features 

    Parameters
    ----------
    contour: numpy.ndarray
        A strutural outlines of objects in an image
    
    Returns
    -------
    shape_features: list
        All shape feature values.

    """

    #Geometric features
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour,True)
    M = cv2.moments(contour)
    _, _, physiological_width, physiological_height = cv2.boundingRect(contour)
    circularity = ((perimeter)**2)/area
    equi_diameter = np.sqrt(4*area/np.pi)

    #Morphological features
    aspect_ratio = float(physiological_width)/physiological_height
    form_factor = (4*np.pi*area)/(perimeter**2)
    rectangularity = physiological_width*physiological_height/area
    narrow_factor = equi_diameter/physiological_height
    ratio1 = perimeter/equi_diameter
    ratio2 = perimeter/(physiological_width+physiological_height)


    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area

    shape_features = list(M.values()) + [area,perimeter,physiological_width,physiological_height,aspect_ratio,rectangularity,circularity,equi_diameter,\
                  equi_diameter,form_factor,narrow_factor,ratio1,ratio2,solidity]
    return shape_features

def get_vein_image(gray):
    """
    Vein image extraction
    Vein images are obtained by applying morphological operations

    Parameters
    ----------
    gray: numpy.ndarray
        An image in gray scale
    
    Returns
    -------
    vein_image: numpy.ndarray
        A vein leaf image
    """
    ####Procedure
    #1. Gaussian Filter
    blur = cv2.GaussianBlur(gray, (25,25),0)
    #2. Create disk-shaped structuring elements of each radius 
    r4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    #3. Erosion and dilation
    e4 = cv2.erode(blur, r4, iterations=1)
    d4 = cv2.dilate(e4, r4, iterations=1)
    #4. The operation creates subtracted grayscale images of the leaf.
    sd4 = blur - d4

    vein_image = (sd4 > 0).astype(np.uint8)
    
    return vein_image

def get_texture_features(gray):
    """
    Texture feature extraction
    Texture features are calculated from a gray level co-occurrence matrix (GLCM)

    Parameters
    ----------
    gray: numpy.ndarray
        An image in gray scale
    
    Returns
    -------
    texture_features: list
        Fourteen haralick features.

    """
    textures = mt.features.haralick(gray)
    ht_mean = textures.mean(axis=0)
    texture_features = list(ht_mean)

    return texture_features

def get_elliptic_fourier_descriptors(mask, N=10):
    """
    Fourier descriptor feature extraction

    Parameters
    ----------
    mask: numpy.ndarray
        The boundary points of the leaf region after con- verting them into the binary image. 
    
    Returns
    -------
    fourier_features: list
        Fourier Descriptor feature vector

    """
    efds_instance = EllipticFourierDescriptors()
    efds, _, _ = efds_instance.elliptic_fourier_descriptors(mask, N=N)
    efds = efds[0]
    fourier_features = list(efds.flatten())

    return fourier_features

def get_xyprojection(mask, bins=30):
    """
    Vertical and Horizontal projection feature extraction

    Parameters
    ----------
    mask: numpy.ndarray
        The boundary points of the leaf region after con- verting them into the binary image. 
    
    Returns
    -------
    xyprojection_features: list
        A vector of length 60 consisting of two xy-projection histograms.

    """
    width = mask.shape[0]//bins
    n_total_pixels = width*mask.shape[0]

    projection = np.empty(shape=(bins*2), dtype=np.float32)
    for i, bin_s in enumerate(range(0, mask.shape[0], width)):
        bin_e = bin_s + width
        projection[i] = np.sum(mask[bin_s:bin_e,:] > 0)
        projection[i+bins] = np.sum(mask[:,bin_s:bin_e] > 0)
    projection = projection/n_total_pixels

    return projection

def get_features(image):

    ## extract vein image
    vein = get_vein_image(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    vein = 255*cv2.resize(vein, (300,300))

    ## preparation for handcrafted feature extraction
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask, contour = run_preprocessing.get_binary_mask(gray, return_contour=True)

    ## color
    color = get_color_features(image, mask)

    ## shape
    shape = get_shape_features(contour)

    ## texture
    texture = get_texture_features(gray)

    ## fourier
    fourier = get_elliptic_fourier_descriptors(mask, N=10)

    ## xyprojection
    xyprojection = get_xyprojection(mask)

    # return resized, vein, color, shape, texture, fourier, xyprojection
    return vein, xyprojection, color, texture, fourier, shape


def main(input_filepath, output_filepath):
    """ 
    Runs build_features scripts to extract all feautures from (..data/processed)  to (saved in ../processed).

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

    f_images = np.empty((1907,300,300,3), dtype=np.uint8)
    f_veins = np.empty((1907,300,300), dtype=np.uint8)
    f_colors = np.empty((1907,36), dtype=np.float32)
    f_shapes = np.empty((1907,38), dtype=np.float32)
    f_textures = np.empty((1907,13), dtype=np.float32)
    f_fouriers = np.empty((1907,40), dtype=np.float32)
    f_xyprojections = np.empty((1907, 60), dtype=np.float32)

    end_progress = len(files)

    for i, file_name in enumerate(files):
        file_path = os.path.join(input_filepath, file_name)
        # processed_path = os.path.join(output_filepath, file_name)

        # Read image 
        image = cv2.imread(file_path)
        resized_image = cv2.resize(image, (300,300))
        # Get features
        vein, xyprojection, color, texture, fourier, shape = get_features(resized_image)

        f_images[i] = resized_image
        f_veins[i] = vein
        f_colors[i] = color
        f_shapes[i] = shape
        f_textures[i] = texture
        f_fouriers[i] = fourier
        f_xyprojections[i] = xyprojection

        progressBar(i+1, end_progress)
        
        #Save features
        np.save(__feature_files__['image'], f_images)
        np.save(__feature_files__['vein'], f_veins)
        np.save(__feature_files__['color'], f_colors)
        np.save(__feature_files__['shape'], f_shapes)
        np.save(__feature_files__['texture'], f_textures)
        np.save(__feature_files__['fourier'], f_fouriers)
        np.save(__feature_files__['xyprojection'], f_xyprojections)

if __name__ == '__main__':
    input_filepath = "data/processed/"
    output_filepath = "data/features/"
    tt = time()
    main(input_filepath,output_filepath)
    print()
    print("Running time: ", time() - tt)
    print("Features extraction: Done.")
    

    