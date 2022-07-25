# -*- coding: utf-8 -*-
# @author: Boi Mai Quach <quachmaiboi@gmail.com>
##################### 

import numpy as np
import os
import cv2
import sys


class Preprocessing:
    def __init__(self):
        """
        Data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).

        Parameters
        ----------
        image_path: str
            A string representing the path of the image to be read.
        
        Returns
        -------
        rotated_img: numpy.ndarray
            Preprocessed image
        """

    ### Preprocessing
    def pad_image_to_square(self, img, addition=0):
        """
        Convert initial image into a square boundary

        Parameters
        ----------
        img: numpy.ndarray
            A RGB image

        addition: int
            Additional boundary
        
        Returns
        -------
        constant: numpy.ndarray
            A square image
        """
        height, width = img.shape[:2]
        if width > height:
            dif = width - height
            if dif % 2 == 0:
                top, bottom = dif//2, dif//2
            else:
                top, bottom = dif//2 + 1, dif//2
            constant = cv2.copyMakeBorder(img,top + addition,bottom + addition,
                                        addition,addition,
                                        cv2.BORDER_CONSTANT,
                                        value=[255,255,255])

        else:
            dif = height - width
            if dif % 2 == 0:
                left, right = dif//2, dif//2
            else:
                left, right = dif//2 + 1, dif//2
            constant = cv2.copyMakeBorder(img,addition,addition,
                                        left + addition,right + addition,
                                        cv2.BORDER_CONSTANT,
                                        value=[255,255,255])

        return constant

    def get_binary_mask(self,img_gs, return_contour=False):
        """
        Convert RGB image into black/white image

        Parameters
        ----------
        img_gs: numpy.ndarray
            A Gray image
        return_contour: bool
            Keep contour or not
        
        Returns
        -------
        mask: numpy.ndarray
            Binary mask and contour
        """
        _, thresh = cv2.threshold(img_gs, 239, 1, cv2.THRESH_BINARY_INV)

        # only keep biggest contour
        contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        biggest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros(img_gs.shape[:2], np.uint8)
        # hull = cv2.convexHull(biggest_contour)
        cv2.drawContours(mask, [biggest_contour], -1, 1, -1)
        if return_contour:
            return mask, biggest_contour
        return mask

    def crop_and_flip_image(self, img, mask=None):
        """
        Crop and flip image

        Parameters
        ----------

        img: numpy.ndarray
            A RGB image

        mask: None
            The Mask of a given image
        
        Returns
        -------
        flip_crop_img: numpy.ndarray
            Image after cropping and flipping
        """
        if not mask:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            mask = self.get_binary_mask(gray)
        # crop
        leave_indices = np.argwhere(mask > 0)
        x1 = np.min(leave_indices[:,0])
        x2 = np.max(leave_indices[:,0]) + 1
        y1 = np.min(leave_indices[:,1])
        y2 = np.max(leave_indices[:,1]) + 1
        
        crop_img =  img[x1:x2, y1:y2, :]
        mask = mask[x1:x2, y1:y2]

        flip_crop_img = crop_img
        
        # flip
        h, w = crop_img.shape[:2]
        left = np.sum(mask[:,:w//2])
        right = np.sum(mask[:,w//2:])
        if left < right:
            flip_crop_img = cv2.flip(crop_img, 1)
            
        upper = np.sum(mask[:h//2,:])
        lower = np.sum(mask[h//2:,:])
        if upper > lower:
            flip_crop_img = cv2.flip(crop_img, 0)

        return flip_crop_img

    def get_rotation_angle(self, mask):
        """
        Calculate the rotation angle.
        Using PCA to find 2 main principal components in which the first one is corresponding to the main vein.
        Rotating the image until the first component is horizontal.

        Parameters
        ----------
        mask: numpy.ndarray
            Binary mask and contour 
        
        Returns
        -------
        cor_center: numpy.ndarray
            coordinate of a center point in xy axis
        angle: numpy.float64
            The angle used to rotate the image
        eigenvectors: numpy.ndarray
            a special set of vectors associated with two principal components
        """   
        leaf_indices = np.argwhere(mask > 0).astype(np.float32)
        mean = np.mean(leaf_indices, axis=0)
        center, eigenvectors = cv2.PCACompute(leaf_indices, mean=np.asarray([mean]))
        angle = np.arctan(eigenvectors[0,0]/eigenvectors[0,1])  
        angle = angle * 180.0/np.pi 
        cor_center = center[0,::-1]
        return cor_center, angle, eigenvectors



    def rotate_img(self, img):
        """
        Rotate image

        Parameters
        ----------
        img: numpy.ndarray
            A RGB image
        
        Returns
        -------
        rotated_img: numpy.ndarray
            Image after preprocessing
        """

        ## get rotate angle
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        mask = self.get_binary_mask(gray)
        if False:
            resized = cv2.resize(mask, (300,300))
        gray[mask  < 1] = 0
        _, angle, _ = self.get_rotation_angle(mask)

        ## padding image that prevents leaves from cut out while rotating
        pad_size = int(0.5*max(img.shape[:2]))
        pad_img = cv2.copyMakeBorder(img,pad_size,pad_size,
                                pad_size,pad_size,
                                cv2.BORDER_CONSTANT,
                                value=[255,255,255])
        ## rotate
        height, width = pad_img.shape[:2]
        # rotation_matrix = cv2.getRotationMatrix2D(tuple(center +np.asarray([pad_size,pad_size])) , angle, 1)  # get Rotation Matrix
        rotation_matrix = cv2.getRotationMatrix2D((height//2, width//2) , angle, 1)  # get Rotation Matrix
        rotated_img = cv2.warpAffine(pad_img, rotation_matrix, (width, height), borderValue=(255, 255, 255))
        
        ## crop the leave patch and pad to square size
        rotated_img = self.pad_image_to_square(self.crop_and_flip_image(rotated_img))

        return rotated_img

    


        