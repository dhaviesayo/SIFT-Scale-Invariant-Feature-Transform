import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

!git clone "https://github.com/rmislam/PythonSIFT" pysift
import pysift.pysift as pysift



class sift_keypoint(torch.nn.modules.Module):
    def __init__(self ,  train_img ,  query_img):
        super(sift_keypoint , self).__init__()
        
        def to_gray(color_img):
            gray = transforms.Grayscale()(color_img)
            return gray

        self.train= train_img
        self.query= query_img
        self.to_gray = to_gray
        

    def forward(self ,  draw_n_matches):
        # Initialise SIFT detector
        sift = cv2.SIFT_create()
        
        to_gray = self.to_gray
        train_image = self.train
        query_image = self.query
        
        train_img_gray = transforms.Normalize([0.5,] , [0.5,] )(to_gray(train_image.to(torch.float64)))
        query_img_gray = transforms.Normalize([0.5,] , [0.5,] )(to_gray(query_image.to(torch.float64)))

        
        # Generate SIFT keypoints and descriptors
        train_kp, train_desc = pysift.computeKeypointsAndDescriptor(np.array(train_img_gray.to(torch.uint8)), None)
        query_kp, query_desc = pysift.computeKeypointsAndDescriptor(np.array(query_img_gray.to(torch.uint8)), None)

        # create a BFMatcher object which will match up the SIFT features
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        matches = bf.match(train_desc, query_desc)

        # Sort the matches in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        # draw the top N matches
        N_MATCHES = draw_n_matches

        match_img = cv2.drawMatches(
            train_img, train_kp,
            query_img, query_kp,
            matches[:N_MATCHES], query_img.copy(), flags=0)
        
        return match_img , train_kp , train_desc , query_kp , query_desc
