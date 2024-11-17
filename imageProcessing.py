import cv2
from datetime import datetime
import os
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from scipy import ndimage
import skimage.color as sc
import numpy as np
import skimage
from scipy.ndimage import gaussian_filter as gauss
import math


class ImgProcesing:
    
    def __init__(self):
        self.img = ''
        self.sigma = 5 # 1, 3, 5, 10, 20
        self.kernel_size = 3 
       
        self.vertices =   np.array([[(0,64),(0,10),(90,40),(60,64)]], dtype=np.int64)

    def gaussBlur(self, img):
        return cv2.GaussianBlur(img,(3,3), sigmaX=0, sigmaY=0)
    
    def Sobel(self, img):
        gX = cv2.Sobel(self.gaussBlur(img), ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
        gY = cv2.Sobel(self.gaussBlur(img), ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
        gX = cv2.convertScaleAbs(gX)
        gY = cv2.convertScaleAbs(gY)
        combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
        return combined
    
    def Laplaciano(self, img):
        laplacian_img = cv2.Laplacian(img, ddepth=cv2.CV_64F, ksize=3)
        return laplacian_img
    
    def Canny(self, img):
        canny_img = cv2.Canny(img, threshold1=150, threshold2=250)
        return canny_img
    
    def Erosion(self, img):
        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.dilate(img, kernel, 2)
        return erosion
    
    def RoiFilter(self, img):
        roi_img = np.zeros_like(img)
        roi_img = cv2.fillPoly(roi_img, self.vertices, 255)
        roi_mask = cv2.bitwise_and(img, roi_img)
        return roi_mask
    
    def displayImage(self, img):
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap='gray')

    def pipeline(self, img):
        gauss_blur = self.gaussBlur(img)
        gauss_blur = self.gaussBlur(gauss_blur)
        #gauss_blur = self.gaussBlur(gauss_blur)
        #sobel_filter= self.Sobel(gauss_blur)
        canny_filter = self.Canny(gauss_blur)
        #sobel_filter= self.Sobel(canny_filter)
        #lapacian_filter= self.Laplaciano(gauss_blur)
        img_mask = self.RoiFilter(canny_filter)
        return img_mask
    
    def Hough(self, img):
        rho = 1
        theta = np.pi/180
        threshold = 15
        min_line_len = 10
        max_line_gap = 5
        img_maks = self.pipeline(img)
        lines = cv2.HoughLinesP(img_maks, rho, theta, threshold,
                                        np.array([]), minLineLength=min_line_len,
                                        maxLineGap=max_line_gap)
        img_lines = np.zeros((img_maks.shape[0], img_maks.shape[1], 3), dtype=np.uint8)
      
        return lines
        
        #print('lines',lines)
            
    
    def addWeighted(self, rgbImg, img_lines):
        alpha = 1
        beta = 1
        gamma = 1
        img_lane_lines = cv2.addWeighted(rgbImg, alpha, img_lines, beta, gamma)
        return img_lane_lines
    
    def calculate_steering_angle(self, lines):
    # Calcular el ángulo promedio de las líneas detectadas
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
                angles.append(angle)
            average_angle = np.mean(angles)
            return np.round(average_angle)
        else:
            return 0 
    
def main():
    ImgProc = ImgProcesing()
    img = cv2.imread('./Images_auto/image.png')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ImgProc.displayImage(ImgProc.Sobel(img_gray))

if __name__ == "__main__":
    main()