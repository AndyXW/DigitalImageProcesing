import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    for img_name in os.listdir('pic'):
        print(img_name)
        # read image, to gray and normalize
        # img_name = './pic/13.bmp'
        img_name = './pic/' + img_name
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = (img - np.min(img)) / (np.max(img)-np.min(img))

        # calculate laplacien and gradient of img 
        laplace = cv2.Laplacian(img, -1, ksize=3)

        grad_x = cv2.Sobel(img, -1, 1, 0)
        grad_y = cv2.Sobel(img, -1, 0, 1)
        gradx = cv2.convertScaleAbs(grad_x)
        grady = cv2.convertScaleAbs(grad_y)
        gradient = np.sqrt(gradx**2+grady**2)

        # get boundary pixels
        flag1 = (gradient <= 0)
        flag2 = (laplace >= 0.015)
        boundary = flag1 & flag2
        
        threshold = np.sum(img * boundary) / np.sum(boundary)
        # threshold segmentation
        retval, segmentation = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
        save_img_name = img_name.split('/')[-1].split('.')[0] + '_seg.png'

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(img, cmap='gray')
        plt.text(220, -10, 'Origin', fontsize=16)
        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.imshow(segmentation, cmap='gray')
        plt.text(220, -10, 'My', fontsize=16)
        plt.savefig(save_img_name, dpi=500, facecolor='w')

if __name__ == '__main__':
    main()
