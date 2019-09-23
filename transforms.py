import os

import torchvision.transforms as transforms
import numpy as np
import cv2
import PIL
from sklearn.cluster import DBSCAN

def binaryzation(img, binary_threshold):
    _, b_img = cv2.threshold(img, binary_threshold, 255, cv2.THRESH_BINARY_INV)
    
    return b_img

def opening(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    return opening_img

def huge_filter(indexes, areas, huge_threshold):
    normal_indexes = []
    
    for index in indexes:
        if areas[index] < huge_threshold:
            normal_indexes.append(index)
            
    return normal_indexes

def denoise(img, area_threshold):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    areas = []
    for contour in contours:
        areas.append(cv2.contourArea(contour))
    
    filtered_indexes = []
    for i, area in enumerate(areas):
        if area > area_threshold:
            filtered_indexes.append(i)
    
    return filtered_indexes, areas, contours

def circularity(contour):
    area = cv2.contourArea(contour)
    center, r = cv2.minEnclosingCircle(contour)
    
    ratio = np.min((1., area / (r * r * np.pi)))

    return ratio

def cell_filter(indexes, areas, contours, circularity_threshold, cell_threshold):
    c_indexes = []

    for index, area in zip(indexes, areas):
        if circularity(contours[index]) > circularity_threshold and area > cell_threshold:
            pass
        else:
            c_indexes.append(index)

    return c_indexes

def dbscan_filter(indexes, contours, eps, min_samples, img_size):
    # find out contour centers
    centers = []
    for index in indexes:
        center, r = cv2.minEnclosingCircle(contours[index])
        centers.append(center)
    
    if len(centers) == 0:
        return []

    dataset = np.array(centers)
    pred = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(dataset)
    
    # select the cluster in the center
    clusters = {}
    
    for i, p in enumerate(pred):
        center = dataset[i]
        
        if p!= -1:
            if p not in clusters.keys():
                clusters[p] = { 'center': center, 'count': 1 }
            else:
                clusters[p]['count'] += 1
                clusters[p]['center'] += center
            
    center_dists = []
    img_center = np.array(img_size) / 2
    
    for key in clusters.keys():
        cluster = clusters[key]
        cluster_center = cluster['center'] / cluster['count']
        center_dists.append(np.linalg.norm(img_center - cluster_center))
    
    if len(center_dists) == 0:
        return []

    selected_cluster = list(clusters.keys())[np.argmin(center_dists)]
    selected_indexes = []
    
    for i, p in enumerate(pred):
        if p == selected_cluster:
            selected_indexes.append(indexes[i])
        
    return selected_indexes

class AutoMask:
    def __init__(
        self,
        binary_threshold=225,
        noise_threshold=200,
        opening_kernel_size=5,
        huge_threshold=30000,
        circularity_threshold=0.65,
        cell_threshold=7500,
        dbscan_eps=200,
        dbscan_samples=5
        ):
        self.binary_threshold = binary_threshold
        self.noise_threshold = noise_threshold
        self.opening_kernel_size = opening_kernel_size
        self.huge_threshold = huge_threshold
        self.circularity_threshold = circularity_threshold
        self.cell_threshold = cell_threshold
        self.dbscan_eps = dbscan_eps
        self.dbscan_samples = dbscan_samples

    def __call__(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        b_img = binaryzation(gray, self.binary_threshold)
        o_img = opening(b_img, self.opening_kernel_size)
        indexes, areas, contours = denoise(o_img, self.noise_threshold)
        indexes = huge_filter(indexes, areas, huge_threshold=self.huge_threshold)
        indexes = cell_filter(indexes, areas, contours, self.circularity_threshold, self.cell_threshold)
        indexes = dbscan_filter(indexes, contours, self.dbscan_eps, self.dbscan_samples, [gray.shape[1], gray.shape[0]])

        # create mask
        mask = np.zeros_like(gray)
        for i in indexes:
            cv2.drawContours(mask, contours, i, 255, -1)

        # apply mask
        new_img = np.full_like(gray, 255)
        np.copyto(new_img, gray, where=(mask>127))

        return new_img

# convert PIL.Image to ndarray
class ToNumpy(object):
    def __init__(self):
        pass
    
    def __call__(self, img):
        return np.asarray(img)

class Invert(object):
    def __init__(self):
        pass
    
    def __call__(self, img):
        return PIL.ImageOps.invert(img)

class AutoLevel(object):
    def __init__(self, max_rate=0.0001, min_rate=0.0001):
        self.max_rate = max_rate
        self.min_rate = min_rate

    def __call__(self, img):
        # convert from PIL to ndarray
        image = np.array(img)

        if image.ndim == 3:
            h, w, c = image.shape
            new_image = np.zeros([h, w, c])

            for i in range(c):
                image_hist = self._compute_hist(image[:, :, i])
                min_level = self._compute_min_level(image_hist, self.min_rate, h * w)
                max_level = self._compute_max_level(image_hist, self.max_rate, h * w)
                new_map = self._linear_map(max_level, min_level)

                if len(new_map) == 0:
                    continue
                
                for j in range(h):
                    new_image[j, :, i] = new_map[image[j, :, i]]
        else: # gray picture
            h, w= image.shape
            new_image = np.zeros([h, w], dtype=np.uint8)

            image_hist = self._compute_hist(image)
            min_level = self._compute_min_level(image_hist, self.min_rate, h * w)
            max_level = self._compute_max_level(image_hist, self.max_rate, h * w)
            new_map = self._linear_map(max_level, min_level)

            if len(new_map) == 0:
                return img
            
            for j in range(h):
                new_image[j, :] = new_map[image[j, :]]
            
            new_image = new_image[:, :, np.newaxis]

        return transforms.functional.to_pil_image(new_image)

        

    def __repr__(self):
        return self.__class__.__name__ + '(max_rate={0}, min_rate={1})'.format(self.max_rate, self.min_rate)

    def _compute_hist(self, image):
        h, w = image.shape
        hist, bin_edge = np.histogram(image.reshape(1, h * w), bins=list(range(257)))

        return hist

    def _compute_min_level(self, hist, rate, count):
        sum = 0

        for i in range(256):
            sum += hist[i]
            if sum >= count * rate:
                return i

    def _compute_max_level(self, hist, rate, count):
        sum = 0

        for i in range(256):
            sum += hist[255 - i]
            if sum >= count * rate:
                return 255 - i

    def _linear_map(self, max_level, min_level):
        if min_level >= max_level:
            return []
        else:
            new_map = np.zeros(256, dtype=np.uint8)

            for i in range(256):
                if i < min_level:
                    new_map[i] = 0
                elif i > max_level:
                    new_map[i] = 255
                else:
                    new_map[i] = (i - min_level) / (max_level - min_level) * 255
            
            return new_map
