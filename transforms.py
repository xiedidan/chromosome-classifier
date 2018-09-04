import torchvision.transforms as transforms
import numpy as np
import cv2

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
