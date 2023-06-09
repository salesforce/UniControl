import cv2

class Blurrer:
    def __call__(self, img, ksize):
        img_new = cv2.GaussianBlur(img, (ksize, ksize), cv2.BORDER_DEFAULT)
        img_new = img_new.astype('ubyte')
        return img_new
