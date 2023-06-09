import numpy as np

class Inpainter:
    def __call__(self, img, rand_h, rand_h_1, rand_w, rand_w_1):
        h = img.shape[0]
        w = img.shape[1]
        h_new = int(float(h) / 100.0 * float(rand_h))
        w_new = int(float(w) / 100.0 * float(rand_w))
        h_new_1 = int(float(h) / 100.0 * float(rand_h_1))
        w_new_1 = int(float(w) / 100.0 * float(rand_w_1))
        
        img_new = img
        img_new[(h-h_new)//2:(h+h_new_1)//2, (w-w_new)//2:(w+w_new_1)//2] = 0
        img_new = img_new.astype('ubyte')
        return img_new
