from skimage import color

class GrayscaleConverter:
    def __call__(self, img):
        return (color.rgb2gray(img) * 255.0).astype('ubyte')
