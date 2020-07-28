import numpy as np
def tranpose(x):
    result = [[x[j][i] for j in range(len(x))] for i in range(len(x[0]))]
    result = np.array(result)
    return result
def covariance(x):
    result = [[x[j][i] for j in range(len(x))] for i in range(len(x[0]))]
    result = np.array(result)
    return result
def rgb_to_gray(image):
    grayValue = 0.2989 * image[:,:,2] + 0.5870 * image[:,:,1] + 0.1140 * image[:,:,0]
    gray_img = grayValue.astype(np.uint8)
    return gray_img
def rgb_to_hsv(image):
    height, width, channel = image.shape

    # Create balnk HSV image
    img_hsv = np.zeros((height, width, 3))
    for i in np.arange(height):
        for j in np.arange(width):
            r = image.item(i, j, 2)
            g = image.item(i, j, 1)
            b = image.item(i, j, 0)
            r_ = r / 255.
            g_ = g / 255.
            b_ = b / 255.
            Cmax = max(r_, g_, b_)
            Cmin = min(r_, g_, b_)
            delta = Cmax - Cmin
            # Hue Calculation
            if delta == 0:
                H = 0
            elif Cmax == r_:
                H = ((60 * ((g_ - b_) / delta))%6)
            elif Cmax == g_:
                H = 60 * (((b_ - r_) / delta) + 2)
            elif Cmax == b_:
                H = 60 * (((r_ - g_) / delta) + 4)
            # Saturation Calculation
            if Cmax == 0:
                S = 0
            else:
                S = (delta / Cmax)*255
            # Value Calculation
            V = Cmax*255

            # Set H,S,and V to image
            img_hsv.itemset((i, j, 0), int(H))
            img_hsv.itemset((i, j, 1), int(S))
            img_hsv.itemset((i, j, 2), int(V))

    return img_hsv
