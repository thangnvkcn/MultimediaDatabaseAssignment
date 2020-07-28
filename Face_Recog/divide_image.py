
import numpy as np
import cv2
from Face_Recog.ultils import rgb_to_hsv
from matplotlib import pyplot as plt
import imutils
class FeatureExtraction:
	def __init__(self, bins):
		# store the number of bins for the 3D histogram
		self.bins = bins
	def extract(self, image):
		image = cv2.resize(image, (240, 360))
		mage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		features = []
		(h, w) = image.shape[:2]
		(cX, cY) = (int(w * 0.5), int(h * 0.5))
		segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
					(0, cX, cY, h)]
		(axesX, axesY) = (int(w * 0.7) // 2, int(h * 0.7) // 2)
		ellipMask = np.zeros(image.shape[:2], dtype="uint8")
		cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
		cv2.imshow("elipmask", ellipMask)
		cv2.waitKey(0)
		for (startX, endX, startY, endY) in segments:
			cornerMask = np.zeros(image.shape[:2], dtype="uint8")
			cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
			cornerMask = cv2.subtract(cornerMask, ellipMask)
			# cv2.imshow("cornermask",cornerMask)
			# cv2.waitKey(0)
			hist = self.histogram(image, cornerMask)
			features.extend(hist)
		hist = self.histogram(image, ellipMask)
		features.extend(hist)
		return features

	def histogram(self, image, mask):
		hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
							[0, 180, 0, 256, 0, 256])
		# hist = hist.flatten()
		# plt.plot(hist)
		# plt.xlim([0, 256])
		# plt.show()

		# normalize the histogram
		hist = cv2.normalize(hist, hist).flatten()
		# plt.plot(hist)
		# plt.xlim([0,256])
		# plt.show()
		return hist