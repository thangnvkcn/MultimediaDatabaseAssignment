# import the necessary packages
from Face_Recog.divide_image import ColorDescriptor
from Face_Recog.vectors_distance import Searcher
from mlxtend.image import extract_face_landmarks
import math
import cv2
import imageio
import numpy as np
# construct the argument parser and parse the arguments
indexed_path =  "C:/Users/maiho/PycharmProjects/DPT/Face_Recog/features.csv"
query_path = "C:/Users/maiho/PycharmProjects/DPT/Face_Recog/s50_04.jpg"
results_path =  "C:/Users/maiho/PycharmProjects/DPT/database"

image = imageio.imread(query_path)
image = cv2.resize(image, (160, 240))
print(image.shape)
landmarks = extract_face_landmarks(image)
# print(landmarks.shape)

img2 = image.copy()
p1 = landmarks[37]
p2 = landmarks[40]
p3 = landmarks[43]
p4 = landmarks[46]
p5 = landmarks[32]
p6 = landmarks[36]
p7 = landmarks[49]
p8 = landmarks[55]
p9 = landmarks[52]
p10 = landmarks[58]
p11 = landmarks[28]
p12 = landmarks[31]
p13 = landmarks[28]
p14 = landmarks[40]
distance_eye_left = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
distance2 = math.sqrt(((p3[0] - p4[0]) ** 2) + ((p3[1] - p4[1]) ** 2))
distance3 = math.sqrt(((p5[0] - p6[0]) ** 2) + ((p5[1] - p6[1]) ** 2))
distance4 = math.sqrt(((p7[0] - p8[0]) ** 2) + ((p7[1] - p8[1]) ** 2))
distance5 = math.sqrt(((p9[0] - p10[0]) ** 2) + ((p9[1] - p10[1]) ** 2))
distance6 = math.sqrt(((p11[0] - p12[0]) ** 2) + ((p11[1] - p12[1]) ** 2))
distance7 = math.sqrt(((p13[0] - p14[0]) ** 2) + ((p13[1] - p14[1]) ** 2))
features = []
features.append(distance_eye_left)
features.append(distance2)
features.append(distance3)
features.append(distance4)
features.append(distance5)
features.append(distance6)
features.append(distance7)
norm = np.linalg.norm(features)
normal_array = features/norm
print(normal_array)
from sklearn.preprocessing import scale
# features = scale( features, axis=0, with_mean=True, with_std=True, copy=True )
# print(features)

# perform the search
searcher = Searcher(indexed_path)
results = searcher.search(normal_array)
# display the query

cv2.imshow("Query", image)
cv2.waitKey(0)
# loop over the results
for (score, resultID) in results:
	# load the result image and display it
	print(resultID)
	print("=======")
	print(score)
	result = cv2.imread(results_path + "/" + resultID)
	result = cv2.resize(result, (500, 500))
	cv2.imshow("Result", result)
	cv2.waitKey(0)