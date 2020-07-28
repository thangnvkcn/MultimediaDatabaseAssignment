# import the necessary packages
from Face_Recognition_usingPCA.vectors_distance import Searcher
from Face_Recognition_usingPCA.description import extract_features
from mlxtend.image import extract_face_landmarks
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np

# construct the argument parser and parse the arguments
indexed_path =  "C:/Users/maiho/PycharmProjects/DPT/Face_Recognition_usingPCA/features.csv"
query_path = "C:/Users/maiho/PycharmProjects/DPT/Face_Recog/s"
results_path =  "C:/Users/maiho/PycharmProjects/DPT/database"

image = imageio.imread(query_path)
image = cv2.resize(image, (360, 540))
print(image.shape)
landmarks = extract_face_landmarks(image)
# print(landmarks.shape)

img2 = image.copy()

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 3, 1)
ax.imshow(image)
ax = fig.add_subplot(1, 3, 2)
ax.scatter(landmarks[:, 0], -landmarks[:, 1], alpha=0.8)
ax = fig.add_subplot(1, 3, 3)
for idx,p in enumerate(landmarks):
	img2[p[1] - 3:p[1] + 3, p[0] - 3:p[0] + 3, :] = (255, 255, 255)
	cv2.putText(img2, str(idx), (p[0], p[1]), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
				fontScale=0.4,
				color=(0, 0, 255))

ax.imshow(img2)
plt.show()
features = extract_features(image)
norm = np.linalg.norm(features)
normal_array = features/norm
from sklearn.preprocessing import scale
# features = scale( features, axis=0, with_mean=True, with_std=True, copy=True )
print(normal_array)

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
