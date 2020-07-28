# import the necessary packages
from Face_Recog.divide_image import FeatureExtraction
from Face_Recog.vectors_distance import Searcher
from Face_Recog.features_extraction import avg_face_vector,eigen_faces
import cv2
import numpy as np
# construct the argument parser and parse the arguments
indexed_path =  "C:/Users/maiho/PycharmProjects/DPT/Face_Recog/features_pca.csv"
query_path = "C:/Users/maiho/PycharmProjects/DPT/Face_Recog/s02_04.jpg"
results_path =  "C:/Users/maiho/PycharmProjects/DPT/database"

# initialize the image descriptor
cd = FeatureExtraction((16, 24, 8))
# load the query image and describe it
query = cv2.imread(query_path)

query = cv2.resize(query,(240,360))
features_query = cd.extract(query)

features_query = np.asarray(features_query)
features_query = features_query.reshape(features_query.shape[0],1)
test_normalized_face_vector = features_query - avg_face_vector
test_weight = np.transpose(test_normalized_face_vector).dot(np.transpose(eigen_faces))
test_weight = np.asarray(test_weight)

#array to vector
test_weight = np.concatenate(test_weight)
# perform the search
searcher = Searcher(indexed_path)
results = searcher.knn(test_weight)

# display the query
cv2.imshow("Query", query)
cv2.waitKey(0)
# loop over the results
for (score, resultID) in results:
	# load the result image and display it
	print(resultID)
	print("===============")
	print(score)
	result = cv2.imread(results_path + "/" + resultID)
	result = cv2.resize(result, (240, 360))
	cv2.imshow("Result", result)
	cv2.waitKey(0)