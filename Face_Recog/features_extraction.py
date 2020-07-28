from Face_Recog.divide_image import FeatureExtraction
import glob
import cv2
import numpy as np
from Face_Recog.ultils import tranpose,covariance

data_path = "C:/Users/maiho/PycharmProjects/DPT/database/Face_Detected"
features_path =  "C:/Users/maiho/PycharmProjects/DPT/Face_Recog/features_color_histogram.csv"
# initialize the color descriptor
cd = FeatureExtraction((16, 24, 8))
# open the output index file for writing
output = open(features_path, "w")
# use glob to grab the image paths and loop over them
face_vector = []
for imagePath in glob.glob(data_path + "/*.jpg"):

	imageID = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)
	# print(image.shape)
	# describe the image
	features = cd.extract(image)
	face_vector.append(features)
	features = [str(f) for f in features]

	output.write("%s,%s\n" % (imageID, ",".join(features)))
# close the index file
output.close()

face_vector = np.asarray(face_vector)
face_vector = face_vector.transpose()

#STEP2: Normalize the face vectors by calculating the average face vector and subtracting it from each vector
avg_face_vector = face_vector.mean(axis=1)
avg_face_vector = avg_face_vector.reshape(face_vector.shape[0], 1)
normalized_face_vector = face_vector - avg_face_vector

#STEP3: Calculate the Covariance Matrix or the Sigma
covariance_matrix = np.cov(np.transpose(normalized_face_vector))
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

# chuyen vi ma tran

# tranpose_matrix = tranpose(normalized_face_vector)
# covariance_matrix = covariance(tranpose_matrix)
# covariance_matrix = np.transpose(normalized_face_vector).dot(normalized_face_vector)
# print(covariance_matrix)

# STEP4: Calculate Eigen Vectors


# STEP5: Select the K best Eigen Faces, K < M

k = 30
k_eigen_vectors = eigen_vectors[0:k, :]
eigen_faces = k_eigen_vectors.dot(np.transpose(normalized_face_vector))


# STEP7: Represent Each eigen face as combination of the K Eigen Vectors
weights = np.transpose(normalized_face_vector).dot(np.transpose(eigen_faces))
pca_features_path =  "C:/Users/maiho/PycharmProjects/DPT/Face_Recog/features_pca.csv"
output_1 = open(pca_features_path, "w")

# use glob to grab the image paths and loop over them
for i,imagePath in enumerate(glob.glob(data_path + "/*.jpg")):
	# extract the image ID (i.e. the unique filename) from the image
	# path and load the image itself
	imageID = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)

	features = weights[i]
	# write the features to file
	features = [str(f) for f in features]

	output_1.write("%s,%s\n" % (imageID, ",".join(features)))
output_1.close()





















# # STEP8: Testing Phase
# test_add = "C:/Users/maiho/PycharmProjects/DPT/Face_Recog/s50_15.jpg"
#
# cd = ColorDescriptor((8, 12, 3))
# # load the query image and describe it
# query = cv2.imread(test_add)
# query = cv2.resize(query,(240,360))
# print(query.shape)
# features_r = cd.describe(query)
#
# features_r = np.asarray(features_r)
# print(features_r.shape)
# features_r = features_r.reshape(1440,1)
# test_normalized_face_vector = features_r - avg_face_vector
# test_weight = np.transpose(test_normalized_face_vector).dot(np.transpose(eigen_faces))
# index = np.argmin(np.linalg.norm(test_weight - weights, axis=1))
# # print("------------------")
# # print(weights[345])
# print(index)
# for i,imagePath in enumerate(glob.glob(data_path + "/*.jpg")):
#     if(i==index):
#         result = cv2.imread(imagePath)
#         result = cv2.resize(result, (256, 256))
#         cv2.imshow("Result", result)
#         cv2.waitKey(0)

