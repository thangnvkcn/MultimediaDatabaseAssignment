# import the necessary packages
import glob
import numpy as np
import cv2

from matplotlib import pyplot as plt
import os
# construct the argument parser and parse the arguments
data_path = "C:/Users/maiho/PycharmProjects/DPT/database/Face_Detected"
indexed_path =  "C:/Users/maiho/PycharmProjects/DPT/Face_Recog/index.csv"



face_vector = []
image_width = 240
image_length = 360
total_pixels = image_width*image_length


for imagePath in glob.glob(data_path + "/*.jpg"):
    face_image = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_RGB2GRAY)

    face_image = cv2.resize(face_image, (240, 360))
    # cv2.imshow("a",face_image)
    # cv2.waitKey(0)
    #     plt.imshow(face_image, cmap = 'gray', interpolation = 'bicubic')
    #     plt.show()
    face_image = face_image.reshape(total_pixels, )
    face_vector.append(face_image)

face_vector = np.asarray(face_vector)

face_vector = face_vector.transpose()


#STEP2: Normalize the face vectors by calculating the average face vector and subtracting it from each vector
avg_face_vector = face_vector.mean(axis=1)

avg_face_vector = avg_face_vector.reshape(face_vector.shape[0], 1)
normalized_face_vector = face_vector - avg_face_vector
print(normalized_face_vector)

#STEP3: Calculate the Covariance Matrix or the Sigma
covariance_matrix = np.cov(np.transpose(normalized_face_vector))
# covariance_matrix = np.transpose(normalized_face_vector).dot(normalized_face_vector)
print(covariance_matrix)

# STEP4: Calculate Eigen Vectors
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

# STEP5: Select the K best Eigen Faces, K < M
print(eigen_vectors.shape)
k = 30
k_eigen_vectors = eigen_vectors[0:k, :]
print(k_eigen_vectors.shape)

#STEP6: Convert lower dimensionality K Eigen Vectors to Original Dimensionality
eigen_faces = k_eigen_vectors.dot(np.transpose(normalized_face_vector))
print(eigen_faces.shape)

# STEP7: Represent Each eigen face as combination of the K Eigen Vectors
# weights = eigen_faces.dot(normalized_face_vector)
weights = np.transpose(normalized_face_vector).dot(np.transpose(eigen_faces))
print(weights[1])

# STEP8: Testing Phase
test_add = "C:/Users/maiho/PycharmProjects/DPT/Face_Recog/s50_15.jpg"
test_img = cv2.imread(test_add)
test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
test_img = cv2.resize(test_img, (240, 360))
test_img = test_img.reshape(total_pixels, 1)

test_normalized_face_vector = test_img - avg_face_vector
test_weight = np.transpose(test_normalized_face_vector).dot(np.transpose(eigen_faces))
print(test_weight.shape)
index = np.argmin(np.linalg.norm(test_weight - weights, axis=1))
# print("------------------")
# print(weights[345])
print(index)
for i,imagePath in enumerate(glob.glob(data_path + "/*.jpg")):
    if(i==index):
        result = cv2.imread(imagePath)
        result = cv2.resize(result, (256, 256))
        cv2.imshow("Result", result)
        cv2.waitKey(0)
