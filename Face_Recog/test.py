import imageio
import matplotlib.pyplot as plt
import  glob
import math
import cv2
from mlxtend.image import extract_face_landmarks
data_path = "C:/Users/maiho/PycharmProjects/DPT/database/Face_Detected"
features_path =  "C:/Users/maiho/PycharmProjects/DPT/Face_Recog/features.csv"
output = open(features_path, "w")
for i,imagePath in enumerate(glob.glob(data_path + "/*.jpg")):
    imageID = imagePath[imagePath.rfind("/") + 1:]
    image = imageio.imread(imagePath)
    print(image)
    # image = cv2.resize(image,(160,240))
    # print(image.shape)
    # landmarks = extract_face_landmarks(image)
    # # print(landmarks.shape)
    #
    # img2 = image.copy()
    # p1 = landmarks[37]
    # p2 = landmarks[40]
    # p3 = landmarks[43]
    # p4 = landmarks[46]
    # p5 = landmarks[32]
    # p6 = landmarks[36]
    # p7 = landmarks[49]
    # p8 = landmarks[55]
    # p9 = landmarks[52]
    # p10 = landmarks[58]
    # p11 = landmarks[28]
    # p12 = landmarks[31]
    # p13 = landmarks[28]
    # p14 = landmarks[40]
    # distance_eye_left = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    # distance2 = math.sqrt(((p3[0] - p4[0]) ** 2) + ((p3[1] - p4[1]) ** 2))
    # distance3 = math.sqrt(((p5[0] - p6[0]) ** 2) + ((p5[1] - p6[1]) ** 2))
    # distance4 = math.sqrt(((p7[0] - p8[0]) ** 2) + ((p7[1] - p8[1]) ** 2))
    # distance5 = math.sqrt(((p9[0] - p10[0]) ** 2) + ((p9[1] - p10[1]) ** 2))
    # distance6 = math.sqrt(((p11[0] - p12[0]) ** 2) + ((p11[1] - p12[1]) ** 2))
    # distance7 = math.sqrt(((p13[0] - p14[0]) ** 2) + ((p13[1] - p14[1]) ** 2))
    # features = []
    # features.append(distance_eye_left)
    # features.append(distance2)
    # features.append(distance3)
    # features.append(distance4)
    # features.append(distance5)
    # features.append(distance6)
    # features.append(distance7)
    # import numpy as np
    # norm = np.linalg.norm(features)
    # normal_array = features / norm
    #
    # from sklearn.preprocessing import scale
    #
    # # features = scale(features, axis=0, with_mean=True, with_std=True, copy=True)
    # features = [str(f) for f in normal_array]
    # output.write("%s,%s\n" % (imageID, ",".join(features)))
# close the index file
# output.close()
#     if(i==743):
#         p1 = landmarks[37]
#         print(landmarks[0:10])
#         fig = plt.figure(figsize=(15, 5))
#         ax = fig.add_subplot(1, 3, 1)
#         ax.imshow(image)
#         ax = fig.add_subplot(1, 3, 2)
#         ax.scatter(landmarks[:, 0], -landmarks[:, 1], alpha=0.8)
#         ax = fig.add_subplot(1, 3, 3)
#         for p in landmarks:
#             img2[p[1] - 3:p[1] + 3, p[0] - 3:p[0] + 3, :] = (255, 255, 255)
#             # note that the values -3 and +3 will make the landmarks
#             # overlayed on the image 6 pixels wide; depending on the
#             # resolution of the face image, you may want to change
#             # this value
#
#         ax.imshow(img2)
#         plt.show()
#
