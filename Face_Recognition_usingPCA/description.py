import imageio
import matplotlib.pyplot as plt
import  glob
import math
import cv2
from mlxtend.image import extract_face_landmarks
import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
def extract_features(image):
    image = cv2.resize(image, (360, 540))
    print(image.shape)
    landmarks = extract_face_landmarks(image)
    # img2 = image.copy()
    # fig = plt.figure(figsize=(15, 5))
    # ax = fig.add_subplot(1, 3, 1)
    # ax.imshow(image)
    # ax = fig.add_subplot(1, 3, 2)
    # ax.scatter(landmarks[:, 0], -landmarks[:, 1], alpha=0.8)
    # ax = fig.add_subplot(1, 3, 3)
    # for idx,p in enumerate(landmarks):
    #     img2[p[1] - 3:p[1] + 3, p[0] - 3:p[0] + 3, :] = (255, 255, 255)
    #     cv2.putText(img2, str(idx), (p[0], p[1]), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    #                 fontScale=0.4,
    #                 color=(0, 0, 255))
    # ax.imshow(img2)
    # plt.show()
    # landmarks = extract_face_landmarks(image)
    print(landmarks.shape)
    img2 = image.copy()
    p1 = landmarks[36]
    p2 = landmarks[39]
    p3 = landmarks[30]


    a = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    b = math.sqrt(((p1[0] - p3[0]) ** 2) + ((p1[1] - p3[1]) ** 2))
    c = math.sqrt(((p2[0] - p3[0]) ** 2) + ((p2[1] - p3[1]) ** 2))
    s = (a + b + c) / 2
    area_A1 = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    p4 = landmarks[42]
    a = math.sqrt(((p2[0] - p4[0]) ** 2) + ((p2[1] - p4[1]) ** 2))
    b = math.sqrt(((p4[0] - p3[0]) ** 2) + ((p4[1] - p3[1]) ** 2))
    c = math.sqrt(((p2[0] - p3[0]) ** 2) + ((p2[1] - p3[1]) ** 2))
    s = (a + b + c) / 2
    area_A2 = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    p5 = landmarks[45]
    p6 = landmarks[54]
    a = math.sqrt(((p4[0] - p5[0]) ** 2) + ((p4[1] - p5[1]) ** 2))
    b = math.sqrt(((p4[0] - p6[0]) ** 2) + ((p4[1] - p6[1]) ** 2))
    c = math.sqrt(((p5[0] - p6[0]) ** 2) + ((p5[1] - p6[1]) ** 2))
    s = (a + b + c) / 2
    area_A3 = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    p7 = landmarks[48]
    a = math.sqrt(((p1[0] - p7[0]) ** 2) + ((p1[1] - p7[1]) ** 2))
    b = math.sqrt(((p1[0] - p3[0]) ** 2) + ((p1[1] - p3[1]) ** 2))
    c = math.sqrt(((p3[0] - p7[0]) ** 2) + ((p3[1] - p7[1]) ** 2))
    s = (a + b + c) / 2
    area_A4 = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    a = math.sqrt(((p4[0] - p6[0]) ** 2) + ((p4[1] - p6[1]) ** 2))
    b = math.sqrt(((p3[0] - p6[0]) ** 2) + ((p3[1] - p6[1]) ** 2))
    c = math.sqrt(((p3[0] - p4[0]) ** 2) + ((p3[1] - p4[1]) ** 2))
    s = (a + b + c) / 2
    area_A5 = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    a = math.sqrt(((p3[0] - p6[0]) ** 2) + ((p3[1] - p6[1]) ** 2))
    b = math.sqrt(((p3[0] - p7[0]) ** 2) + ((p3[1] - p7[1]) ** 2))
    c = math.sqrt(((p6[0] - p7[0]) ** 2) + ((p6[1] - p7[1]) ** 2))
    s = (a + b + c) / 2
    area_A6 = (s * (s - a) * (s - b) * (s - c)) ** 0.5

    p19 = landmarks[19]
    p37 = landmarks[37]
    p24 = landmarks[24]
    p44 = landmarks[44]
    p27 = landmarks[27]
    p30 = landmarks[30]
    p36 = landmarks[36]
    p39 = landmarks[39]
    p42 = landmarks[42]
    p45 = landmarks[45]
    p57 = landmarks[57]
    p8 = landmarks[8]
    mat_trai_x = round((p36[0] + p39[0]) // 2)
    mat_trai_y = round((p36[1] + p39[1]) // 2)

    mat_phai_x = round((p42[0] + p45[0]) // 2)
    mat_phai_y = round((p42[1] + p45[1]) // 2)
    distance_two_eye = math.sqrt(((mat_trai_x - mat_phai_x) ** 2) + ((mat_trai_y - mat_phai_y) ** 2))
    distance_eyebrow_eye_left = math.sqrt(((p19[0] - p37[0]) ** 2) + ((p19[1] - p37[1]) ** 2))
    distance_eyebrow_eye_right = math.sqrt(((p24[0] - p44[0]) ** 2) + ((p24[1] - p44[1]) ** 2))
    nose_along_lengh = math.sqrt(((p27[0] - p30[0]) ** 2) + ((p27[1] - p30[1]) ** 2))
    distance_eye_left = math.sqrt(((p36[0] - p39[0]) ** 2) + ((p36[1] - p39[1]) ** 2))
    distance_eye_right = math.sqrt(((p42[0] - p45[0]) ** 2) + ((p42[1] - p45[1]) ** 2))
    distance_chin_mouth = math.sqrt(((p8[0] - p57[0]) ** 2) + ((p8[1] - p57[1]) ** 2))
    features = []

    # distance_1 = math.sqrt(((mat_trai_x - p15[0]) ** 2) + ((mat_trai_y - p15[1]) ** 2))
    # distance_2 = math.sqrt(((mat_trai_x - p16[0]) ** 2) + ((mat_trai_y - p16[1]) ** 2))
    # distance_eye_left = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    # distance2 = math.sqrt(((p3[0] - p4[0]) ** 2) + ((p3[1] - p4[1]) ** 2))
    # distance_nose = math.sqrt(((p5[0] - p6[0]) ** 2) + ((p5[1] - p6[1]) ** 2))
    # distance_mouth_nose = math.sqrt(((p7[0] - p8[0]) ** 2) + ((p7[1] - p8[1]) ** 2))
    # distance_mouth = math.sqrt(((p9[0] - p10[0]) ** 2) + ((p9[1] - p10[1]) ** 2))
    # distance6 = math.sqrt(((p11[0] - p12[0]) ** 2) + ((p11[1] - p12[1]) ** 2))
    # distance7 = math.sqrt(((p13[0] - p14[0]) ** 2) + ((p13[1] - p14[1]) ** 2))
    # tam_x = round((mat_phai_x + mat_trai_x) // 2)
    # tam_y = round((mat_phai_y + mat_trai_y) // 2)
    # distance_8 = math.sqrt(((tam_x - p17[0]) ** 2) + ((tam_y - p17[1]) ** 2))
    # features.append(distance_two_eye)
    # features.append(distance_nose)
    # features.append(distance_mouth_nose)
    # features.append(distance_mouth)
    # features.append(distance6)
    # features.append(distance7)
    # features.append(distance_two_eye/distance_nose)
    # features.append(distance_two_eye/distance_mouth_nose)
    # features.append(distance_two_eye/distance_mouth)
    # features.append(distance_nose / distance_mouth_nose)
    # features.append(distance_nose / distance_mouth)
    # features.append(distance_mouth_nose / distance_mouth)
    # features.append(area_A1)
    # features.append(area_A2)
    # features.append(area_A3)
    # features.append(area_A4)
    # features.append(area_A5)
    # features.append(area_A6)
    features.append(distance_two_eye)
    features.append(distance_eyebrow_eye_left)
    features.append(distance_eyebrow_eye_right)
    features.append(nose_along_lengh)
    features.append(distance_eye_left)
    features.append(distance_eye_right)
    features.append(distance_chin_mouth)
    return features