import imageio
import matplotlib.pyplot as plt
import  glob
import math
import cv2
import numpy as np
from Face_Recognition_usingPCA.description import extract_features
from mlxtend.image import extract_face_landmarks
data_path = "C:/Users/maiho/PycharmProjects/DPT/database/Face_Detected"
features_path =  "C:/Users/maiho/PycharmProjects/DPT/Face_Recognition_usingPCA/features.csv"
output = open(features_path, "w")
for i,imagePath in enumerate(glob.glob(data_path + "/*.jpg")):
    imageID = imagePath[imagePath.rfind("/") + 1:]
    image = cv2.imread(imagePath)

    features = extract_features(image)
    norm = np.linalg.norm(features)
    normal_array = features / norm
    features = [str(f) for f in normal_array]
    output.write("%s,%s\n" % (imageID, ",".join(features)))
output.close()