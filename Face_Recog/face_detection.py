import cv2
import  glob
from Face_Recog.ultils import rgb_to_gray
face__folder_paths = glob.glob("C:/Users/maiho/PycharmProjects/DPT/database/Face_data/*")
for sub_folder in face__folder_paths:
    sub = sub_folder[sub_folder.rfind("\\") + 1:]
    sub_folder = glob.glob(sub_folder+"/*.jpg")

    for face_paths in sub_folder:
        imageID = face_paths[face_paths.rfind("\\") + 1:]
        image = cv2.imread(face_paths)
        gray = rgb_to_gray(image)

        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60)
        )

        print("[INFO] Found {0} Faces.".format(len(faces)))

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = image[y:y + h, x:x + w]
            print("[INFO] Object found. Saving locally.")
            cv2.imwrite("C:/Users/maiho/PycharmProjects/DPT/database/Face_Detected/"+sub+"_"+imageID, roi_color)


