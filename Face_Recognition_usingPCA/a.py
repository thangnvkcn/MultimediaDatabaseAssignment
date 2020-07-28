import sklearn
from scipy.spatial import distance
import numpy as np
import librosa
import os
import progressbar

labels = ["Dog","Rooster","Cow",  "Cat", "Hen", "Frog"]

def dtw(s1, s2, dist=distance.euclidean, plot=False):
    s1 = np.insert(s1, 0, 0, axis=0)
    s2 = np.insert(s2, 0, 0, axis=0)
    m = len(s1)
    n = len(s2)
    DTW = np.full((n, m), float("inf"), dtype=float)
    DTW[0, 0] = 0
    for i in range(1, n):
        for j in range(1, m):
            cost = dist(s1[j], s2[i])
            DTW[i, j] = np.min(np.array([DTW[i - 1, j], DTW[i - 1, j - 1], DTW[i, j - 1]])
                               + cost)

    return DTW[n - 1, m - 1] / (n + m - 2)

def wav2mfcc(file_path, n_mfcc=12, hop_length=1024):
    wave, sr = librosa.load(file_path, sr=16000, dtype=np.float32)
    yf, _ = librosa.effects.trim(wave, top_db=20)
    mfcc = librosa.feature.mfcc(y=yf, sr=16000, hop_length=hop_length, n_mfcc=n_mfcc)  #n_mfcc x n_frame_$
    spectral_centroid_feat = librosa.feature.spectral_centroid(y=wave, sr=16000, hop_length=hop_length) #1 x n_frame
    mse_feat = np.sqrt(librosa.feature.rmse(y=wave, hop_length=hop_length)) #1xn_frame
    return [mfcc, spectral_centroid_feat, mse_feat]

def load_data(path):
    X = []
    y = []
    count = 0
    for subdir in os.listdir(path):
        for filename in os.listdir(path+'/'+subdir):
            features = wav2mfcc(path+'/'+subdir+'/'+filename)
            X.append(features)
            y.append(count)
        count+=1
    return X, y

def voting(list_predict):
    return max(list_predict, key=list_predict.count)

def recognie(X_train, X_test, y_train):
    y_pred = []
    bar = progressbar.ProgressBar(maxval=len(X_test))
    for sample in bar(X_test):
        print('yes')
        min_dist = np.inf
        predicted_class = None
        min_dist1 = np.inf
        predicted_class1 = None
        min_dist2 = np.inf
        predicted_class2 = None
        for other, other_class in zip(X_train, y_train):
            distance = dtw(sample[0].T, other[0].T)
            distance1 = dtw(sample[1].T, other[1].T)
            distance2 = dtw(sample[2].T, other[2].T)
            if distance < min_dist:
                min_dist = distance
                predicted_class = other_class
            if distance1 < min_dist1:
                min_dist1 = distance1
                predicted_class1 = other_class
            if distance2 < min_dist2:
                min_dist2 = distance2
                predicted_class2 = other_class
        list_predict = [predicted_class, predicted_class1, predicted_class2]
        y_pred.append(voting(list_predict))
    return y_pred

X_train, y_train = load_data("audi2_Copy")
# print(y_train)
X_test, y_test = load_data("test")
y_pred = recognie(X_train,X_test,y_train)

# print(sklearn.metrics.classification_report(y_test, y_pred, digits=2))
print('done')