from GENERATE_REPRESENTATIONS_DEFESA import Representations
import cv2
import numpy as np
import os
import dlib
from imutils import face_utils
import math
from sklearn.model_selection import train_test_split


unlabeled_path = './kyoto/'

def load_unlabeled_database(dir_path):
    size_input_data = [96, 96, 1]
    cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
    img_list = os.listdir(dir_path)
    img_data_list = []

    if '.' in img_list[0]:
        for img in img_list:
            input_img = cv2.imread(dir_path + img, cv2.IMREAD_GRAYSCALE)
            input_img = cv2.resize(input_img, (size_input_data[0], size_input_data[1]))
            input_img = np.reshape(input_img, (size_input_data[0], size_input_data[1], size_input_data[2]))
            img_data_list.append(input_img)
    else:
        for dataset in img_list:
            imgs = os.listdir(dir_path + '/' + dataset)
            for img in imgs:
                input_img = cv2.imread(dir_path + '/' + dataset + '/' + img, cv2.IMREAD_GRAYSCALE)

                rects = cascade.detectMultiScale(input_img, 1.3, 3, cv2.CASCADE_SCALE_IMAGE, (50, 50))

                if len(rects) > 0:
                    facerect = rects[0]
                    input_img = input_img[facerect[1]:facerect[1] + facerect[3], facerect[0]:facerect[0] + facerect[2]]

                input_img = cv2.resize(input_img, (size_input_data[0], size_input_data[1]))
                input_img = np.reshape(input_img, (size_input_data[0], size_input_data[1], size_input_data[2]))
                img_data_list.append(input_img)

    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32') / 255.0
    n_train = int(img_data.shape[0] * 0.7)
    trainX, testX = img_data[:n_train, :], img_data[n_train:, :]

    return trainX, testX

if __name__ == '__main__':
    # carregar base nao rotulada
    trainx, testx = load_unlabeled_database(unlabeled_path)

    teste = Representations(trainx, testx)


    teste.Generate_all(epochs = 3, seeds_rep = False, arch_rep=False, hidden_rep = True)

