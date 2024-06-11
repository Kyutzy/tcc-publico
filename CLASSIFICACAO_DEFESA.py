import numpy as np
import os
import cv2
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from deslib.des import KNORAU
from numpy.random import seed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential, model_from_json
import dlib
from imutils import face_utils
import math
import time
from sklearn.decomposition import PCA
import glob

labeled_path = "./jaffe/"
size_input_data = [96, 96, 1]
landmarks_predictor_model = './shape_predictor_68_face_landmarks.dat'
cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmarks_predictor_model)

def load_labeled_database(labeled_path):
    image_dir = "cohn-kanade-images"
    label_dir = "Emotion"

    features = []
    labels = np.zeros((327, 1))
    indiv_nomes = []
    counter = 0
    # Maybe sort them
    for participant in os.listdir(os.path.join(labeled_path, image_dir)):
        for sequence in os.listdir(os.path.join(labeled_path, image_dir, participant)):
            if sequence != ".DS_Store":
                image_files = sorted(os.listdir(os.path.join(labeled_path, image_dir, participant, sequence)))
                image_file = image_files[-1]
                input_img = cv2.imread(os.path.join(labeled_path, image_dir, participant, sequence, image_file))
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
                rects = cascade.detectMultiScale(input_img, 1.3, 3, cv2.CASCADE_SCALE_IMAGE, (150, 150))
                if len(rects) > 0:
                    facerect = rects[0]
                    input_img = input_img[facerect[1]:facerect[1] + facerect[3], facerect[0]:facerect[0] + facerect[2]]
                    input_img = cv2.resize(input_img, (size_input_data[0], size_input_data[1]))
                    # input_img = np.reshape(input_img, (size_to_resize[0], size_to_resize[1], size_to_resize[2]))
                features.append(input_img)
                indiv_nomes.append(participant)
                label_file = open(
                    os.path.join(labeled_path, label_dir, participant, sequence, image_file[:-4] + "_emotion.txt"))
                labels[counter] = eval(label_file.read())
                label_file.close()
                counter += 1

    print("individuos:", counter)
    img_data = np.array(features)
    img_data_preprocessing = preprocessing(img_data)

    return img_data_preprocessing, labels

def load_labeled_database_jaffe(labeled_path):
    expres_code = ['NE', 'HA', 'AN', 'DI', 'FE', 'SA', 'SU']

    data_dir_list = os.listdir(labeled_path)
    counter = 0
    features = []
    labels = np.zeros((213, 1))
    img_names = []

    for dataset in data_dir_list:
        img_list = os.listdir(labeled_path + '/' + dataset)
        for img in img_list:
            # imarray = cv2.imread(jaffe_dir + '/' + dataset + '/' + img, cv2.IMREAD_GRAYSCALE)
            imarray = cv2.imread(labeled_path + '/' + dataset + '/' + img)
            imarray = cv2.cvtColor(imarray, cv2.COLOR_BGR2GRAY)

            rects = cascade.detectMultiScale(imarray, 1.3, 3, cv2.CASCADE_SCALE_IMAGE, (150, 150))
            if len(rects) > 0:
                facerect = rects[0]
                imarray = imarray[facerect[1]:facerect[1] + facerect[3], facerect[0]:facerect[0] + facerect[2]]

            imarray = cv2.resize(imarray, (size_input_data[0], size_input_data[1]))
            features.append(imarray)
            label = img[3:5]  # each name of image have 2 char for label from index 3-5
            labels[counter] = expres_code.index(label)
            names = img[0:2]
            img_names.append(names)
            counter += 1
    img_data = np.array(features)
    img_data_preprocessing = preprocessing(img_data)
    return img_data_preprocessing, labels

def preprocessing(input_images):
    normalized_feature_vector_array = []
    for gray in input_images:
        left_eye, rigth_eye = detect_eyes(gray)

        angle = angle_line_x_axis(left_eye, rigth_eye)
        rotated_img = rotateImage(gray, angle)

        # line length
        D = cv2.norm(np.array(left_eye) - np.array(rigth_eye))

        # center of the line
        D_point = [(left_eye[0] + rigth_eye[0]) / 2, (left_eye[1] + rigth_eye[1]) / 2]

        # Face ROI
        x_point = int(D_point[0] - (0.9 * D))
        y_point = int(D_point[1] - (0.6 * D))
        width_point = int(1.8 * D)
        height_point = int(2.2 * D)
        r = [x_point, y_point, width_point, height_point]
        face_roi = rotated_img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        # resize to (96, 128)
        face_roi = cv2.resize(face_roi, (96, 96))
        face_roi = cv2.equalizeHist(face_roi)
        face_roi = np.reshape(face_roi, (96, 96, 1))

        # Pass through encoder and resize
        # feature_vector = featuremodel_AECNN.predict(face_roi)

        # Reshape feature vector
        shape = face_roi.shape
        aux = 1
        for i in shape:
            aux *= i
        feature_vector = face_roi.reshape(aux)
        normalized_feature_vector_array.append(face_roi)
    normalized_feature_vector_array = np.array(normalized_feature_vector_array)
    return normalized_feature_vector_array

def detect_eyes(gray):
    rects = detector(gray, 1)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        pts_right = shape[36: 42]  # right eye landmarks
        pts_left = shape[42: 48]  # left eye landmarks

        hull_right = cv2.convexHull(pts_right)
        M_right = cv2.moments(hull_right)
        # calculate x,y coordinate of center
        cX_right = int(M_right["m10"] / M_right["m00"])
        cY_right = int(M_right["m01"] / M_right["m00"])
        right_eye_center = (cX_right, cY_right)

        hull_left = cv2.convexHull(pts_left)
        M_left = cv2.moments(hull_left)
        # calculate x,y coordinate of center
        cX_left = int(M_left["m10"] / M_left["m00"])
        cY_left = int(M_left["m01"] / M_left["m00"])
        left_eye_center = (cX_left, cY_left)

    return left_eye_center, right_eye_center

def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def angle_line_x_axis(point1, point2):
    angle_r = math.atan2(point1[1] - point2[1], point1[0] - point2[0]);
    angle_degree = angle_r * 180 / math.pi;
    return angle_degree

X_target, Y_target = load_labeled_database_jaffe(labeled_path)
np.save('y.npy', Y_target)

quant_representation_path = "./temp_autoencoder/50 REP"

#LOAD THE GENERATED AUTOENCODERS AND DOES FEATURE BUILDING OF LABELED DATA

arquivosJSON = glob.glob(quant_representation_path + "/*.json")
arquivosH5 = glob.glob(quant_representation_path + "/*.h5")

size = round(len(arquivosJSON))
for index, arq in enumerate(arquivosJSON):
        for i in range(size):
            filename = "%s.json" % i
            filename_h = "%s.h5" % i

            # load json and create model
            json_file = open(arq, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(arquivosH5[index])
            print("Loaded model from disk")

            loaded_model.summary()

            # hidden_layer
            tamanho = len(loaded_model.layers)
            k = 0
            for layer in loaded_model.layers:
                if layer.name == 'hidden_layer':
                    print(layer.name)
                    k = k + 1
                    break
                k = k + 1
            pop = tamanho - k
            sliced_loaded_model = Sequential(loaded_model.layers[:pop])

            sliced_loaded_model.summary()

            x_target = sliced_loaded_model.predict(X_target)
            techs = os.path.basename(arq).split('.')[0]
            if not os.path.exists('./temp2/JAFFE-CK/' + str(size) + ' REP/' + techs):
                os.makedirs('./temp2/JAFFE-CK/' + str(size) + ' REP/' + techs)

            number = i
            np.save('./temp2/JAFFE-CK/' + str(size) + ' REP/' + techs + '/' + "images_%s" % number, x_target)

#CLASSIFICATION
quant_representation_path = "./temp2/JAFFE-CK/"
for quant_reprs in os.listdir(os.path.join(quant_representation_path)):
    for techique_rep in os.listdir(os.path.join(quant_representation_path, quant_reprs)):
        NE = 100
        PB = 1.0
        Lclf = []
        Lpb = []
        matriz_soma = []
        acc_soma = []
        acc_produto = []
        acc_oraculo = []
        base_teste = []
        base_validacao = []
        base_treino = []
        y_teste = []
        y_validacao = []
        y_treino = []

        subjects = []
        subject_index = 0

        rng = np.random.RandomState(42)
        models = []
        Lpb = []
        Lpb_knora = []

        acc_soma_dinamic = []
        acc_produto_dinamic = []
        acc_oraculo_dinamic = []
        acc_stacked = []
        acc_stacked_knora = []
        accsoma = 0
        np.random.seed(42)
        data = []

        #LOAD THE REPRESENTATIONS
        dir_repr = quant_representation_path + '/' + quant_reprs + '/' + techique_rep
        data_dir_list = os.listdir(dir_repr)
        for repr in data_dir_list:
            dta = np.load(dir_repr + '/' + repr)
            data.append(dta)
            print(repr)

        #LOAD THE LABELS FILE
        y = np.load('y.npy')
        y = y.reshape(-1)

        NC = 150
        LX = []
        for i in range(0, len(data)):
            pca = PCA(n_components=NC)
            X = pca.fit(data[i]).transform(data[i])
            LX.append(X)

        base = SVC(C=1e-6, kernel="linear", probability=True, class_weight='balanced')
        # base = DecisionTreeClassifier(max_depth=10, class_weight='balanced')
        #base = OneVsRestClassifier(SVC(C=1e-6, kernel="linear", probability=True, class_weight='balanced'))

        # clf = BaggingClassifier(base_estimator=base, n_estimators=NE, max_samples=PB, random_state=42)
        clf = RandomForestClassifier(n_estimators=NE, max_depth=10)
        # clf = OneVsRestClassifier(SVC(C=1e-6, kernel="linear", probability=True, class_weight='balanced'))


        for i in range(0, len(data)):
            Lclf.append(clf)


        classifier_staked = LogisticRegression(solver='lbfgs', class_weight='balanced', C=0.1)
        classifier_staked_knora = LogisticRegression(solver='lbfgs', class_weight='balanced', C=0.1)

        for participant in os.listdir(os.path.join(labeled_path)):
            subjects.append(subject_index)
            for sequence in os.listdir(os.path.join(labeled_path, participant)):
                if sequence != ".DS_Store":
                    subject_index += 1

        loso = np.zeros((len(data), len(subjects)))
        loso_results_soma = np.zeros((len(data), len(subjects)))
        loso_results = []
        loso_dinamic = np.zeros((len(data), len(subjects)))
        loso_results_soma_dinamic = np.zeros((len(data), len(subjects)))
        predicted_staked = []
        predicted_produto = []
        predicted_soma = []
        predicted_staked_knora = []
        predicted_produto_knora = []
        predicted_soma_knora = []

        for i, subject in enumerate(subjects):
            print("Subject:", i)
            X_train = []
            y_train = []
            X_test = []
            y_test = []
            Lpb = []
            Lpb_train = []
            Lpb_train_knora = []
            Lpb_knora = []
            Lpb_decision = []
            LXval = [] #X - VALIDACAO
            Lyval = [] # Y - VALIDAÇÃO
            LXtrain = [] #X TREINAMENTO
            Lytrain = [] #Y TREINAMENTO

            stack_train = None
            stack_test = None
            stack_train_knora = None
            stack_test_knora = None

            for j in range(0, len(data)):
                if i == len(subjects) - 1:
                    X_train.append(LX[j][0:subject])
                    y_train.append(y[0:subject])
                    X_test.append(LX[j][subject:])
                    y_test.append(y[subject:])
                else:
                    length = subjects[i + 1] - subjects[i]
                    X_train.append(np.vstack((LX[j][0:subject], LX[j][subject + length:])))
                    y_train.append(np.hstack((y[0:subject], y[subject + length:])))
                    X_test.append(LX[j][subject:subject + length])
                    y_test.append(y[subject:subject + length])

            for k in range(0, len(data)):
                #DIVIDE A BASE DE TREINAMENTO (9 SUJEITOS) EM TREINAMENTO / VALIDAÇÃO PARA KU
                X_train_cf, X_val, y_train_cf, y_val = train_test_split(X_train[k], y_train[k], test_size=0.2,
                                                                        stratify=y_train[k],
                                                                        random_state=42)
                LXval.append(X_val)
                Lyval.append(y_val)

                LXtrain.append(X_train_cf)
                Lytrain.append(y_train_cf)

                Lclf[k].fit(X_train_cf, y_train_cf)
                pb = Lclf[k].predict_proba(X_test[k])
                Lpb.append(pb)
                loso[int(k), int(i)] = Lclf[k].score(X_test[k], y_test[k])

                pb_train = Lclf[k].predict_proba(X_train_cf)
                Lpb_train.append(pb_train)

                knorau = KNORAU(Lclf[k], random_state=rng)
                knorau.fit(X_val, y_val)

                pb_knora = knorau.predict_proba(X_test[k])

                Lpb_knora.append(pb_knora)
                loso_dinamic[int(k), int(i)] = knorau.score(X_test[k], y_test[k])

                pb_train_knora = knorau.predict_proba(X_val)
                Lpb_train_knora.append(pb_train_knora)

            cc_oraculo = 0
            ee_oraculo = 0

            base_teste.append(X_test)
            y_teste.append(y_test)
            base_validacao.append(LXval)
            y_validacao.append(Lyval)
            base_treino.append(X_train)
            y_treino.append(y_train)
            #predictproba.append(Lpb)

            soma = np.sum([Lpb], axis=1)
            soma = soma.reshape(soma.shape[1], soma.shape[2])
            predicted_ensemble = np.argmax(soma, axis=1)
            predicted_soma.append(predicted_ensemble)

            cc = 0
            ee = 0
            for m in range(0, soma.shape[0]):
                b = predicted_ensemble[m]
                c = y_test[0][m]
                if (b == c):
                    cc = cc + 1
                else:
                    ee = ee + 1
            accsoma = cc + accsoma
            # ('Acurácia Soma: ', cc / (cc + ee))
            acc_soma.append(cc / (cc + ee))

            soma_dinamic = np.sum([Lpb_knora], axis=1)
            soma_dinamic = soma_dinamic.reshape(soma_dinamic.shape[1], soma_dinamic.shape[2])
            predicted_ensemble_soma = np.argmax(soma_dinamic, axis=1)
            predicted_soma_knora.append(predicted_ensemble_soma)

            cc = 0
            ee = 0
            for m in range(0, soma_dinamic.shape[0]):
                b = predicted_ensemble_soma[m]
                c = y_test[0][m]
                if (b == c):
                    cc = cc + 1
                else:
                    ee = ee + 1
            print('Acurácia Soma KNORA: ', cc / (cc + ee))
            acc_soma_dinamic.append(cc / (cc + ee))

            prod = np.product([Lpb], axis=1)
            prod = prod.reshape(prod.shape[1], prod.shape[2])
            predicted_ensemble_product = np.argmax(prod, axis=1)
            predicted_produto.append(predicted_ensemble_product)

            cc = 0
            ee = 0
            for m in range(0, prod.shape[0]):
                b = predicted_ensemble_product[m]
                c = y_test[0][m]
                if (b == c):
                    cc = cc + 1
                else:
                    ee = ee + 1
            print('Acurácia Produto: ', cc / (cc + ee))
            acc_produto.append(cc / (cc + ee))

            prod_dinamic = np.product([Lpb_knora], axis=1)
            prod_dinamic = prod_dinamic.reshape(prod_dinamic.shape[1], prod_dinamic.shape[2])
            predicted_ensemble_product_dinamic = np.argmax(prod_dinamic, axis=1)
            predicted_produto_knora.append(predicted_ensemble_product_dinamic)

            cc = 0
            ee = 0
            for m in range(0, prod_dinamic.shape[0]):
                b = predicted_ensemble_product_dinamic[m]
                c = y_test[0][m]
                if (b == c):
                    cc = cc + 1
                else:
                    ee = ee + 1
            print('Acurácia Produto KNORA: ', cc / (cc + ee))
            acc_produto_dinamic.append(cc / (cc + ee))

            for m in range(0, len(data)):
                if stack_train is None:
                    stack_train = Lpb_train[m]
                    stack_test = Lpb[m]
                else:
                    stack_train = np.dstack((stack_train, Lpb_train[m]))
                    stack_test = np.dstack((stack_test, Lpb[m]))

            stack_train = stack_train.reshape((stack_train.shape[0], stack_train.shape[1] * stack_train.shape[2]))
            stack_test = stack_test.reshape((stack_test.shape[0], stack_test.shape[1] * stack_test.shape[2]))
            classifier_staked.fit(stack_train, y_train_cf)
            predictions_staked = classifier_staked.predict(stack_test)
            predicted_staked.append(predictions_staked)

            cc = 0
            ee = 0
            for m in range(0, len(y_test[0])):
                b = predictions_staked[m]
                c = y_test[0][m]
                if (b == c):
                    cc = cc + 1
                else:
                    ee = ee + 1
            print('Acurácia STACKED: ', cc / (cc + ee))
            acc_stacked.append(cc / (cc + ee))

            for m in range(0, len(data)):
                if stack_train_knora is None:
                    stack_train_knora = Lpb_train_knora[m]
                    stack_test_knora = Lpb_knora[m]
                else:
                    stack_train_knora = np.dstack((stack_train_knora, Lpb_train_knora[m]))
                    stack_test_knora = np.dstack((stack_test_knora, Lpb_knora[m]))

            stack_train_knora = stack_train_knora.reshape(
                (stack_train_knora.shape[0], stack_train_knora.shape[1] * stack_train_knora.shape[2]))
            stack_test_knora = stack_test_knora.reshape(
                (stack_test_knora.shape[0], stack_test_knora.shape[1] * stack_test_knora.shape[2]))
            classifier_staked_knora.fit(stack_train_knora, y_val)
            predictions_staked_knora = classifier_staked_knora.predict(stack_test_knora)
            predicted_staked_knora.append(predictions_staked_knora)

            cc = 0
            ee = 0
            for m in range(0, len(y_test[0])):
                b = predictions_staked_knora[m]
                c = y_test[0][m]
                if (b == c):
                    cc = cc + 1
                else:
                    ee = ee + 1
            print('Acurácia STACKED KNORA: ', cc / (cc + ee))
            acc_stacked_knora.append(cc / (cc + ee))

            print("======================================================")

        print("============RESULTADOS====================")

        print("CLASSIFICADORES")
        std_dev = []
        sum_loso = loso.sum(axis=1)
        for i in range(0, len(data)):
            res = sum_loso[i] / len(subjects)
            print("Classifier ", [i])
            res1 = sum_loso[i] / len(subjects) * 100
            print(res1)
            std_dev.append(res1)

        x_loso = np.std(std_dev)
        print("STD DEV:", x_loso)

        std_dev_knora = []
        sum_loso = loso_dinamic.sum(axis=1)
        for i in range(0, len(data)):
            res = sum_loso[i] / len(subjects)
            print("Classifier KNORA ", [i])
            res2 = sum_loso[i] / len(subjects) * 100
            print(res2)
            std_dev_knora.append(res2)

        x_loso_knora = np.std(std_dev_knora)
        print("STD DEV KNORA:", x_loso_knora)

        # AUC SOMA
        for j in range(len(subjects)):
            if j == 0:
                concatena = predicted_soma[j]
                concatena_y = y_teste[j][0]
            else:
                result = np.concatenate((concatena, predicted_soma[j]), axis=None)
                result_teste = np.concatenate((concatena_y, y_teste[j][0]), axis=None)
                concatena = result
                concatena_y = result_teste

        c = confusion_matrix(concatena_y, concatena)
        print(c)

        cc = 0
        ee = 0
        for m in range(0, len(concatena)):
            b = concatena[m]
            c = concatena_y[m]
            if (b == c):
                cc = cc + 1
            else:
                ee = ee + 1
        res_soma = cc / (cc + ee)

        # AUC PRODUTO
        for j in range(len(subjects)):
            if j == 0:
                concatena = predicted_produto[j]
                concatena_y = y_teste[j][0]
            else:
                result = np.concatenate((concatena, predicted_produto[j]), axis=None)
                result_teste = np.concatenate((concatena_y, y_teste[j][0]), axis=None)
                concatena = result
                concatena_y = result_teste

        c = confusion_matrix(concatena_y, concatena)
        print(c)

        cc = 0
        ee = 0
        for m in range(0, len(concatena)):
            b = concatena[m]
            c = concatena_y[m]
            if (b == c):
                cc = cc + 1
            else:
                ee = ee + 1
        res_produto = cc / (cc + ee)

        # AUC STACKING
        for j in range(len(subjects)):

            if j == 0:
                concatena = predicted_staked[j]
                concatena_y = y_teste[j][0]
            else:
                result = np.concatenate((concatena, predicted_staked[j]), axis=None)
                result_teste = np.concatenate((concatena_y, y_teste[j][0]), axis=None)
                concatena = result
                concatena_y = result_teste
        c = confusion_matrix(concatena_y, concatena)
        print(c)

        cc = 0
        ee = 0
        for m in range(0, len(concatena)):
            b = concatena[m]
            c = concatena_y[m]
            if (b == c):
                cc = cc + 1
            else:
                ee = ee + 1
        res_stacking = cc / (cc + ee)

        print("==============================================================")
        print("Acurácia Soma:", res_soma)
        print("Acurácia Produto:", res_produto)
        print("Acurácia STAKED", res_stacking)
        print("==============================================================")

        # AUC SOMA KNORA
        for j in range(len(subjects)):
            if j == 0:
                concatena = predicted_soma_knora[j]
                concatena_y = y_teste[j][0]
            else:
                result = np.concatenate((concatena, predicted_soma_knora[j]), axis=None)
                result_teste = np.concatenate((concatena_y, y_teste[j][0]), axis=None)
                concatena = result
                concatena_y = result_teste
        c = confusion_matrix(concatena_y, concatena)
        print(c)

        cc = 0
        ee = 0
        for m in range(0, len(concatena)):
            b = concatena[m]
            c = concatena_y[m]
            if (b == c):
                cc = cc + 1
            else:
                ee = ee + 1
        res_soma_knora = cc / (cc + ee)

        # AUC PRODUTO KNORA
        for j in range(len(subjects)):

            if j == 0:
                concatena = predicted_produto_knora[j]
                concatena_y = y_teste[j][0]
            else:
                result = np.concatenate((concatena, predicted_produto_knora[j]), axis=None)
                result_teste = np.concatenate((concatena_y, y_teste[j][0]), axis=None)
                concatena = result
                concatena_y = result_teste
        c = confusion_matrix(concatena_y, concatena)
        print(c)

        cc = 0
        ee = 0
        for m in range(0, len(concatena)):
            b = concatena[m]
            c = concatena_y[m]
            if (b == c):
                cc = cc + 1
            else:
                ee = ee + 1
        res_produto_knora = cc / (cc + ee)

        # AUC STACKING
        for j in range(len(subjects)):

            if j == 0:
                concatena = predicted_staked_knora[j]
                concatena_y = y_teste[j][0]
            else:
                result = np.concatenate((concatena, predicted_staked_knora[j]), axis=None)
                result_teste = np.concatenate((concatena_y, y_teste[j][0]), axis=None)
                concatena = result
                concatena_y = result_teste
        c = confusion_matrix(concatena_y, concatena)
        print(c)

        cc = 0
        ee = 0
        for m in range(0, len(concatena)):
            b = concatena[m]
            c = concatena_y[m]
            if (b == c):
                cc = cc + 1
            else:
                ee = ee + 1
        res_stacking_knora = cc / (cc + ee)

        print("==============================================================")
        print("Acurácia Soma:", res_soma_knora)
        print("Acurácia Produto:", res_produto_knora)
        print("Acurácia STAKED", res_stacking_knora)
        print("==============================================================")