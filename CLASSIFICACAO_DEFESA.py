import numpy as np
import os
import cv2
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from deslib.des import KNORAU
from numpy.random import seed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import tensorflow as tf
#from tensorflow.keras.models import Sequential, model_from_json
import dlib
from imutils import face_utils
import math
import time
from sklearn.decomposition import PCA
import pandas
import glob
import random
#pylint: skip-file

labeled_path = r"L:/cesar/Dataset"
size_input_data = [96, 96, 1]

def load_yildirim(dir_path):
    images = []
    size_input_data = [96, 96, 1]
    image_files = glob.glob(f"{dir_path}/*.png", recursive=True)
    labels = [os.path.basename(os.path.dirname(file)) for file in image_files ]

    for image_file in image_files:
        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (size_input_data[0], size_input_data[1]))
        img = np.reshape(img, (size_input_data[0], size_input_data[1], size_input_data[2]))
        images.append(img)

    loaded_images = np.array(images)
    loaded_images = loaded_images.astype('float32') / 255.0
    labels = np.array(labels)
    return loaded_images, labels

def load_labeled_database(dir_path):
    train_x, train_y = load_yildirim(f"{dir_path}/Train")
    test_x, test_y = load_yildirim(f"{dir_path}/Test")

    return train_x, train_y, test_x, test_y

def split_train_test_by_number_of_autoencoders(number_of_autoencoders):
    all_types_of_files = {
        'kidney_train' : load_yildirim(f'{labeled_path}/Train/Kidney_stone'),
        'kidney_test' : load_yildirim(f'{labeled_path}/Test/Kidney_stone'),
        'normal_train' : load_yildirim(f'{labeled_path}/Train/Normal'),
        'normal_test' : load_yildirim(f'{labeled_path}/Test/Normal')
    }

    all_types_of_files_splitted = {
        'kidney_train' : [[],[]],
        'kidney_test' : [[],[]],
        'normal_train' : [[],[]],
        'normal_test' : [[],[]]
    }

    for key in all_types_of_files:
        amount_of_samples = len(all_types_of_files[key][0])
        desired_amount_of_samples = amount_of_samples // number_of_autoencoders
        all_types_of_files_splitted[key][0] = np.array_split(all_types_of_files[key][0], desired_amount_of_samples)
        all_types_of_files_splitted[key][1] = np.array_split(all_types_of_files[key][1], desired_amount_of_samples)
    return all_types_of_files, all_types_of_files_splitted

def combine_images_and_labels(images, labels):
    sample = list(zip(images, labels))
    return sample

def create_sample(sample):
    np.random.shuffle(sample)
    return list(zip(*sample))[0], list(zip(*sample))[1]




train_x, train_y, test_x, test_y = load_labeled_database(labeled_path)
dataset_complete, dataset_splitted = split_train_test_by_number_of_autoencoders(10)

label_and_image_train_stone = combine_images_and_labels(dataset_complete['kidney_train'][0], dataset_complete['kidney_train'][1])
label_and_image_train_normal = combine_images_and_labels(dataset_complete['normal_train'][0], dataset_complete['normal_train'][1])
label_and_image_test_stone = combine_images_and_labels(dataset_complete['kidney_test'][0], dataset_complete['kidney_test'][1])
label_and_image_test_normal = combine_images_and_labels(dataset_complete['normal_test'][0], dataset_complete['normal_test'][1])

train_x, train_y = create_sample(np.concatenate((label_and_image_train_normal, label_and_image_train_stone), axis=0))
test_x, test_y = create_sample(np.concatenate((label_and_image_test_normal, label_and_image_test_stone), axis=0))



train_x = np.concatenate((dataset_complete['kidney_train'][0], dataset_complete['normal_train'][0]), axis=0)
train_y = np.concatenate((dataset_complete['kidney_train'][1], dataset_complete['normal_train'][1]), axis=0)

test_x = np.concatenate((dataset_complete['kidney_test'][0], dataset_complete['normal_test'][0]), axis=0)
test_y = np.concatenate((dataset_complete['kidney_test'][1], dataset_complete['normal_test'][1]), axis=0)

print(
    f"Shapes: train_x = {train_x.shape}, train_y = {train_y.shape}, test_x = {test_x.shape}, test_y = {test_y.shape}")

# # np.save('Y_train.npy', train_x)
# # np.save('Y_test.npy', test_y)

np.save('all_labels', np.concatenate((train_y, test_y), axis=0))

np.save('Y_train.npy', train_y)
np.save('Y_test.npy', test_y)

quant_representation_path = r"L:/cesar/temp_autoencoder/10 REP"

# #LOAD THE GENERATED AUTOENCODERS AND DOES FEATURE BUILDING OF LABELED DATA

arquivosJSON = glob.glob(quant_representation_path + "/*.json")
arquivosH5 = glob.glob(quant_representation_path + "/*.h5")
#reprs in os.listdir(quant_representation_path):
    #for techs in os.listdir(os.path.join(quant_representation_path + reprs)):
        #for qtd_reprs in os.listdir(os.path.join(quant_representation_path + reprs +'/'+ techs +'/')):
            #size = int(len(os.listdir(os.path.join(quant_representation_path + reprs +'/'+ techs +'/'))) / 2)
size = round(len(arquivosJSON))
for index, arq in enumerate(arquivosJSON):
        for i in range(size):
            filename = "%s.json" % i
            filename_h = "%s.h5" % i

            #load json and create model
            json_file = open(arq, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = tf.keras.models.model_from_json(loaded_model_json)
            #load weights into new model
            loaded_model.load_weights(arquivosH5[index])
            print("Loaded model from disk")

            loaded_model.summary()

            #hidden_layer
            tamanho = len(loaded_model.layers)
            k = 0
            for layer in loaded_model.layers:
                if layer.name == 'hidden_layer':
                    print(layer.name)
                    k = k + 1
                    break
                k = k + 1
            pop = tamanho - k
            sliced_loaded_model = tf.keras.models.Sequential(loaded_model.layers[:pop])

            sliced_loaded_model.summary()

            x_target = sliced_loaded_model.predict(np.array(train_x))
            techs = os.path.basename(arq).split('.')[0]
            if not os.path.exists('./temp2/JAFFE-CK/' + str(size) + ' REP/' + techs):
                os.makedirs('./temp2/JAFFE-CK/' + str(size) + ' REP/' + techs)

            number = i
            np.save('./temp2/JAFFE-CK/' + str(size) + ' REP/' + techs + '/' + "images_%s" % number, x_target)

            
#######################################################################################
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
            print(dta)
            data.append(dta)
            print(repr)

        #LOAD THE LABELS FILE
        y_train_loaded = np.load('Y_train.npy').reshape(-1)
        y_test_loaded = np.load('Y_test.npy').reshape(-1)

        y_all_labels = np.load('all_labels.npy').reshape(-1)


        NC = 150
        LX = []
        for i in range(0, len(data)):
            pca = PCA(n_components=NC)
            X = pca.fit(data[i]).transform(data[i])
            LX.append(X)

        #base = SVC(C=1e-6, kernel="linear", probability=True, class_weight='balanced')
        base = DecisionTreeClassifier(max_depth=10, class_weight='balanced')
        #base = OneVsRestClassifier(SVC(C=1e-6, kernel="linear", probability=True, class_weight='balanced'))

        #clf = BaggingClassifier(base_estimator=base, n_estimators=NE, max_samples=PB, random_state=42)
        clf = RandomForestClassifier(n_estimators=NE, max_depth=10)
        #clf = OneVsRestClassifier(SVC(C=1e-6, kernel="linear", probability=True, class_weight='balanced'))


        for i in range(0, len(data)):
            Lclf.append(clf)


        classifier_staked = LogisticRegression(solver='lbfgs', class_weight='balanced', C=0.1)
        classifier_staked_knora = LogisticRegression(solver='lbfgs', class_weight='balanced', C=0.1)

        #for participant in os.listdir(os.path.join(labeled_path)):
             #subjects.append(subject_index)
             #for sequence in os.listdir(os.path.join(labeled_path, participant)):
                 #if sequence != ".DS_Store":
                     #subject_index += 1

        for folder in glob.glob(f'{labeled_path}/**/**'):
            subjects.append(len(glob.glob(f'{folder}/*.png')))

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

################ ARRUMAR AQUI #######################################
            for j in range(0, len(data)):
                if i == len(subjects) - 1:
                    X_train.append(LX[j][0:subject])
                    y_train.append(y_train_loaded[0:subject])
                    X_test.append(LX[j][subject:])
                    y_test.append(y_test_loaded[subject:])
                else:
                    length = subjects[i + 1] - subjects[i]
                    X_train.append(np.vstack((LX[j][0:subject], LX[j][subject + length:])))
                    y_train.append(np.hstack((y_train_loaded[0:subject], y_train_loaded[subject + length:])))
                    X_test.append(LX[j][subject:subject + length])
                    y_test.append(y_test_loaded[subject:subject + length])
###################################################################################

            for k in range(0, len(data)):
                #DIVIDE A BASE DE TREINAMENTO (9 SUJEITOS) EM TREINAMENTO / VALIDAÇÃO PARA KU
                X_train_cf, X_val, y_train_cf, y_val = train_test_split(X_train[k], y_train[k], test_size=0.2,
                                                                         stratify=y_train[k], random_state=42)
                print(f"\nInicio do loop: k = {k}")
                #X_val = X_test[k]
                #y_val = y_test[k]

                #X_train_cf = X_train[k]
                #y_train_cf = y_train[k]

                print(
                    f"Shapes: X_train_cf = {X_train_cf.shape}, y_train_cf = {y_train_cf.shape}, X_val = {X_val.shape}, y_val = {y_val.shape}")

                LXval.append(X_val)
                Lyval.append(y_val)

                LXtrain.append(X_train_cf)
                Lytrain.append(y_train_cf)

                Lclf[k].fit(X_train_cf, y_train_cf)
                pb = Lclf[k].predict_proba(X_test[k])
                print(f"Shape de pb: {pb.shape}")
                Lpb.append(pb)
                loso[int(k), int(i)] = Lclf[k].score(X_test[k], y_test[k])

                pb_train = Lclf[k].predict_proba(X_train_cf)
                Lpb_train.append(pb_train)

                print(f"\nAntes do KNORAU: ")
                print(
                    f"Shapes: X_train_cf = {X_train_cf.shape}, y_train_cf = {y_train_cf.shape}, X_val = {X_val.shape}, y_val = {y_val.shape}")

                knorau = KNORAU(Lclf[k], random_state=rng)
                knorau.fit(X_val, y_val)

                print(f"Shape de X_test[k]: {X_test[k].shape}")

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
            # predictproba.append(Lpb)

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
            # print('Acurácia Soma KNORA: ', cc / (cc + ee))
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
            # print('Acurácia Produto: ', cc / (cc + ee))
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
            # print('Acurácia Produto KNORA: ', cc / (cc + ee))
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
            # print('Acurácia STACKED: ', cc / (cc + ee))
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
            # print('Acurácia STACKED KNORA: ', cc / (cc + ee))
            acc_stacked_knora.append(cc / (cc + ee))

            # print("======================================================")

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
        # print(c)

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
        # print(c)

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
        # print(c)

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