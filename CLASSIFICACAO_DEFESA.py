import numpy as np
import os
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from sklearn.decomposition import PCA
import glob
from scipy.stats import mode

# pylint: skip-file

labeled_path = r"L:\cesar\Dataset"
size_input_data = [96, 96, 1]


def load_yildirim(dir_path):
    images = []
    size_input_data = [96, 96, 1]
    image_files = glob.glob(f"{dir_path}/*.png", recursive=True)
    labels = [os.path.basename(os.path.dirname(file)) for file in image_files]

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
        'kidney_train': load_yildirim(f'{labeled_path}/Train/Kidney_stone'),
        'kidney_test': load_yildirim(f'{labeled_path}/Test/Kidney_stone'),
        'normal_train': load_yildirim(f'{labeled_path}/Train/Normal'),
        'normal_test': load_yildirim(f'{labeled_path}/Test/Normal')
    }

    all_types_of_files_splitted = {
        'kidney_train': [[], []],
        'kidney_test': [[], []],
        'normal_train': [[], []],
        'normal_test': [[], []]
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
    return list(zip(*sample))[0], list(zip(*sample))[1]


def carregar_modelo_do_json(arquivo_json):
    with open(arquivo_json, 'r') as json_file:
        loaded_model_json = json_file.read()
        loaded_model = tf.keras.models.model_from_json(loaded_model_json)
        return loaded_model


def carregar_pesos_no_modelo(modelo, nome: int):
    modelo.load_weights(f"{nome}")
    return modelo


def extracao_camada_oculta(modelo):
    k = 0
    for index, layer in enumerate(modelo.layers):
        if layer.name == 'hidden_layer':
            k = index + 1
    return tf.keras.models.Sequential(modelo.layers[:k])


def representacao_por_camada_oculta(modelo, dados):
    # print(dados)
    return modelo.predict(dados)


def criacao_pastas(path):
    if not os.path.exists(path):
        os.makedirs(path)


def gerar_representacoes_base_atraves_de_kyoto(representations_path, dados_treino, caminho_resultado):
    arquivosJSON = glob.glob(representations_path + "/*.json")
    # print(arquivosJSON)
    arquivosH5 = glob.glob(representations_path + "/*.h5")
    size = round(len(arquivosJSON))
    criacao_pastas(caminho_resultado)
    for index, arq in enumerate(arquivosJSON):
        for i in range(size):
            modelo = carregar_modelo_do_json(arq)
            modelo = carregar_pesos_no_modelo(modelo, arquivosH5[index])
            camada_oculta = extracao_camada_oculta(modelo)
            criacao_pastas(f'{caminho_resultado}{i}')
            np.save(f'{caminho_resultado}{index}/images_{i}',
                    representacao_por_camada_oculta(camada_oculta, dados_treino))


def criacao_classificador():
    svm = SVC(C=1e-6, kernel="linear", probability=True, class_weight='balanced')


def carregar_representacoes(path):
    representacoes = glob.glob(f'{path}/**/*.npy')
    for representacao in representacoes:
        yield np.load(representacao)


def carrega_etiquetas(path):
    return np.load(path)


def usar_PCA_na_representacao(representation, n_components=150) -> np.array:
    pca = PCA(n_components=n_components)
    return pca.fit(representation).transform(representation)


def treinar_classificador(representation, labels, classifier):
    classifier.fit(representation, labels)
    return classifier


def predizer_classificacao(classifier, test_x):
    return classifier.predict(test_x)


def predizer_probabilidade(classifier, test_x):
    return classifier.predict_proba(test_x)


def calcular_acuracia(predicoes, labels):
    certo = 0
    errado = 0
    for i in range(0, len(predicoes)):
        predicao = predicoes[i]
        label = labels[i]
        if predicao == label:
            certo = certo + 1
        else:
            errado = errado + 1
    return certo / (certo + errado)


def calcular_matriz_de_confusao(predicoes, labels):
    return confusion_matrix(labels, predicoes)


def calcular_acuracia_media(predicoes, labels):
    return np.mean([calcular_acuracia(predicao, label) for predicao, label in zip(predicoes, labels)])


def calcular_acuracia_media(predicoes, test_y):
    return np.mean([calcular_acuracia(predicao, test_y) for predicao in predicoes])


def voto_majoritario(predicoes):
    """
    Realiza a votação majoritária entre as previsões de diferentes classificadores.

    :param predicoes: Uma lista de arrays, onde cada array contém as previsões de um classificador.
    :return: Um array contendo as previsões finais após a votação majoritária.
    """
    # Criar mapeamento de classes para inteiros
    classes_unicas = np.unique(predicoes)
    classe_para_int = {classe: i for i, classe in enumerate(classes_unicas)}
    int_para_classe = {i: classe for classe, i in classe_para_int.items()}

    # Converter as predições para inteiros usando o mapeamento
    predicoes_numericas = [[classe_para_int[predicao] for predicao in predicoes_individuais] for predicoes_individuais
                           in predicoes]

    # Transpor e converter para array
    predicoes_transpostas = np.array(predicoes_numericas).T

    # Aplicar o modo para obter a votação majoritária
    predicao_final_numerica, _ = mode(predicoes_transpostas, axis=1)

    # Remapear as predições para os rótulos de classe originais
    predicao_final = [int_para_classe[pred] for pred in predicao_final_numerica.ravel()]

    return predicao_final


def main():
    dataset_complete, dataset_splitted = split_train_test_by_number_of_autoencoders(10)

    train_x = np.concatenate((dataset_complete['kidney_train'][0], dataset_complete['normal_train'][0]), axis=0)
    train_y = np.concatenate((dataset_complete['kidney_train'][1], dataset_complete['normal_train'][1]), axis=0)

    test_x = np.concatenate((dataset_complete['kidney_test'][0], dataset_complete['normal_test'][0]), axis=0)
    test_y = np.concatenate((dataset_complete['kidney_test'][1], dataset_complete['normal_test'][1]), axis=0)

    # Ver as classes e suas quantidades no conjunto de treinamento
    classes_treino, counts_treino = np.unique(train_y, return_counts=True)
    print(f"Classes de treino: {classes_treino}")
    print(f"Quantidade de amostras em cada classe de treino: {counts_treino}")

    # Ver as classes e suas quantidades no conjunto de teste
    classes_teste, counts_teste = np.unique(test_y, return_counts=True)
    print(f"Classes de teste: {classes_teste}")
    print(f"Quantidade de amostras em cada classe de teste: {counts_teste}")

    # Achatar as imagens de 96x96x1 para 9216 (96*96) para que possam ser usadas com classificadores tradicionais
    test_x = test_x.reshape(test_x.shape[0], -1)

    print(
        f"Shapes: train_x = {train_x.shape},"
        f" train_y = {train_y.shape}, "
        f"test_x = {test_x.shape}, test_y = {test_y.shape}")

    # np.save('all_labels', np.concatenate((train_y, test_y), axis=0))

    np.save('Y_train.npy', train_y)
    np.save('Y_test.npy', test_y)

    quant_representation_path = r"L:\cesar\temp_autoencoder\10 REP"

    arquivosJSON = glob.glob(quant_representation_path + "/*.json")
    arquivosH5 = glob.glob(quant_representation_path + "/*.h5")
    size = round(len(arquivosJSON))

    gerar_representacoes_base_atraves_de_kyoto(quant_representation_path, train_x, r"./representations/")

    representations = carregar_representacoes(r"./representations")
    labels = carrega_etiquetas('Y_train.npy')

    # Inicializar e aplicar PCA
    pca = PCA(n_components=150)
    test_x_pca = pca.fit_transform(test_x)

    classifiers = [SVC(C=1.0, kernel="poly", probability=False, class_weight=None, random_state=42) for _ in range(10)]
    # classifiers = [RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42) for _ in range(10)]

    representations = [usar_PCA_na_representacao(representation) for representation in representations]
    classifiers = [treinar_classificador(representation, labels, classifier) for representation, classifier in
                   zip(representations, classifiers)]

    predicoes = [predizer_classificacao(classifier, test_x_pca) for classifier in classifiers]
    # probabilidades = [predizer_probabilidade(classifier, test_x_pca) for classifier in classifiers]

    # Ver as classes previstas pelos classificadores
    for i, classifier in enumerate(classifiers):
        print(f"Classes previstas pelo classificador {i + 1}: {classifier.classes_}")

    acuracias = [calcular_acuracia(predicao, test_y) for predicao in predicoes]
    matrizes_de_confusao = [calcular_matriz_de_confusao(predicao, test_y) for predicao in predicoes]
    acuracia_media = calcular_acuracia_media(predicoes, test_y)

    # Realiza a votação majoritária
    predicao_final = voto_majoritario(predicoes)

    # Calcula a acurácia final após a votação majoritária
    acuracia_final = calcular_acuracia(predicao_final, test_y)
    matriz_de_confusao_final = calcular_matriz_de_confusao(predicao_final, test_y)

    print(f"Acurácia final: {acuracia_final}")
    print(f"Matriz de confusão final:\n{matriz_de_confusao_final}")

    # print(acuracias)
    # print(matrizes_de_confusao)
    # print(acuracia_media)


if __name__ == '__main__':
    main()
