"""
Arquivo para classificação de imagens de pedras nos rins e imagens normais.
"""
# pylint: disable=line-too-long,unused-import,unused-variable,no-member,too-many-locals,invalid-name
import os
from typing import Union
import glob

import numpy as np

import cv2
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tensorflow as tf

from scipy.stats import mode


LABELED_PATH = r".\bases-tcc\Dataset"

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


def show_images(images, titles=None):
    """Função para mostrar uma lista de imagens."""
    n = len(images)
    plt.figure(figsize=(12, 6))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i], cmap='gray')
        if titles is not None:
            plt.title(titles[i])
        plt.axis('off')
    plt.show()

def aumentar_contraste(img: np.ndarray) -> np.ndarray:
    """Aumenta o contraste da imagem usando equalização de histograma."""
    return cv2.equalizeHist(img)

def load_yildirim(dir_path: str) -> Union[np.array, np.array]:
    images = []
    size_input_data = [224, 224, 1]
    image_files = glob.glob(f"{dir_path}/*.png", recursive=True)
    labels = [os.path.basename(os.path.dirname(file)) for file in image_files]

    for image_file in image_files:
        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (size_input_data[0], size_input_data[1]))
        img = aumentar_contraste(img)

        img = np.reshape(img, (size_input_data[0], size_input_data[1], size_input_data[2]))
        # show_images([img], 'reshape')
        images.append(img)

    loaded_images = np.array(images)
    loaded_images = loaded_images.astype('float32') / 255.0
    labels = np.array(labels)
    return loaded_images, labels



def load_labeled_database(dir_path: str) -> Union[np.array, np.array, np.array, np.array]:
    """
    Faz o cerregamento das bases de dados de treino e teste.

    Args:
        dir_path (str): Caminho onde estão as imagens e as labels.

    Returns:
        Union[np.array, np.array, np.array, np.array]: Retorna uma tupla com os dados de treino e teste
    """

    train_x, train_y = load_yildirim(f"{dir_path}/Train")
    test_x, test_y = load_yildirim(f"{dir_path}/Test")
    return train_x, train_y, test_x, test_y


def split_train_test_by_number_of_autoencoders() -> dict:
    """
    Faz a separação da base de dados para cada autoencoder utilizar corretamente

    Args:
        number_of_autoencoders (int): a quantidade de autoencoders que serão utilizados

    Returns:
        Union[dict, dict]: Retorna um dicionário com os dados completos e um dicionário com os dados separados
    """
    all_types_of_files = {
        'kidney_train': load_yildirim(f'{LABELED_PATH}/Train/Kidney_stone'),
        'kidney_test': load_yildirim(f'{LABELED_PATH}/Test/Kidney_stone'),
        'normal_train': load_yildirim(f'{LABELED_PATH}/Train/Normal'),
        'normal_test': load_yildirim(f'{LABELED_PATH}/Test/Normal')
    }
    return all_types_of_files


def combine_images_and_labels(images: np.ndarray, labels: np.ndarray) -> list:
    """
    Combina as imagens e as labels em uma lista.

    Args:
        images (np.ndarray): ndarray com as imagens
        labels (np.ndarray): ndarray com as labels

    Returns:
        list: Retorna uma lista de tuplas com o par imagem e label
    """
    sample = list(zip(images, labels))
    return sample


def create_sample(sample: list) -> tuple:
    """Separa o par imagem e label em duas listas diferentes.

    Args:
        sample (list): lista com par imagem e label

    Returns:
        tuple: Retorna uma tupla com duas listas, a primeira contendo as imagens e a segunda contendo as labels
    """

    return list(zip(*sample))[0], list(zip(*sample))[1]


def carregar_modelo_do_json(arquivo_json: str) -> tf.keras.models.Model:
    """Realiza o corregamento do modelo a partir de um arquivo json.

    Args:
        arquivo_json (str): caminho do arquivo json

    Returns:
        tf.keras.models.Model: Retorna o modelo carregado a partir do arquivo json com a base kyoto
    """
    with open(arquivo_json, 'r', encoding='utf-8') as json_file:
        loaded_model_json = json_file.read()
        loaded_model = tf.keras.models.model_from_json(loaded_model_json)
        return loaded_model


def carregar_pesos_no_modelo(modelo: tf.keras.models.Model, nome: int) -> tf.keras.models.Model:
    """
    Carrega os pesos no modelo.

    Args:
        modelo (tf.keras.models.Model): modelo carregado a partir do arquivo json
        nome (int): peso a carregar no modelo

    Returns:
        tf.keras.models.Model: Retorna o modelo com os pesos carregados
    """
    modelo.load_weights(f"{nome}")
    return modelo


def extracao_camada_oculta(modelo: tf.keras.models.Model) -> tf.keras.models.Sequential:
    """
    Extrai a camada oculta do modelo, descartando a camada de decode do autoencoder.

    Args:
        modelo (tf.keras.models.Model): modelo carregado a partir do arquivo json

    Returns:
        tf.keras.models.Sequential: retorna toda a camada input e a camada oculta do modelo
                                    retirando a camada de decode
    """
    k = 0
    for index, layer in enumerate(modelo.layers):
        if layer.name == 'hidden_layer':
            k = index + 1
            print(modelo.summary())
    return tf.keras.models.Sequential(modelo.layers[:k])


def representacao_por_camada_oculta(modelo: tf.keras.models.Sequential, dados: np.ndarray) -> np.ndarray:
    """
    Gera a representação do modelo utilizando as camadas de input e camada oculta para possuir um vetor latente.

    Args:
        modelo (tf.keras.models.Sequential): modelo com as camadas de input e camada oculta
        dados (np.ndarray): dados de treino

    Returns:
        np.ndarray: retorna a representação do modelo
    """
    # print(dados)
    print("Forma dos dados:", dados.shape)  # Deve corresponder ao esperado pelo modelo
    return modelo.predict(dados)


def criacao_pastas(path: str) -> None:
    """
    Cria as pastas caso elas não existam.

    Args:
        path (str): caminho para criar a pasta
    """
    if not os.path.exists(path):
        os.makedirs(path)

def gerar_representacoes_base_atraves_de_kyoto(representations_path: str, dados: np.ndarray, caminho_resultado: str) -> None:
    """Realiza a geração das representações utilizando a base

    Args:
        representations_path (str): caminho com as representações dos autoencoders
        dados (np.ndarray): dados para gerar as representações das imagens que serão utilizadas
        caminho_resultado (str): caminho onde serão salvas as representações geradas
    """
    arquivos_json = glob.glob(representations_path + "/*.json")
    arquivos_h5 = glob.glob(representations_path + "/*.h5")
    size = len(arquivos_json)  # O número de autoencoders
    criacao_pastas(caminho_resultado)

    for index, arq in enumerate(arquivos_json):
        # Carregar o modelo e os pesos uma vez para cada autoencoder
        modelo = carregar_modelo_do_json(arq)
        modelo = carregar_pesos_no_modelo(modelo, arquivos_h5[index])
        camada_oculta = extracao_camada_oculta(modelo)

        # Criar a pasta para o autoencoder específico
        pasta_autoencoder = f'{caminho_resultado}/{index}'
        criacao_pastas(pasta_autoencoder)

        # Salvar a representação apenas uma vez por autoencoder
        np.save(f'{pasta_autoencoder}/images',
                representacao_por_camada_oculta(camada_oculta, dados))



def carregar_representacoes(path: str) -> np.array:
    """Carrega as representações geradas anteriormente para uso nesta sessão

    Args:
        path (str): caminho onde estão as representações

    Returns:
        np.array: representações carregadas

    Yields:
        Iterator[np.array]: cada representação carregadas
    """
    representacoes = glob.glob(f'{path}/**/*.npy')
    representacoes = [np.load(representacao) for representacao in representacoes]
    return representacoes


def carrega_etiquetas(path: str) -> np.array:
    """Carrega as etiquetas de um arquivo npy

    Args:
        path (str): caminho do arquivo npy

    Returns:
        np.array: etiquetas carregadas
    """
    return np.load(path)


def usar_pca_na_representacao(representation: np.array, n_components: int = 150) -> np.array:
    """Realiza a redução de dimensionalidade utilizando o PCA

    Args:
        representation (np.array): representação a ser reduzida
        n_components (int, optional): quantidade de componentes. padrão é 150.

    Returns:
        np.array: retorna a representação reduzida
    """
    pca = PCA(n_components=n_components)
    return pca.fit(representation).transform(representation)


def treinar_classificador(representation: np.array, labels: np.array, classifier: RandomForestClassifier | SVC) -> RandomForestClassifier | SVC:
    """Treina o classificador

    Args:
        representation (np.array): representação para treinar o classificador
        labels (np.array): etiquetas com a classe de cada representação
        classifier (RandomForestClassifier|SVC): classificador a ser treinado

    Returns:
        RandomForestClassifier | SVC: retorna o classificador treinado
    """
    classifier.fit(representation, labels)
    return classifier


def predizer_classificacao(classifier: RandomForestClassifier | SVC, test_x: np.array) -> np.array:
    """Realiza a predição da classificação

    Args:
        classifier (RandomForestClassifier | SVC): classificador treinado
        test_x (np.array): dados de teste

    Returns:
        np.array: retorna os resultados das predições
    """
    return classifier.predict(test_x)


def predizer_probabilidade(classifier: RandomForestClassifier | SVC, test_x: np.array) -> np.array:
    """Realiza a predição da probabilidade

    Args:
        classifier (RandomForestClassifier | SVC): classificador treinado
        test_x (np.array): dados de teste

    Returns:
        np.array: retorna as probabilidades das predições
    """
    return classifier.predict_proba(test_x)


def calcular_acuracia(predicoes: np.array, labels: np.array) -> float:
    """Calcula a acurácia

    Args:
        predicoes (np.array): predições do classificador
        labels (np.array): rótulos reais

    Returns:
        float: retorna a acurácia
    """
    certo = 0
    errado = 0
    for i, v in enumerate(predicoes):
        predicao = v
        label = labels[i]
        if predicao == label:
            certo = certo + 1
        else:
            errado = errado + 1
    return certo / (certo + errado)


def calcular_matriz_de_confusao(predicoes: np.array, labels: np.array) -> np.ndarray:
    """Calcula a matriz de confusão

    Args:
        predicoes (np.array): predições do classificador
        labels (np.array): rótulos reais

    Returns:
        np.ndarray: retorna a matriz de confusão
    """
    return confusion_matrix(labels, predicoes)


def calcular_acuracia_media(predicoes: np.array, labels: np.array) -> float:
    """Calcula a acurácia média

    Args:
        predicoes (np.array): predições do classificador
        labels (np.array): rótulos reais

    Returns:
        float: retorna a acurácia média
    """
    return np.mean([calcular_acuracia(predicao, label) for predicao, label in zip(predicoes, labels)])


def voto_majoritario(predicoes: np.array) -> np.array:
    """Realiza a votação majoritária

    Args:
        predicoes (np.array): predições do classificador

    Returns:
        np.array: retorna as predições após a votação majoritária
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


def main(number_of_representations: int = 5) -> None:
    """Realiza todas as chamadas de função para classificação

    Args:
        number_of_representations (int, optional): Quantidade de representações. Defaults to 50.
    """
    dataset_complete = split_train_test_by_number_of_autoencoders()

    train_x = np.concatenate((dataset_complete['kidney_train'][0], dataset_complete['normal_train'][0]), axis=0)
    train_y = np.concatenate((dataset_complete['kidney_train'][1], dataset_complete['normal_train'][1]), axis=0)

    test_x = np.concatenate((dataset_complete['kidney_test'][0], dataset_complete['normal_test'][0]), axis=0)
    test_y = np.concatenate((dataset_complete['kidney_test'][1], dataset_complete['normal_test'][1]), axis=0)

    train_x, train_y = shuffle(train_x, train_y, random_state=42)
    test_x, test_y = shuffle(test_x, test_y, random_state=42)

    # Ver as classes e suas quantidades no conjunto de treinamento
    classes_treino, counts_treino = np.unique(train_y, return_counts=True)
    print(f"Classes de treino: {classes_treino}")
    print(f"Quantidade de amostras em cada classe de treino: {counts_treino}")

    # Ver as classes e suas quantidades no conjunto de teste
    classes_teste, counts_teste = np.unique(test_y, return_counts=True)
    print(f"Classes de teste: {classes_teste}")
    print(f"Quantidade de amostras em cada classe de teste: {counts_teste}")

    print(
        f"Shapes: train_x = {train_x.shape},"
        f" train_y = {train_y.shape}, "
        f"test_x = {test_x.shape}, test_y = {test_y.shape}")

    np.save('Y_train.npy', train_y)
    np.save('Y_test.npy', test_y)

    quant_representation_path = rf".\temp_autoencoder\{number_of_representations} REP"

    gerar_representacoes_base_atraves_de_kyoto(quant_representation_path, train_x, r"./representations_train/")
    gerar_representacoes_base_atraves_de_kyoto(quant_representation_path, test_x, r"./representations_test/")

    representations_train = carregar_representacoes(r"./representations_train/")
    representations_test = carregar_representacoes(r"./representations_test/")
    labels_train = carrega_etiquetas('Y_train.npy')
    labels_test = carrega_etiquetas('Y_test.npy')

    classifiers = [SVC(C=1.0, kernel="sigmoid", probability=False, class_weight='balanced', random_state=42) for _ in range(50)]
    #classifiers = [RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42) for _ in range(number_of_representations)]

    representations_train = [usar_pca_na_representacao(representation) for representation in representations_train]
    representations_test = [usar_pca_na_representacao(representation) for representation in representations_test]
    classifiers = [treinar_classificador(representation, labels_train, classifier) for representation, classifier in
                   zip(representations_train, classifiers)]

    predicoes = [predizer_classificacao(classifier, representation) for representation, classifier in
                 zip(representations_test, classifiers)]

    # Ver as classes previstas pelos classificadores
    for i, classifier in enumerate(classifiers):
        print(f"Classes previstas pelo classificador {i + 1}: {classifier.classes_}")

    # Realiza a votação majoritária
    predicao_final = voto_majoritario(predicoes)

    # Calcula a acurácia final após a votação majoritária
    acuracia_final = calcular_acuracia(predicao_final, labels_test)
    matriz_de_confusao_final = calcular_matriz_de_confusao(predicao_final, labels_test)

    print(f"Acurácia final: {acuracia_final}")
    print(f"Matriz de confusão final:\n{matriz_de_confusao_final}")


if __name__ == '__main__':
    main()