"""
Arquivo para classificação de imagens de pedras nos rins e imagens normais.
"""
# pylint: disable=line-too-long,unused-import,unused-variable,no-member,too-many-locals,invalid-name
import os
from typing import Union
import glob

import numpy as np

import cv2

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import shutil
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

import matplotlib.pyplot as plt
import tensorflow as tf
import keras

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

def load_yildirim(dir_path: str) -> Union[np.array, np.array]:
    """
    Carrega as imagens da base de dados do Yildirim.

    Args:
        dir_path (str): Caminho onde estão as imagens e as labels.

    Returns:
        Union[np.array, np.array]: retorna uma tupla onde o primeiro elemento é um array com as imagens e o segundo
        é um array com as labels.
    """
    images = []
    size_input_data = [224, 224, 1]
    image_files = glob.glob(f"{dir_path}/*.png", recursive=True)
    labels = [os.path.basename(os.path.dirname(file)) for file in image_files]

    for image_file in image_files:
        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        #show_images([img], titles='Original')
        img = cv2.resize(img, (size_input_data[0], size_input_data[1]))
        #show_images([img], titles='resize')
        img = np.reshape(img, (size_input_data[0], size_input_data[1], size_input_data[2]))
        #show_images([img], titles='reshape')
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



def carregar_representacoes(path: str):
    """Carrega as representações geradas anteriormente para uso nesta sessão

    Args:
        path (str): caminho onde estão as representações

    Returns:
        List[np.ndarray]: lista de representações carregadas
    """
    # Ordena os diretórios para garantir consistência
    autoencoder_dirs = sorted(glob.glob(f'{path}/**/*.npy'))
    representacoes = []
    for rep_path in autoencoder_dirs:
        rep = np.load(rep_path)
        representacoes.append(rep)
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

def treinar_classificador_com_gridsearch(representation: np.array, labels: np.array, param_grid: dict):
    """
    Trains RandomForestClassifier using GridSearchCV to find the best hyperparameters.

    Returns:
        tuple: (best_estimator, best_params)
    """
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )
    grid_search.fit(representation, labels)
    print(f"Best hyperparameters: {grid_search.best_params_}")
    return grid_search.best_estimator_, grid_search.best_params_

def limpar_diretorio(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def create_mlp_model(input_shape, num_classes):
    model = tf.keras.models.Sequential()

    # Camada de entrada
    model.add(tf.keras.layers.Input(shape=input_shape))
    model.add(tf.keras.layers.Flatten(input_shape=(224, 224, 3)))
    model.add(tf.keras.layers.Dense(units=1000, activation='relu'))
    # Primeira camada oculta
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))

    # Segunda camada oculta
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))

    # Terceira camada oculta
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))

    # Camada de saída
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    return model


# Compilar e treinar o modelo como mostrado anteriormente


def main(number_of_representations: int = 5) -> None:
    n_splits = 5  # Number of folds
    quant_representation_path = rf".\temp_autoencoder\{number_of_representations} REP"

    dataset_complete = split_train_test_by_number_of_autoencoders()

    # Combine training data
    train_x = np.concatenate((dataset_complete['kidney_train'][0], dataset_complete['normal_train'][0]), axis=0)
    train_y = np.concatenate((dataset_complete['kidney_train'][1], dataset_complete['normal_train'][1]), axis=0)

    # Test data remains separate
    test_x = np.concatenate((dataset_complete['kidney_test'][0], dataset_complete['normal_test'][0]), axis=0)
    test_y = np.concatenate((dataset_complete['kidney_test'][1], dataset_complete['normal_test'][1]), axis=0)

    # Shuffle data
    train_x, train_y = shuffle(train_x, train_y, random_state=42)
    test_x, test_y = shuffle(test_x, test_y, random_state=42)

    # Generate representations for the entire training data
    gerar_representacoes_base_atraves_de_kyoto(quant_representation_path, train_x, r"./representations_train_full/")
    representations_train_full = carregar_representacoes(r"./representations_train_full/")

    # Generate representations for the test data
    gerar_representacoes_base_atraves_de_kyoto(quant_representation_path, test_x, r"./representations_test/")
    representations_test = carregar_representacoes(r"./representations_test/")

    #Initialize cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    acuracias = []
    relatorios_classificacao = []
    matrizes_confusao = []
    best_params_list = []

    for fold, (train_index, val_index) in enumerate(skf.split(train_x, train_y)):
        print(f"\nFold {fold + 1}/{n_splits}")

        # Use indices to select subsets from the full representations
        representations_train = [rep[train_index] for rep in representations_train_full]
        representations_val = [rep[val_index] for rep in representations_train_full]

        labels_train = train_y[train_index]
        labels_val = train_y[val_index]

        # Define the parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt'],
            'class_weight': ['balanced']
        }

        # Train classifiers with GridSearchCV
        classifiers = []
        fold_best_params = []

        for representation in representations_train:
            classifier, best_params = treinar_classificador_com_gridsearch(representation, labels_train, param_grid)
            classifiers.append(classifier)
            fold_best_params.append(best_params)

        best_params_list.append(fold_best_params)

        # Make predictions on validation set
        predicoes = [predizer_classificacao(classifier, representation) for classifier, representation in
                     zip(classifiers, representations_val)]

        # Majority voting
        predicao_final = voto_majoritario(predicoes)

        # Evaluate the model
        acuracia = calcular_acuracia(predicao_final, labels_val)
        relatorio = classification_report(labels_val, predicao_final, output_dict=True)
        matriz_confusao = calcular_matriz_de_confusao(predicao_final, labels_val)

        # Store the metrics
        acuracias.append(acuracia)
        relatorios_classificacao.append(relatorio)
        matrizes_confusao.append(matriz_confusao)

        print(acuracias)
        print(relatorios_classificacao)
        print(matrizes_confusao)
        # Optional: Clear temporary representations if need
        # limpar_diretorio(r"./representations_train_full/")
        # limpar_diretorio(r"./representations_test/")

    # Retrain classifiers on full training data using the best parameters from the last fold
    final_classifiers = []

    # Use the best parameters from the last fold
    last_fold_best_params = best_params_list[-1]

    for representation, best_params in zip(representations_train_full, last_fold_best_params):
        classifier = RandomForestClassifier(**best_params, random_state=42)
        classifier.fit(representation, train_y)
        final_classifiers.append(classifier)

    # Predict on test data
    predicoes_teste = [predizer_classificacao(classifier, representation) for classifier, representation in
                       zip(final_classifiers, representations_test)]

    # Majority voting on test predictions
    predicao_final_teste = voto_majoritario(predicoes_teste)

    # Evaluate on test data
    acuracia_teste = calcular_acuracia(predicao_final_teste, test_y)
    relatorio_teste = classification_report(test_y, predicao_final_teste)
    matriz_confusao_teste = calcular_matriz_de_confusao(predicao_final_teste, test_y)

    print("\nResultados no conjunto de teste:")
    print(relatorio_teste)
    print(f"Acurácia no conjunto de teste: {acuracia_teste}")
    print(f"Matriz de confusão no conjunto de teste:\n{matriz_confusao_teste}")

    # Average cross-validation results
    acuracia_media = np.mean(acuracias)
    desvio_padrao = np.std(acuracias)
    print(f"\nAcurácia média na validação cruzada: {acuracia_media:.4f} ± {desvio_padrao:.4f}")

    # Load the data

    # Concatenar as representações
    train_x_combined = np.concatenate(representations_train_full, axis=1)
    test_x_combined = np.concatenate(representations_test, axis=1)

    train_x_combined, train_y = shuffle(train_x_combined, train_y, random_state=42)
    test_x_combined, test_y = shuffle(test_x_combined, test_y, random_state=42)

    # Definir o número de classes
    num_classes = 2  # Ajuste conforme o número de classes no seu problema

    # Criar o modelo MLP
    mlp = create_mlp_model(input_shape=(200,), num_classes=num_classes)

    # Compilar o modelo
    mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Codificar as labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(train_y)
    y_test_encoded = label_encoder.transform(test_y)


    scaler = StandardScaler()
    train_x_combined = scaler.fit_transform(train_x_combined)
    test_x_combined = scaler.transform(test_x_combined)

    selector = SelectKBest(score_func=f_classif, k=200)
    X_train_selected = selector.fit_transform(train_x_combined, y_train_encoded)
    X_test_selected = selector.transform(test_x_combined)

    # Imprimir exemplos de labels antes e depois da codificação
    print(f"Exemplos de y_train antes da codificação: {train_y[:5]}")
    print(f"Exemplos de y_train após codificação: {y_train_encoded[:5]}")

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
    class_weights = dict(enumerate(class_weights))


    # Treinar o modelo
    mlp.fit(X_train_selected, y_train_encoded, epochs=50, batch_size=32, class_weight=class_weights, callbacks=[early_stopping], validation_split=0.3)

    # Fazer predições no conjunto de teste
    predicoes_mlp = mlp.predict(X_test_selected)
    predicoes_mlp = np.argmax(predicoes_mlp, axis=1)

    # Avaliar o modelo no conjunto de teste
    acuracia_mlp = accuracy_score(y_test_encoded, predicoes_mlp)
    relatorio_mlp = classification_report(y_test_encoded, predicoes_mlp)
    matriz_confusao_mlp = confusion_matrix(y_test_encoded, predicoes_mlp)

    print("\nResultados MLP no conjunto de teste:")
    print(relatorio_mlp)
    print(f"Acurácia MLP no conjunto de teste: {acuracia_mlp}")
    print(f"Matriz de confusão MLP no conjunto de teste:\n{matriz_confusao_mlp}")

    print("\nResultados no conjunto de teste:")
    print(relatorio_teste)
    print(f"Acurácia no conjunto de teste: {acuracia_teste}")
    print(f"Matriz de confusão no conjunto de teste:\n{matriz_confusao_teste}")

    # Average cross-validation results
    acuracia_media = np.mean(acuracias)
    desvio_padrao = np.std(acuracias)
    print(f"\nAcurácia média na validação cruzada: {acuracia_media:.4f} ± {desvio_padrao:.4f}")



if __name__ == '__main__':
    main()