from GENERATE_REPRESENTATIONS_DEFESA import Representations
import CLASSIFICACAO_DEFESA
import cv2
import numpy as np
import os
from sys import exit
import gc
import time
import subprocess

unlabeled_path = './bases-tcc/images/'

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def create_results_folder():
    os.makedirs('resultados', exist_ok=True)
    os.makedirs('resultados/mlp', exist_ok=True)
    os.makedirs('resultados/rf', exist_ok=True)
    os.makedirs('resultados/svm', exist_ok=True)

def load_unlabeled_database(dir_path):
    size_input_data = [224, 224, 1]
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
                input_img = cv2.resize(input_img, (size_input_data[0], size_input_data[1]))
                input_img = np.reshape(input_img, (size_input_data[0], size_input_data[1], size_input_data[2]))
                img_data_list.append(input_img)

    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32') / 255.0
    n_train = int(img_data.shape[0] * 0.7)
    trainX, testX = img_data[:n_train, :], img_data[n_train:, :]

    return trainX, testX


if __name__ == '__main__':
    # # create_results_folder()
    # # carregar base nao rotulada
    # # for i in [5, 10, 15, 25, 50]:
    # #     trainx, testx = load_unlabeled_database(unlabeled_path)
    # #     teste = Representations(trainx, testx)
    # #     teste.Generate_all(epochs=5, seeds_rep=True, arch_rep=True, hidden_rep=False, number_of_repr=i)
    # #     CLASSIFICACAO_DEFESA.main(i, 'rf')
    # #     time.sleep(300)
    
    # # for i in [5,10,15, 25, 50]:
    # #     time.sleep(300)
    # #     trainx, testx = load_unlabeled_database(unlabeled_path)
    # #     teste = Representations(trainx, testx)
    # #     teste.Generate_all(epochs=5, seeds_rep=True, arch_rep=True, hidden_rep=False, number_of_repr=i)
    # #     CLASSIFICACAO_DEFESA.main(i, 'svm')
    # #     gc.collect()
    # #     time.sleep(300)
        

    # for i in [50]:
    #     trainx, testx = load_unlabeled_database(unlabeled_path)
    #     teste = Representations(trainx, testx)
    #     teste.Generate_all(epochs=5, seeds_rep=True, arch_rep=True, hidden_rep=False, number_of_repr=i)
    #     CLASSIFICACAO_DEFESA.main(i, 'mlp')
    #     gc.collect()
    #     time.sleep(300)

    try:
        subprocess.run(["git", "add", "."], check=True)
        print("Arquivos adicionados com sucesso.")
                
        # Faz o commit com a mensagem especificada
        commit_message = "results(CVRemoval): upload dos resultados"
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        print(f"Commit realizado com a mensagem: '{commit_message}'")

        # Realiza o push para o branch 'CVRemoval'
        subprocess.run(["git", "push", "origin", "CVRemoval"], check=True)
        print("Push realizado com sucesso para o branch 'CVRemoval'.")
    except subprocess.CalledProcessError as e:
        print(f"Ocorreu um erro ao executar o comando: {e.cmd}")
        print(f"O retorno foi: {e.returncode}")
        print(f"O output foi: {e.output}")
    finally:
        os.system('shutdown /s /t 20')


# from itertools import product

# # Supondo que load_unlabeled_database e Representations já estejam implementados corretamente
# trainx, testx = load_unlabeled_database(unlabeled_path)

# # Todas as combinações possíveis de parâmetros booleanos
# combinations = list(product([True, False], repeat=3))  # Gera todas as combinações de True/False para 3 parâmetros

# # Itera sobre todas as combinações e executa o método Generate_all e a função CLASSIFICACAO_DEFESA.main
# for seeds_rep, arch_rep, hidden_rep in combinations:
#     print(f"Executando com seeds_rep={seeds_rep}, arch_rep={arch_rep}, hidden_rep={hidden_rep}")
#     teste = Representations(trainx, testx)
#     teste.Generate_all(epochs=5, seeds_rep=seeds_rep, arch_rep=arch_rep, hidden_rep=hidden_rep, number_of_repr=5)
#     CLASSIFICACAO_DEFESA.main(5, 'rf', seeds_rep=seeds_rep, arch_rep=arch_rep, hidden_rep=hidden_rep)


