# Classificação de Imagens de Pedras nos Rins e Imagens Normais

Este repositório contém um código para a classificação de imagens de pedras nos rins e imagens normais usando técnicas de aprendizado de máquina e redes neurais. O código utiliza autoencoders para extração de características e, posteriormente, aplica classificadores como RandomForest ou SVM para a predição.

## Estrutura do Projeto

- **`LABELED_PATH`**: Caminho para a base de dados rotulada, contendo as imagens de treino e teste.
- **`load_yildirim`**: Função que carrega as imagens da base de dados do Yildirim, redimensiona e normaliza as imagens.
- **`split_train_test_by_number_of_autoencoders`**: Função que separa os dados de treino e teste para diferentes autoencoders.
- **`carregar_modelo_do_json`**: Função para carregar um modelo de autoencoder a partir de um arquivo JSON.
- **`extracao_camada_oculta`**: Função que extrai a camada oculta de um modelo de autoencoder, removendo a camada de decodificação.
- **`representacao_por_camada_oculta`**: Função que gera a representação latente de dados de treino ou teste usando a camada oculta do autoencoder.
- **`gerar_representacoes_base_atraves_de_kyoto`**: Gera representações de base utilizando modelos de autoencoder.
- **`treinar_classificador`**: Função para treinar classificadores como RandomForest ou SVM.
- **`voto_majoritario`**: Função que realiza a votação majoritária entre os classificadores.

## Como Usar

1. **Configuração do Ambiente**:
   - Certifique-se de que todas as dependências estão instaladas:
     - `numpy`
     - `cv2`
     - `scikit-learn`
     - `tensorflow`
     - `scipy`
   
   Instale os pacotes necessários utilizando pip:
   ```bash
   pip install numpy opencv-python scikit-learn tensorflow scipy
   ```

2. **Configuração dos Dados**:
   - Defina o caminho da base de dados no `LABELED_PATH`.
   - As imagens devem estar organizadas em diretórios `Train` e `Test`, contendo subdiretórios `Kidney_stone` e `Normal`.

3. **Treinamento e Classificação**:
   - O script pode ser executado diretamente utilizando:
   ```bash
   python <nome_do_arquivo>.py
   ```
   - O código irá carregar as imagens, treinar os autoencoders, gerar representações latentes, treinar os classificadores e realizar a predição.
   - O resultado final será exibido no console, incluindo a acurácia final e a matriz de confusão.

4. **Modificação dos Classificadores**:
   - O código permite a escolha entre RandomForest e SVM como classificador. Isso pode ser ajustado na função `main()`.

## Comentários de Código

- **Pylint Directives**:
  - `# pylint: disable=line-too-long,unused-import,unused-variable,no-member,too-many-locals,invalid-name`
  - Essas diretivas foram adicionadas para desativar certas verificações do Pylint, garantindo que o código seja analisado sem alertas irrelevantes.

## Contribuições

Contribuições são bem-vindas. Por favor, abra um *pull request* ou *issue* para discussões sobre melhorias ou correções.
