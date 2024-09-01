<p align="center">
  <img src="https://sites.pucpr.br/enade/wp-content/uploads/sites/20/2021/06/logo-pucpr.png"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"/>
  <img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white"/>
  <img src="https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white"/>
  <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black"/>
</p>


# Projeto de Classificação de Imagens com Redes Neurais e Autoencoders

Este projeto realiza a classificação de imagens utilizando redes neurais com autoencoders para extração de representações latentes e classificação com algoritmos de aprendizado de máquina.

## Estrutura do Projeto

- `MAIN_NOVO_DEFESA.py`: Arquivo principal para a execução do projeto. Carrega a base de dados, gera representações utilizando autoencoders e realiza a classificação das imagens.
- `GENERATE_REPRESENTATIONS_DEFESA.py`: Contém a classe `Representations`, responsável por gerar as representações latentes das imagens usando autoencoders.
- `CLASSIFICACAO_DEFESA.py`: Realiza a classificação das imagens com base nas representações geradas, utilizando classificadores como Random Forest e SVM.

## Dependências

Este projeto utiliza as seguintes bibliotecas:

- `opencv-contrib-python==4.8.1.78`
- `scikit-learn==1.4.2`
- `numpy==1.23.5`
- `tensorflow==2.12.0`
- `imutils==0.5.4`
- `keras==2.12.0`

Todas as dependências podem ser instaladas utilizando o arquivo `requirements.txt`.

### Instalando Dependências

```bash
pip install -r requirements.txt
```

## Criando e Utilizando um Ambiente Virtual (venv)

### 1. Criando o Ambiente Virtual

Navegue até a pasta do seu projeto no terminal ou prompt de comando e crie um novo ambiente virtual:

```bash
python -m venv venv
```

### 2. Ativando o Ambiente Virtual

- **Windows:**

  ```bash
  venv\Scripts\activate
  ```

- **macOS e Linux:**

  ```bash
  source venv/bin/activate
  ```

### 3. Instalando as Dependências

Com o ambiente virtual ativado, instale as dependências listadas no arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Desativando o Ambiente Virtual

Quando terminar de trabalhar no projeto, desative o ambiente virtual com o seguinte comando:

```bash
deactivate
```

## Clonando o Repositório

Para clonar este repositório com o submódulo incluído, utilize o comando:

```bash
git clone --recurse-submodules <URL_DO_REPOSITORIO>
```

Caso já tenha clonado o repositório sem o submódulo, você pode inicializá-lo com os seguintes comandos:

```bash
git submodule init
git submodule update
```

### Atualizando Submódulos

Sempre que quiser garantir que o submódulo esteja atualizado, utilize:

```bash
git submodule update --remote
```

## Executando o Projeto

1. **Inicialize o [Ambiente Virtual](#2-ativando-o-ambiente-virtual)**
   - O ambiente terá todas as bibliotecas já carregadas nele para uso futuro

2. **Execute o Script Principal:**
   - O script `MAIN_NOVO_DEFESA.py` irá carregar os dados, gerar as representações, e classificar as imagens.

```bash
python MAIN_NOVO_DEFESA.py
```

## Observações

- Certifique-se de que todas as pastas necessárias para armazenar as representações e os modelos estejam criadas antes de executar os scripts, ou o próprio script criará as pastas automaticamente.
- Este projeto foi desenvolvido para classificação de imagens de pedras nos rins e imagens normais, mas pode ser adaptado para outras bases de dados com a devida modificação.

## Contato

## Contato

Para quaisquer dúvidas ou contribuições, sinta-se à vontade para abrir uma issue ou entrar em contato:

- **[Lukas Jacon Barboza](mailto:lukas.barboza@pucpr.edu.br)**
- **[Cesar Cunha Ziobro](mailto:cesar.ziobro@pucpr.edu.br)**
- **[Thiago Krügel](mailto:thiago.krugel@pucpr.edu.br)**


---

Este projeto é parte de uma pesquisa acadêmica e deve ser utilizado para fins educacionais.
