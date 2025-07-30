# scikitlearn-imdb
Usa regressão de logística para fazer classificação de palavras ruins e palavras boas do site de IMDB.

# Classificador de palavras do site do IMDb

Este projeto é um classificador de palavras utilizando o dataset **IMDb**, implementado em Python com as bibliotecas pandas, scikit-learn e datasets.
Eu treinei um modelo de **classificação binária** (positivo vs negativo) para avaliar a performance de um modelo de regressão logística com vetorização TF-IDF (de texto para numeração vetorial)

- Fonte: [Hugging Face – IMDB](https://huggingface.co/datasets/imdb)
- 25.000 amostras para treino
- 25.000 amostras para teste

- Vetorizador: TfidfVectorizer
- Modelo: Regressão de logística
- Métricas: Acurácia, precisão, recall, F1-score e matriz de confusão

- Python
- Pandas
- Scikit-learn
- Hugging Face Datasets

Eu quis fazer o cálculo na mão msm pra estudar para um concurso público, mas tá ai o código com as métricas de classificações prontas:
# Métricas de classificação:
# print("Classification Report:")
# print(classification_report(y_test, y_pred))
