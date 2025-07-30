from datasets import load_dataset
# facilitar leitura de texto para máquina, transformando texto em vetor numérico.
from sklearn.feature_extraction.text import TfidfVectorizer
# modelo utilizado para modelo de classificação binária.
from sklearn.linear_model import LogisticRegression
# retorna aconfusão de matriz e desempenho do modelo
from sklearn.metrics import classification_report, confusion_matrix
# lib pd para manipular dataframe.
import pandas as pd
# numpy pra retornar alguns coeficientes da regressão
import numpy as np


# Carregando os dados
dataset = load_dataset("imdb")

# Convertendo para DataFrame para facilitar
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

# Separando features (textos) e labels (0 = neg, 1 = pos)
X_train = train_df['text']
y_train = train_df['label']
X_test = test_df['text']
y_test = test_df['label']

# Transformando texto em valores vetoriais em até 10000 palavras
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Treinando o modelo, usando classificação binária
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Fazendo previsões
y_pred = model.predict(X_test_vec)

# Métricas de classificação:
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# Retorno de matriz de confusão:
print(confusion_matrix(y_test, y_pred))

# Pega os nomes das palavras
words = np.array(vectorizer.get_feature_names_out())

coef = model.coef_[0]

# Classifica as palavras
top_positive_words = words[np.argsort(coef)[-10:]]
top_negative_words = words[np.argsort(coef)[:10]]
print("positivas:", top_positive_words)
print("negativas:", top_negative_words)
