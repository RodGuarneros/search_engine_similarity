# Función elaborada por Rodrigo Guarneros Gutiérrez
# 22.05.2023

# Dependencias
import json
import pandas as pd
from collections import Counter
import pandas as pd
import seaborn as sn
import unicodedata
import math
import seaborn as sns
import numpy as np
import nltk
import string
import pickle
from scipy.stats import rankdata
from nltk.corpus import stopwords
import random
import itertools
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import time
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ipywidgets import interact
from IPython.display import display
from ipywidgets import widgets
from nltk.stem import PorterStemmer
from scipy import sparse
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

nltk.download('punkt')
nltk.download('universal_tagset')
nltk.download('tagsets')
nltk.download('cess_esp')
nltk.download('stopwords')


simbolos = {'\n', '`', "'", '/', '%', 'ø', '\xad', '+', 'μ', 'æ', 'ß', '_', '·', 'ð', '&', '=', '``', "'", "$", '\\', '.', '(', ')', '--', ':', '``',
            ':', ';', '!', '"', '#', ',', ' ,', ' ,,', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'rt', '-', '--', 'aa', 'aal', 'aamlo'}

stemmer = PorterStemmer()

# Usamos cache para mejor desempeño
cache = {}

def remover_letras_repetidas(text):
    """
    Esta función remueve las letras como siiiiii para que diga si en
    las palabras de los twitts.
    """

    return re.sub(r"(.)\1{2,}", r"\1", text)

def remover_https(text):
    """
    Esta función remueve los términos con https
    las palabras de los twitts pues no generan información
    de búsqueda.
    """    
    return re.sub(r'\S*http\S*', '', text)

def remover_twitter_users(text):
    """
    Esta función remueve los usuarios @usuario
    toda vez que ya tenemos un key en el diccionario
    para ese efecto.
    """    
    return re.sub(r'@\w+', '', text)

def remover_stopwords(tokens):
    """
    Esta función remueve las stop words en español.
    """    
    stop_words = set(nltk.corpus.stopwords.words('spanish'))
    return [word for word in tokens if word not in stop_words and len(word) > 1]

def normalize(text):
    """
    Esta función global incluye las funciones de pre-procesamiento anteriores
    previo a una vectorización de la bolsa de palabras.
    """    
    if text in cache:
        return cache[text]

    conjunto_minusculas = text.lower()
    conjunto_minusculas_simbol = re.sub('(' + '|'.join(re.escape(simbolo) for simbolo in simbolos) + ')', '', conjunto_minusculas, flags=re.IGNORECASE)
    conjunto_minusculas_simbol_no_usuarios = remover_twitter_users(conjunto_minusculas_simbol)
    conjunto_minusculas_simbol_no_usuarios_https = remover_https(conjunto_minusculas_simbol_no_usuarios)
    tokens = nltk.word_tokenize(conjunto_minusculas_simbol_no_usuarios_https)
    conjunto_minusculas_simbol_no_usuarios_token_repeat = remover_letras_repetidas(" ".join(tokens))
    tokens = nltk.word_tokenize(conjunto_minusculas_simbol_no_usuarios_token_repeat)
    tokens = remover_stopwords(tokens)
    conjunto_minusculas_simbol_no_usuarios_token_repeat_token_stop_stemm = [stemmer.stem(word) for word in tokens]

    cache[text] = conjunto_minusculas_simbol_no_usuarios_token_repeat_token_stop_stemm
    return conjunto_minusculas_simbol_no_usuarios_token_repeat_token_stop_stemm

def galloping_search(data, words):
    """
    Esta función aplica el algoritmo de búsqueda
    que de acuerdo con nuestro análisis previo
    es más eficiente y requiere menos comparaciones.
    """    
    keys = sorted(list(data.keys()))
    results = []

    for key in keys:
        if all(word in key for word in words):
            results.append((key, data[key]))

    return results

def preprocess_data():
    """
    Esta función remueve garantiza que nuestro 
    índice invertido, previamente construido 
    sea un diccionario json.
    """    
    # Obtenemos el json con el índice invertido
    with open('indice_invertido_final.json') as file:
        data = json.load(file)

    preprocessed_data = {}
    for word, indices in data.items():
        preprocessed_data[word] = [(index, normalize(word)) for index in indices]

    return preprocessed_data

def search_documents(k):
    """
    Las siguientes dependencias deben estar 
    instaladas en el ambiente virtual previo 
    a la aplicación de esta función:
        import pandas as pd
        import json
        import re
        import nltk
        from nltk.stem import PorterStemmer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from scipy import sparse
        from sklearn.metrics.pairwise import cosine_similarity
        import tkinter as tk
        from tkinter import messagebox
        from tkinter import ttk

    """
    # Obtenemos los datos preprocesados
    data = preprocess_data()

    # normalizamos el query
    query = query_entry.get()

    # Perform the search by matching individual words
    results = []
    for word in normalize(query):
        if word in data:
            results.extend(data[word])

    # Remover los duplicados mientras se preserva el orden.
    seen = set()
    unique_results = []
    for result in results:
        if result[0] not in seen:
            unique_results.append(result)
            seen.add(result[0])

    if len(unique_results) == 0:
        messagebox.showinfo("Búsqueda", "No se encontraron documentos que coincidan con la consulta.")
        return

    # Se obtiene el df de pandas que se creo previamente con todos los comentarios de twitter
    df = pd.read_csv('datos_panda.csv')

    # vectorizador con pesado de Tfidf
    tfidf = TfidfVectorizer()

    # Campo vectorial para todos los documentos en el json
    tfidf_matrix = tfidf.fit_transform(df['text'])

    # Obtener la vectorización del query
    query_vector = tfidf.transform([query])

    # Inicializar la lista para los comentarios, score de similaridad, e índices
    documents = []
    similarity_scores = []

    # Get the indices of documents in unique_results
    document_indices = [index for index, _ in unique_results]

    # Calculate similarity scores for all documents
    similarity_matrix = cosine_similarity(query_vector, tfidf_matrix[document_indices])

    # Iterate over the results
    for index, (_, words) in enumerate(unique_results):
        similarity_score = similarity_matrix[0][index]
        doc_index = document_indices[index]
        documents.append((df.loc[doc_index, 'text'], df.loc[doc_index, 'user'], doc_index))
        similarity_scores.append(similarity_score)

    # ordenamos los comentarios por similaridad de coseno
    sorted_documents = sorted(zip(documents, similarity_scores), key=lambda x: x[1], reverse=True)

    # Obtenemos el twitter, el usuario, el índice, la tasa de seguridad y la similaridad de coseno.
    documents_text = [text for ((text, _, _), _) in sorted_documents]
    documents_users = [user for ((_, user, _), _) in sorted_documents]
    documents_indexes = [index for ((_, _, index), _) in sorted_documents]
    documents_safe = [df.loc[index, 'safe'] if index is not None else None for index in documents_indexes]
    documents_similarity_scores = [score for (_, score) in sorted_documents]

    # Imprimir los comentarios encontrados en orden descendente por similaridad de coseno
    result_text.delete(1.0, tk.END)
    for text, user, index, safe, similarity_score in zip(
        documents_text[:k],
        documents_users[:k],
        documents_indexes[:k],
        documents_safe[:k],
        documents_similarity_scores[:k]
    ):
        result_text.insert(tk.END, '_' * 50 + '\n')
        result_text.insert(tk.END, f"---> (Similaridad de coseno: {similarity_score})\n")
        result_text.insert(tk.END, f"---> Comentario twitter: {text}\n")
        result_text.insert(tk.END, f"---> Usuario: {user}\n")
        result_text.insert(tk.END, f"---> (Score de seguridad: {safe})\n")
        result_text.insert(tk.END, f"---> ID_Comentario: {index}\n") # para estar seguros de que el comentario repetido es otro comentario y no el mismo
        result_text.insert(tk.END, '_' * 50 + '\n')

# Crear la ventana principal
window = tk.Tk()
window.title("Máquina de Búsqueda de Twitters ordenados por Similaridad de Coseno v.1.0. Elaboró Rodrigo Guarneros Gutiérrez ")
window.geometry("1000x1900")

# Etiqueta y campo de entrada para el query
query_label = tk.Label(window, text="Consulta:")
query_label.pack()
query_entry = tk.Entry(window, width=100)
query_entry.pack()

# Slider para seleccionar el número de resultados
k_label = tk.Label(window, text="Elige el número de resultados que deseas:")
k_label.pack()
k_slider = tk.Scale(window, from_=1, to=20, orient=tk.HORIZONTAL)
k_slider.pack()

# Botón para realizar la búsqueda
search_button = tk.Button(window, text="Buscar", command=lambda: search_documents(k_slider.get()))
search_button.pack()

# Resultados de la búsqueda
result_text = tk.Text(window, width=100, height=25)
result_text.pack()

# Ejecutar la ventana principal
window.mainloop()
