import pandas as pd
import re
import string
from collections import defaultdict, Counter
import math
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Cargar dataset
df = pd.read_csv("spam_ham.csv", sep=';', encoding='latin-1')
df.columns = ['label', 'message']
df = df.dropna(subset=['message'])  # eliminar mensajes vacíos

def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'\d+', '', texto)
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    palabras = texto.split()
    palabras = [p for p in palabras if p not in stopwords.words('english')]
    return palabras

df['tokens'] = df['message'].apply(limpiar_texto)


# Calcular frecuencias
spam_tokens = []
ham_tokens = []

for _, row in df.iterrows():
    if row['label'] == 'spam':
        spam_tokens.extend(row['tokens'])
    else:
        ham_tokens.extend(row['tokens'])

spam_count = Counter(spam_tokens)
ham_count = Counter(ham_tokens)

# Vocabulario
vocabulario = set(spam_count.keys()).union(set(ham_count.keys()))
V = len(vocabulario)

# Probabilidades base
total_spam = df[df['label'] == 'spam'].shape[0]
total_ham = df[df['label'] == 'ham'].shape[0]
total = total_spam + total_ham

P_spam = total_spam / total
P_ham = total_ham / total

# Calcular P(w|spam) y P(w|ham) con Laplace
def calcular_probabilidad(palabra, clase_count, total_palabras_clase):
    return (clase_count[palabra] + 1) / (total_palabras_clase + V)

# Total de palabras
total_spam_words = sum(spam_count.values())
total_ham_words = sum(ham_count.values())

def clasificar_mensaje(mensaje):
    palabras = limpiar_texto(mensaje)
    log_prob_spam = math.log(P_spam)
    log_prob_ham = math.log(P_ham)

    predictivas = []

    for palabra in palabras:
        p_w_spam = calcular_probabilidad(palabra, spam_count, total_spam_words)
        p_w_ham = calcular_probabilidad(palabra, ham_count, total_ham_words)

        log_prob_spam += math.log(p_w_spam)
        log_prob_ham += math.log(p_w_ham)
        predictivas.append((palabra, p_w_spam))

    prob_spam_final = 1 / (1 + math.exp(log_prob_ham - log_prob_spam))  # sigmoid

    # Resultado
    print(f"\nMensaje: {mensaje}")
    print(f"Probabilidad de SPAM: {prob_spam_final:.4f}")

    # Top 3 palabras predictivas de SPAM
    predictivas.sort(key=lambda x: x[1], reverse=True)
    print("Palabras más predictivas de SPAM:")
    for palabra, p in predictivas[:3]:
        print(f"  - '{palabra}': P(w|spam) = {p:.6f}")

# Consola
while True:
    texto = input("\nEscribe un mensaje para clasificar (o 'salir'): ")
    if texto.lower() == "salir":
        break
    clasificar_mensaje(texto)
