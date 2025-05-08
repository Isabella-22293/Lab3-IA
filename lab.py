import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocesamiento con NLTK
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def nltk_preprocess_text(text: str) -> str:
    text = text.lower()  # minúsculas
    tokens = word_tokenize(text)  # tokenización
    tokens = [word for word in tokens if word.isalnum()]  # quitar puntuación
    tokens = [word for word in tokens if word not in stop_words]  # quitar stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # lematización
    return ' '.join(tokens)

# Normalización de etiqueta
def clean_label(label: str) -> str:
    lab = str(label).strip().replace('"', '').lower()
    return 'spam' if 'spam' in lab else 'ham'

# Funciones de plotting
def plot_text_count(total: int):
    plt.figure(figsize=(6,4))
    plt.text(0.5, 0.5, f'Total de mensajes:\n{total}',
             ha='center', va='center', fontsize=18)
    plt.title('Cantidad total de mensajes')
    plt.axis('off')
    plt.show()
    plt.close()

def plot_bar(items, counts, title, xlabel='', ylabel='Cantidad', rotate=False):
    plt.figure(figsize=(8,4))
    plt.bar(items, counts)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if rotate:
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_pie(counts, title):
    plt.figure(figsize=(6,6))
    counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title(title)
    plt.ylabel('')
    plt.show()
    plt.close()

def plot_kde(series, title, xlabel=''):
    plt.figure(figsize=(6,4))
    series.plot.kde()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.show()
    plt.close()

def plot_wordcloud(text, title, stopwords_set):
    wc = WordCloud(width=800, height=400, background_color='white',
                   stopwords=stopwords_set, random_state=42).generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()
    plt.close()

# Top N palabras
def top_n_words(corpus, n=20, stop_words='english'):
    vec = CountVectorizer(stop_words=stop_words, max_features=5000)
    bag = vec.fit_transform(corpus)
    sums = bag.sum(axis=0)
    freqs = [(word, sums[0, idx]) for word, idx in vec.vocabulary_.items()]
    freqs = sorted(freqs, key=lambda x: x[1], reverse=True)
    return zip(*freqs[:n])

# Carga y limpieza de datos
df = pd.read_csv('spam_ham.csv', sep=';', encoding='latin-1', engine='python')

# Limpieza de etiquetas
df['Label'] = df['Label'].apply(clean_label)

# Preprocesamiento con NLTK
df['nltk_cleaned'] = df['SMS_TEXT'].astype(str).apply(nltk_preprocess_text)
df['length'] = df['nltk_cleaned'].str.len()

# Muestreo reproducible
sample = df['SMS_TEXT'].sample(5, random_state=42)
print("5 mensajes aleatorios:\n", sample.to_string(index=False))

# Cantidad total
plot_text_count(len(df))

# Distribución y proporción spam vs ham
counts = df['Label'].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(12,5))
counts.plot(kind='bar', ax=axes[0], title='Distribución spam vs ham')
counts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', title='Proporción spam vs ham')
axes[1].set_ylabel('')
plt.tight_layout()
plt.show()
plt.close()

# Densidad de longitud por clase
df_spam = df[df['Label']=='spam']
df_ham  = df[df['Label']=='ham']
plot_kde(df_spam['length'], 'Densidad longitud - Spam', 'Longitud')
plot_kde(df_ham['length'],  'Densidad longitud - Ham',  'Longitud')

# Top 20 palabras
words_spam, freqs_spam = top_n_words(df_spam['nltk_cleaned'], n=20, stop_words='english')
plot_bar(words_spam, freqs_spam, 'Top 20 palabras - Spam', rotate=True)

words_ham, freqs_ham = top_n_words(df_ham['nltk_cleaned'], n=20, stop_words='english')
plot_bar(words_ham, freqs_ham, 'Top 20 palabras - Ham', rotate=True)

# WordCloud para cada clase
text_spam_nltk = ' '.join(df_spam['nltk_cleaned'])
text_ham_nltk  = ' '.join(df_ham['nltk_cleaned'])

plot_wordcloud(text_spam_nltk, 'WordCloud NLTK - Spam', set())
plot_wordcloud(text_ham_nltk,  'WordCloud NLTK - Ham',  set())
