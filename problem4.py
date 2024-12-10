import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA


df = pd.read_csv("csv/PGR210_NLP_data1.csv", index_col=0)
df.index.name = "index"
print(df.head())
print(df.info())

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Text Preprocessing

def clean_text(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum() and word not in stop_words]
    cleaned_text = " ".join(words)
    return cleaned_text

df["cleaned"] = df["text"].apply(clean_text)
print(df[["text", "cleaned"]].head())

# TF Model
x_tf = CountVectorizer().fit_transform(df["cleaned"])
print(x_tf.shape)

# TD-IDF Model
x_tfidf = TfidfVectorizer().fit_transform(df["cleaned"])
print(x_tfidf.shape)

vectorizer = CountVectorizer(max_features=1000, stop_words='english')
text_matrix = vectorizer.fit_transform(df['cleaned'])

# Fit LDA model
lda = LDA(n_components=5, random_state=42)
lda.fit(text_matrix)

# Get the feature names (terms)
terms = vectorizer.get_feature_names_out()
terms = list(terms)

# Display topics with top words
for idx, topic in enumerate(lda.components_):
    print(f"Topic {idx + 1}:")
    print(" ".join([terms[i] for i in topic.argsort()[:-10 - 1:-1]]))  # Top 10 words per topic

from sklearn.decomposition import TruncatedSVD

# Fit Truncated SVD model on the TF-IDF matrix (number of topics = 5)
svd = TruncatedSVD(n_components=5, random_state=42)
svd.fit(x_tfidf)

# Print the top words for each topic
for topic_idx, topic in enumerate(svd.components_):
    print(f"Topic {topic_idx + 1}:")
    print(" ".join([terms[i] for i in topic.argsort()[:-10 - 1:-1]]))  # Top 10 words per topic
    print("\n")