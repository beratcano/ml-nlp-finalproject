import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import TruncatedSVD
from wordcloud import WordCloud

df = pd.read_csv("csv/PGR210_NLP_data1.csv", index_col=0)
df.index.name = "index"
print(df.head())
print(df.info())

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

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

text_matrix = CountVectorizer(max_features=1000, stop_words='english').fit_transform(df['cleaned'])

lda = LDA(n_components=5, random_state=42)
lda.fit(text_matrix)
terms = CountVectorizer(max_features=1000, stop_words='english').get_feature_names_out()
terms = list(terms)

# Display topics with top words
for idx, topic in enumerate(lda.components_):
    print(f"Topic {idx + 1}:")
    print(" ".join([terms[i] for i in topic.argsort()[:-10 - 1:-1]]))  # Top 10 words per topic

x_tfidf = TfidfVectorizer(max_features=1000, stop_words='english').fit_transform(df['cleaned'])
svd = TruncatedSVD(n_components=5, random_state=42)
svd.fit(x_tfidf)
terms = list(TfidfVectorizer(max_features=1000, stop_words='english').get_feature_names_out())

for topic_idx, topic in enumerate(svd.components_):
    print(f"Topic {topic_idx + 1}:")
    print(" ".join([terms[i] for i in topic.argsort()[:-10 - 1:-1]]))  # Top 10 words
    print("\n")

for idx, topic in enumerate(lda.components_):
    wordcloud = WordCloud(background_color='white').generate_from_frequencies(
        {terms[i]: topic[i] for i in topic.argsort()[:-10 - 1:-1]}
    )
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"LDA Topic {idx + 1}")
    plt.show()

# Visualize SVD topics as WordClouds
for topic_idx, topic in enumerate(svd.components_):
    # Generate a WordCloud for each topic
    wordcloud = WordCloud(background_color='white').generate_from_frequencies(
        {terms[i]: topic[i] for i in topic.argsort()[:-10 - 1:-1]}  # Top 10 words per topic
    )
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"SVD Topic {topic_idx + 1}")
    plt.show()

df['LDA_Topic'] = lda.transform(text_matrix).argmax(axis=1)
spam_topic_dist = df[df['spam'] == 1]['LDA_Topic'].value_counts()
non_spam_topic_dist = df[df['spam'] == 0]['LDA_Topic'].value_counts()
print("Spam topic distribution:\n", spam_topic_dist)
print("Non-spam topic distribution:\n", non_spam_topic_dist)

spam_topic_dist.plot(kind='bar', title='Spam Topic Distribution (LDA)')
plt.xlabel('Topics')
plt.ylabel('Count')
plt.show()

non_spam_topic_dist.plot(kind='bar', title='Non-Spam Topic Distribution (LDA)')
plt.xlabel('Topics')
plt.ylabel('Count')
plt.show()

df['SVD_Topic'] = svd.transform(x_tfidf).argmax(axis=1)
spam_svd_topic_dist = df[df['spam'] == 1]['SVD_Topic'].value_counts()
non_spam_svd_topic_dist = df[df['spam'] == 0]['SVD_Topic'].value_counts()
print("Spam topic distribution (SVD):\n", spam_svd_topic_dist)
print("Non-spam topic distribution (SVD):\n", non_spam_svd_topic_dist)

spam_svd_topic_dist.plot(kind='bar', title='Spam Topic Distribution (SVD)', color='orange')
plt.xlabel('Topics')
plt.ylabel('Count')
plt.show()

non_spam_svd_topic_dist.plot(kind='bar', title='Non-Spam Topic Distribution (SVD)', color='blue')
plt.xlabel('Topics')
plt.ylabel('Count')
plt.show()