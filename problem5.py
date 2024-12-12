import pandas as pd
import matplotlib.pyplot as plt
import ast
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Main DF
dfcsv = pd.read_csv("csv/PGR210_NLP_data2.csv", delimiter=";")

df = dfcsv.copy()

# FIXING GENRES
df["genres"] = df["genres"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

all_genres = set([(genre["id"],genre["name"]) for sublist in df["genres"] for genre in sublist])
genre_df = pd.DataFrame(list(all_genres), columns=["ID","Name"])
genre_df.set_index('ID', inplace=True)

def extract_genres(genre_list):
    return [(genre["name"]) for genre in genre_list]

df["genres"] = df["genres"].apply(extract_genres)

# print(df[["title", "genres"]].head())
# print(genre_df)
# print(genre_df.shape)                 // (20,1)

# FIXING KEYWORDS
df["keywords"] = df["keywords"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

all_keywords = set([(kw["id"],kw["name"]) for sublist in df["keywords"] for kw in sublist])
kw_df = pd.DataFrame(list(all_keywords), columns=["ID","Name"])
kw_df.set_index('ID', inplace=True)

def extract_kw(kw_list):
    return [(kw["name"]) for kw in kw_list]

df["keywords"] = df["keywords"].apply(extract_kw)

# print(df[["title", "keywords"]].head())
# print(kw_df)
# print(kw_df.shape)                    // (9813,1)

# FIXING PRODUCTION COMPANIES
df["production_companies"] = df["production_companies"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

all_companies = set([(comp["name"],comp["id"]) for sublist in df["production_companies"] for comp in sublist])
comp_df = pd.DataFrame(list(all_companies), columns=["Name","ID"])
comp_df.set_index('ID', inplace=True)

def extract_comp(comp_list):
    return [(comp["name"]) for comp in comp_list]

df["production_companies"] = df["production_companies"].apply(extract_comp)

# print(df[["title", "production_companies"]].head())
# print(comp_df)
# print(comp_df.shape)                    // (5047,1) 

# FIXING PRODUCTION COUNTRIES
df["production_countries"] = df["production_countries"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

all_countries = set([(coun["iso_3166_1"],coun["name"]) for sublist in df["production_countries"] for coun in sublist])
coun_df = pd.DataFrame(list(all_countries), columns=["Code","Name"])
coun_df.set_index('Code', inplace=True)

def extract_coun(coun_list):
    return [(coun["name"]) for coun in coun_list]

df["production_countries"] = df["production_countries"].apply(extract_coun)

# print(df[["title", "production_countries"]].head())
# print(coun_df)
# print(coun_df.shape)                     // (88,1)

# FIXING LANGUAGES
df["spoken_languages"] = df["spoken_languages"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

all_languages = set([(lang["iso_639_1"],lang["name"]) for sublist in df["spoken_languages"] for lang in sublist])
lang_df = pd.DataFrame(list(all_languages), columns=["Code","Name"])
lang_df.set_index('Code', inplace=True)

def extract_lang(lang_list):
    return [(lang["name"]) for lang in lang_list]

df["spoken_languages"] = df["spoken_languages"].apply(extract_coun)

# print(df[["title", "spoken_languages"]].head())
# print(lang_df)
# print(lang_df.shape)                      // (87,1)
       
df["description"] = df["tagline"].fillna('') + " " + df["overview"].fillna('') 
df["description"] = df["description"].apply(lambda x: x.strip() if x.strip() else "No Description Available")# Answer for Task 2.2.1

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def preprocessing(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum() and word not in stop_words]
    cleaned_text = " ".join(words)
    return cleaned_text

def withoutWhiteSpace(text):
    text = re.sub(r"[\s+]", "", text)
    return text

df["ppc"] = df["description"].apply(preprocessing)
df["wws"] = df["ppc"].apply(withoutWhiteSpace)
# print(df["description"].head())
# print(df["ppc"].head())
# print(df["wws"].head())

vec_tf = CountVectorizer()
vec_tfidf = TfidfVectorizer()

x_tf = vec_tf.fit_transform(df["ppc"])
x_tf_w = vec_tf.fit_transform(df["wws"])
print("TF shape:", x_tf.shape)
print("TF(W) shape:", x_tf_w.shape)

x_tfidf = vec_tfidf.fit_transform(df["ppc"])
x_tfidf_w = vec_tfidf.fit_transform(df["wws"])
print("TF-IDF shape:", x_tfidf.shape)
print("TF-IDF(W) shape:", x_tfidf_w.shape)

spiderman_index = df[df["title"].str.contains("^Spider-Man$", case=False, na=False)].index[0]
# print(spiderman_index)            // index : 160

from sklearn.metrics.pairwise import cosine_similarity

similarities_tf = cosine_similarity(x_tf[spiderman_index], x_tf) # type: ignore
similarities_tfidf = cosine_similarity(x_tfidf[spiderman_index], x_tfidf) # type: ignore

similar_movies_tf = similarities_tf.argsort()[0][::-1]
similar_movies_tfidf = similarities_tfidf.argsort()[0][::-1]

df["similarity_tf"] = similarities_tf.flatten()
top_similar_movies_tf = df.sort_values("similarity_tf", ascending=False)[1:11]
print(top_similar_movies_tf[["title", "similarity_tf"]])

df["similarity_tfidf"] = similarities_tfidf.flatten()
top_similar_movies_tfidf = df.sort_values("similarity_tfidf", ascending=False)[1:11]
print(top_similar_movies_tfidf[["title", "similarity_tfidf"]])


top_movies = top_similar_movies_tfidf
plt.barh(top_movies["title"], top_movies["similarity_tfidf"], color="blue")
plt.xlabel("Similarity Score")
plt.title("Top 10 Movies Similar to Spider-Man")
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.tight_layout()
plt.show()