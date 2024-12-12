# Task 2.2: Searching for Similar Movies

## Step 1: Data Preparation

1. **Loading the Data**:
   - Load the dataset (`PGR210_NLP_data2.csv`) into a pandas DataFrame.
   - Check for missing or null values and understand the structure of the dataset using the `df.info()` and `df.head()` functions.
   - The dataset contains 4804 rows and 20 columns, including important columns like `genres`, `keywords`, `production_companies`, and `description`.

2. **Creating the 'Description' Column**:
   - Combine the `tagline` and `overview` columns to create a new column `description`, which contains the text description of each movie.
   - The formula used for this is: `df["description"] = df["tagline"].fillna('') + " " + df["overview"].fillna('')`.

3. **Text Preprocessing**:
   - Clean the `description` column by applying multiple steps:
     - Convert the text to lowercase.
     - Remove punctuation and special characters using regular expressions.
     - Tokenize the text and remove stop words using the NLTK stopword list.
     - Optionally, remove whitespaces.
   
   The preprocessing function is defined as follows:

```python
    def preprocessing(text):
       text = text.lower()
       text = re.sub(r"[^\w\s]", "", text)
       words = word_tokenize(text)
       words = [word for word in words if word.isalnum() and word not in stop_words]
       cleaned_text = " ".join(words)
       return cleaned_text
```
    
	•	A secondary function is used to remove whitespaces:

```python
    def withoutWhiteSpace(text):
        text = re.sub(r"[\s+]", "", text)
        return text
```

4.	Creating Preprocessed Columns:
	•	Two columns were created:
	•	ppc (preprocessed text without whitespaces).
	•	wws (preprocessed text with whitespaces removed).

Step 2: TF and TF-IDF Representation
	1.	TF Representation:
	•	Use the CountVectorizer from sklearn to convert the preprocessed text into a Term Frequency (TF) matrix.
	•	The resulting shape of the TF matrix for ppc (preprocessed text with punctuation and stopwords removed) is (4804, 24143), while the wws matrix (with whitespaces removed) is (4804, 4799).
	2.	TF-IDF Representation:
	•	Use TfidfVectorizer from sklearn to compute the Term Frequency-Inverse Document Frequency (TF-IDF) matrix.
	•	The resulting shape of the TF-IDF matrix for ppc is (4804, 24143) and for wws is (4804, 4799).
	3.	Comparison of TF and TF-IDF:
	•	Both TF and TF-IDF are useful for finding word frequency distributions across the dataset.
	•	TF captures the raw frequency of words, whereas TF-IDF adjusts frequencies based on the rarity of terms in the entire dataset.

Step 3: Finding Similar Movies
	1.	Using Cosine Similarity:
	•	Cosine similarity is calculated between the Spider-Man movie and all other movies based on their TF and TF-IDF representations.
	•	cosine_similarity from sklearn.metrics.pairwise is used to compute the similarity between the Spider-Man description and all others.

```python
similarities_tf = cosine_similarity(x_tf[spiderman_index], x_tf)
similarities_tfidf = cosine_similarity(x_tfidf[spiderman_index], x_tfidf)
```

2.	Extracting Similar Movies:
	•	For both TF and TF-IDF, the top 10 similar movies (excluding “Spider-Man”) are selected by sorting the similarity scores.
	•	The results are then plotted using a horizontal bar plot to visualize the similarity scores.

```python
top_similar_movies_tf = df.sort_values("similarity_tf", ascending=False)[1:11]
top_similar_movies_tfidf = df.sort_values("similarity_tfidf", ascending=False)[1:11]
```

	•	For visualization, the top 10 similar movies are plotted using matplotlib.

	3.	Results:
	•	The output shows the top 10 movies most similar to “Spider-Man” based on both TF and TF-IDF similarity scores.
	•	TF Results:
    Title                          Similarity
Spider-Man                      1.000000
The Amazing Spider-Man 2        0.175114
Arachnophobia                  0.163973
The Amazing Spider-Man         0.158537

	•	TF-IDF Results:
    Title                          Similarity
Spider-Man                      1.000000
The Amazing Spider-Man 2        0.175114
Arachnophobia                  0.163973

4.	Analysis of Results:
	•	The results show that Spider-Man itself has a similarity score of 1.000000, which is expected.
	•	Other movies like The Amazing Spider-Man 2 and Arachnophobia show lower similarity scores.
	•	While TF and TF-IDF produce slightly different results, both methods identify The Amazing Spider-Man 2 and The Amazing Spider-Man as the most similar.
	•	TF Method tends to show broader similarity because it directly reflects raw frequency counts.
	•	TF-IDF Method adjusts for term importance, giving better weighting to rarer terms that may be more specific to certain movies.

Step 4: Visualization of Similar Movies
	•	Visualization was performed using matplotlib, and the horizontal bar plot displays the similarity scores of the top 10 movies most similar to Spider-Man.
	•	matplotlib helps in comparing the similarities visually by plotting the title of the movie on the y-axis and the similarity score on the x-axis.

Notes on Task:
	•	The task involved text processing, creating new features, applying TF and TF-IDF models, and finding similarities based on those models.
	•	The results and visualizations were analyzed and compared to answer the task question effectively.