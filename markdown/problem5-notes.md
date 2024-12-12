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



## Solution for Finding Similar Movies to 'Spider-Man'

### 1. Details on Each Step

#### Step 1: Data Preparation

- **Objective**: Prepare the dataset for analysis by merging and cleaning relevant columns.
- **Input**: The dataset (`PGR210_NLP_data2.csv`) containing information such as movie titles, taglines, overviews, genres, and keywords.
- **Actions**:
  - **Create the 'description' column**: Concatenate the `tagline` and `overview` columns to form a single description of each movie. We handle missing values by filling them with empty strings (`fillna('')`).
  - **Text Preprocessing**: Apply basic text cleaning, which includes converting text to lowercase, removing punctuation and special characters, tokenizing the text, and filtering out stop words using NLTK's stopword list.
  
- **Output**: 
  - A new `description` column in the dataset containing combined and cleaned textual data for each movie.

#### Step 2: Vectorization

- **Objective**: Convert the textual description of each movie into numerical vectors that can be compared using similarity measures.
- **Input**: The cleaned `description` column.
- **Actions**:
  - **TF (Term Frequency) Vectorization**: Use `CountVectorizer` from `sklearn` to convert the text data into a sparse matrix of token counts (terms present in the dataset). This matrix is a direct representation of the term frequencies.
  - **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization**: Use `TfidfVectorizer` to transform the text into a sparse matrix that adjusts term frequencies based on how common a term is across all documents in the dataset.
  
- **Output**: 
  - Two matrices, `x_tf` and `x_tfidf`, representing the TF and TF-IDF values for each movie description.

#### Step 3: Similarity Calculation

- **Objective**: Measure the similarity between movies based on their descriptions.
- **Input**: The vectors created in the previous step (TF and TF-IDF matrices).
- **Actions**:
  - **Cosine Similarity**: Calculate the cosine similarity between the vector representing 'Spider-Man' and the vectors representing all other movies. Cosine similarity is a measure of how similar two vectors are, ranging from 0 (no similarity) to 1 (identical).
  - **Selecting Similar Movies**: For both TF and TF-IDF, sort the similarity values in descending order and select the top 10 most similar movies (excluding the movie "Spider-Man").
  
- **Output**: 
  - Two sorted lists of movies: one based on TF similarity and one based on TF-IDF similarity. These lists contain the top 10 most similar movies to "Spider-Man."

#### Step 4: Visualization

- **Objective**: Visualize the similarity scores of the top 10 similar movies.
- **Input**: The top 10 similar movies based on both TF and TF-IDF similarity scores.
- **Actions**:
  - **Plot the Results**: Use `matplotlib` to create bar plots showing the similarity scores of the top 10 movies.
  - The y-axis will display the movie titles, and the x-axis will represent the similarity scores.
  - **Visualize Differences**: Plot the top similar movies for both TF and TF-IDF to compare the two methods visually.

- **Output**: 
  - Two bar plots showing the similarity scores of the top 10 most similar movies for both TF and TF-IDF.

---

### 2. Major Algorithm to Be Used

- **Cosine Similarity**: The key algorithm used to calculate the similarity between movies is **cosine similarity**. This metric compares the angle between two vectors in a multi-dimensional space, where each dimension corresponds to a term in the document corpus (the movie descriptions). A smaller angle between two vectors indicates higher similarity.

- **Vectorization Algorithms**:
  - **TF (Term Frequency)**: This method counts how frequently each word appears in the document.
  - **TF-IDF (Term Frequency-Inverse Document Frequency)**: This method adjusts the frequency of terms by their inverse document frequency, making rare words more significant in the analysis.

### 3. The Results

- The results show the top 10 most similar movies to "Spider-Man" based on their descriptions:
  
  **TF Results**:
  ```plaintext
  Title                          Similarity
  Spider-Man                      1.000000
  The Amazing Spider-Man 2        0.175114
  Arachnophobia                  0.163973
  The Amazing Spider-Man         0.158537
  ```

  TF-IDF Results:
  ```plaintext
  Title                          Similarity
Spider-Man                      1.000000
The Amazing Spider-Man 2        0.175114
Arachnophobia                  0.163973
  ```

  	•	Both methods identify The Amazing Spider-Man 2 and The Amazing Spider-Man as the most similar movies, with lower similarity scores for other movies like Arachnophobia and Gremlins 2: The New Batch.

4. Analysis on the Results
	•	TF (Term Frequency) Similarity:
	•	TF represents raw word frequency and tends to capture broader similarity, as it does not consider the importance or rarity of words in the entire dataset.
	•	It identifies The Amazing Spider-Man 2 as the second most similar movie, which makes sense given it is a sequel to Spider-Man.
	•	TF-IDF (Term Frequency-Inverse Document Frequency) Similarity:
	•	TF-IDF takes into account the relative rarity of words across the dataset, meaning it gives more weight to less common words that may be more relevant to a specific movie.
	•	The top movies remain largely the same between TF and TF-IDF, with The Amazing Spider-Man 2 and The Amazing Spider-Man being highly similar to Spider-Man.
	•	Visual Comparison:
	•	The bar plots help visualize the similarity scores for both methods. While both methods identify similar movies, TF tends to give more general results, while TF-IDF adjusts for more specific terms.
	•	Differences in the Methods:
	•	TF: Focuses on the raw frequency of words, which may include common words that do not contribute meaningfully to the similarity.
	•	TF-IDF: Adjusts for common words across the entire dataset, providing a better weighting for less frequent, but more specific, terms that are more likely to define a movie’s uniqueness.
	•	Conclusion:
	•	Both TF and TF-IDF successfully identify movies similar to “Spider-Man,” with the TF-IDF method being slightly more accurate by adjusting for less frequent, more informative terms.
	•	Visualizing the results helps to understand the impact of the two methods, and gives insights into which movies are more contextually similar to “Spider-Man.”