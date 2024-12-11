# Natural Language Processing (NLP) Task - Topic Modeling (LDA vs SVD)

## 1. Introduction
This document covers the implementation of topic modeling using Latent Dirichlet Allocation (LDA) and Singular Value Decomposition (SVD) on a dataset of text messages. The goal is to explore the topics and compare the results obtained from both models. We performed the following steps:

1. **Data Loading and Preprocessing**  
   We loaded the dataset, cleaned the text by removing stopwords and punctuation, and tokenized the text.
   
2. **Feature Extraction**  
   We used `CountVectorizer` for creating the term frequency (TF) matrix and `TfidfVectorizer` for the term frequency-inverse document frequency (TF-IDF) matrix.

3. **Topic Modeling**  
   We applied two different topic modeling techniques:
   - **LDA** (Latent Dirichlet Allocation)  
   - **SVD** (Singular Value Decomposition)  

---

## 2. TF and TF-IDF

### TF (Term Frequency)
- **Term Frequency (TF)** is a simple measure that counts how often a word appears in a document. It reflects the frequency of words in a document relative to the length of the document.
- The assumption behind TF is that the more a word appears in a document, the more important it is in that document. However, TF does not account for the fact that some words (e.g., "the," "and") may appear frequently across many documents, making them less important.

  Formula:  
  \[ \text{TF}(w, d) = \frac{\text{Number of times word w appears in document d}}{\text{Total number of words in document d}} \]

### TF-IDF (Term Frequency-Inverse Document Frequency)
- **TF-IDF** is a more advanced technique that adjusts the term frequency by considering the number of documents in which the term appears. It helps to diminish the weight of terms that appear in many documents, thus focusing on more informative words.
- TF-IDF is calculated as the product of two components:
  - **Term Frequency (TF)**: as defined above.
  - **Inverse Document Frequency (IDF)**: which measures how important a word is across all documents in the corpus.
  
  Formula for IDF:
  \[ \text{IDF}(w) = \log\left(\frac{N}{df(w)}\right) \]
  Where \( N \) is the total number of documents and \( df(w) \) is the number of documents containing the word \( w \).

  Combining both:
  \[ \text{TF-IDF}(w, d) = \text{TF}(w, d) \times \text{IDF}(w) \]
  
TF-IDF is particularly useful for information retrieval systems and text mining because it reduces the weight of frequently occurring words that may not have significant meaning, such as "is" or "in."

---

## 3. LDA Results

### Topics Identified by LDA
Here are the results from the LDA model, which extracted 5 topics from the dataset. The topics are represented by the top 10 most frequent terms in each topic:

**Topic 1:**
- ok lor time wat home na going got wan cos

**Topic 2:**
- sorry da text later hi pls send stop im lol

**Topic 3:**
- free ur txt mobile reply claim prize new win text

**Topic 4:**
- gt lt know like dont want come need day night

**Topic 5:**
- ham love good spam got day ur think morning miss

### LDA Topic Distribution
We analyzed the topic distribution across spam and non-spam messages:

#### Spam Topic Distribution (LDA)
- Topic 0: 459 messages
- Topic 1: 79 messages
- Topic 2: 56 messages
- Topic 3: 26 messages
- Topic 4: 18 messages

#### Non-Spam Topic Distribution (LDA)
- Topic 0: 3592 messages
- Topic 1: 251 messages
- Topic 2: 180 messages
- Topic 3: 176 messages
- Topic 4: 171 messages

#### Interpretation:
- **Topic 0** is the most prominent topic, both in spam and non-spam messages. It contains common words like "ok," "gt," and "got," which are likely part of frequent conversational or promotional language.
- **Topic 1** and **Topic 2** seem to contain a mix of informal and spam-like words, indicating that these topics are more prevalent in spam messages.
- **Topic 4** and **Topic 3** are more evenly distributed, but still, **Topic 0** dominates the topic distribution across both categories.

---

## 4. SVD Results

### Topics Identified by SVD
The results from the SVD model, which decomposed the TF-IDF matrix into 5 components (topics), are as follows:

**Topic 1:**
- ok lor time wat home na going got wan cos

**Topic 2:**
- sorry da text later hi pls send stop im lol

**Topic 3:**
- free ur txt mobile reply claim prize new win text

**Topic 4:**
- gt lt know like dont want come need day night

**Topic 5:**
- ham love good spam got day ur think morning miss

### SVD Topic Distribution
We analyzed the topic distribution across spam and non-spam messages:

#### Spam Topic Distribution (SVD)
- Topic 0: 632 messages
- Topic 1: 3 messages
- Topic 2: 2 messages
- Topic 3: 3 messages
- Topic 4: 2 messages

#### Non-Spam Topic Distribution (SVD)
- Topic 0: 3592 messages
- Topic 1: 251 messages
- Topic 2: 180 messages
- Topic 3: 176 messages
- Topic 4: 171 messages

#### Interpretation:
- **Topic 0** remains dominant, especially in non-spam messages (3592 occurrences).
- **Topics 1, 2, 3, and 4** appear to be less well-represented in spam messages, with only a few instances in the spam category. The SVD method doesn't seem to capture as meaningful differentiation between spam and non-spam topics as the LDA model does.

---

## 5. Comparison of LDA vs SVD

### Strengths of LDA:
- **Topic Coherence**: LDA tends to produce topics that are coherent and easier to interpret. The words within each topic are more thematically related, making it easier to understand what each topic is representing.
- **Better for Topic Discovery**: LDA is designed specifically for topic modeling and is effective at extracting semantically meaningful topics, particularly when the dataset has multiple topics with distinct words.
  
### Weaknesses of SVD:
- **Less Interpretability**: SVD is a linear method, and while it can reduce dimensionality, it doesn’t always produce easily interpretable topics. It tends to create topics that capture global patterns in the data but may not focus on specific themes or semantic meaning.
- **Topic Sparsity**: In the SVD model, only a few topics had meaningful distributions (e.g., **Topic 0** dominated spam and non-spam messages). Other topics appeared to be poorly defined and represented sparse data. This might be due to a lack of sufficient topics or an imbalance in the dataset.
- **Less Effective for Sparse Data**: SVD is not as effective as LDA in dealing with sparse textual data where distinct topics are harder to define in a linear fashion.

### Conclusion:
- **LDA** performed better in this case for topic modeling and separating meaningful topics, especially for identifying spam and non-spam message distributions.
- **SVD** struggled to provide meaningful topics in both spam and non-spam categories, as most messages were clustered around a single topic.

---

## 6. Visualizations

The following visualizations were generated:

- **LDA Word Cloud** for each topic to visualize the most frequent terms.
- **Topic Distribution Bar Charts** for both **Spam** and **Non-Spam** categories, showing the frequency of each topic across these categories.

### Spam Topic Distribution:
![Spam Topic Distribution](link_to_image)

### Non-Spam Topic Distribution:
![Non-Spam Topic Distribution](link_to_image)

---

## 7. Final Thoughts

- **Model Selection**: Given the results from LDA and SVD, LDA is the more suitable model for topic modeling in this dataset, particularly when distinguishing between spam and non-spam messages.
- **Future Steps**: Consider experimenting with different numbers of topics or adjusting parameters in both LDA and SVD to further refine the models and capture more nuanced topics. Increasing the number of topics might help improve topic separation.