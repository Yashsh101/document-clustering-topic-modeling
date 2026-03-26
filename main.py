import os
import glob
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans


# Download NLTK stopwords if not already present
nltk.download('stopwords')
from nltk.corpus import stopwords

def read_documents_from_folder(root_folder):
    documents = []
    file_paths = []
    for dirpath, _, _ in os.walk(root_folder):
        for file in glob.glob(os.path.join(dirpath, '*.txt')):
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                documents.append(f.read())
                file_paths.append(file)
    return documents, file_paths

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

def main(input_folder, n_clusters=5, n_topics=5, n_top_words=10):
    # Step 1: Read documents
    documents, file_paths = read_documents_from_folder(input_folder)
    print(f"Loaded {len(documents)} documents.")

    # Remove empty documents
    filtered = [(doc, path) for doc, path in zip(documents, file_paths) if doc.strip()]
    if not filtered:
        raise ValueError("No non-empty documents found!")
    documents, file_paths = zip(*filtered)
    documents = list(documents)
    file_paths = list(file_paths)

    if not documents:
        print(documents)
        raise ValueError("No non-empty documents found!")

    # Adjust min_df and max_df
    stop_words = stopwords.words('english')
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=1.0, min_df=1)
    tfidf = tfidf_vectorizer.fit_transform(documents)

    # Step 2: Preprocessing and Vectorization
    # stop_words = stopwords.words('english')
    # tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=0.95, min_df=2)
    # tfidf = tfidf_vectorizer.fit_transform(documents)

    # Step 3: K-means Clustering
    print("K-means Clustering:")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(tfidf)
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf_vectorizer.get_feature_names_out()
    for i in range(n_clusters):
        print(f"Cluster {i + 1}: ", end='')
        print(' '.join([terms[ind] for ind in order_centroids[i, :n_top_words]]))
    print()

    # Step 4: LDA Topic Modeling
    print("LDA Topic Modeling:")
    count_vectorizer = CountVectorizer(stop_words=stop_words, max_df=0.95, min_df=2)
    counts = count_vectorizer.fit_transform(documents)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(counts)
    print_top_words(lda, count_vectorizer.get_feature_names_out(), n_top_words)

    # Optional: Assign clusters/topics to documents
    kmeans_labels = kmeans.predict(tfidf)
    lda_labels = lda.transform(counts).argmax(axis=1)
    df = pd.DataFrame({
        'file_path': file_paths,
        'kmeans_cluster': kmeans_labels,
        'lda_topic': lda_labels
    })
    print(df.head())

if __name__ == "__main__":
    # Change this to your input folder path
    input_folder = "/Users/vaibhav_sharma/Downloads/twenty+newsgroups//20_newsgroups"
    main(input_folder, n_clusters=5, n_topics=5, n_top_words=10)