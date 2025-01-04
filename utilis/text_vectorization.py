import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

class TextVectorisation:
    def __init__(self, max_features=5000, target_dims=300):

        self.max_features = max_features
        self.target_dims = target_dims
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.svd = TruncatedSVD(n_components=target_dims, random_state=42)
        
    def pad_or_truncate_vector(self, vector, target_length):

        current_length = len(vector)
        
        if current_length == target_length:
            return vector
        elif current_length < target_length:
            # Pad with zeros
            return np.pad(vector, (0, target_length - current_length), 'constant')
        else:
            # Truncate
            return vector[:target_length]
            
    def handle_sparse_text(self, text):

        if not str(text).strip():
            return "empty_document_placeholder"
        return str(text)
    
    def fit_transform(self, texts):

        print("Processing texts...")
        processed_texts = [self.handle_sparse_text(text) for text in texts]
        
        print("Vectorizing documents using TF-IDF...")
        try:
            tfidf_matrix = self.vectorizer.fit_transform(tqdm(processed_texts))
            
            # Check if we have enough features for target dimensions
            actual_features = tfidf_matrix.shape[1]
            if actual_features < self.target_dims:
                print(f"Warning: Number of features ({actual_features}) is less than target dimensions ({self.target_dims})")
                self.target_dims = actual_features
                self.svd = TruncatedSVD(n_components=self.target_dims, random_state=42)
            
            print(f"Reducing dimensionality from {tfidf_matrix.shape[1]} to {self.target_dims}...")
            vectors = self.svd.fit_transform(tfidf_matrix)
            
            # Ensure all vectors have the same dimension
            vectors = np.array([self.pad_or_truncate_vector(vec, self.target_dims) for vec in vectors])
            
            explained_variance = self.svd.explained_variance_ratio_.sum() * 100
            print(f"Explained variance after dimensionality reduction: {explained_variance:.2f}%")
            
            return vectors
            
        except ValueError as e:
            print(f"Error during vectorization: {e}")
            # Return zero vectors as fallback
            return np.zeros((len(texts), self.target_dims))
    
    def transform(self, texts):

        processed_texts = [self.handle_sparse_text(text) for text in texts]
        try:
            tfidf_matrix = self.vectorizer.transform(processed_texts)
            vectors = self.svd.transform(tfidf_matrix)
            return np.array([self.pad_or_truncate_vector(vec, self.target_dims) for vec in vectors])
        except Exception as e:
            print(f"Error during transformation: {e}")
            return np.zeros((len(texts), self.target_dims))