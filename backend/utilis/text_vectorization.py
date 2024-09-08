import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models import KeyedVectors

class TextVectorisation:
    def __init__(self, model_path):
        self.load_model(model_path)

    def load_model(self, model_path):
        try:
            self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        except Exception as e:
            print(f"Error loading the model: {e}")
            raise
    
    def get_word2vec_vector(self,text):
        text = str(text)
        words = text.split()
        word_vectors = [self.model[word] for word in words if word in self.model]
        if not word_vectors:
            return np.zeros(self.model.vector_size)  # Return a zero vector if no words are found in the model
        return np.mean(word_vectors, axis=0)

    def vectorize_documents(self, texts):
        return np.array([self.get_word2vec_vector(doc) for doc in tqdm(texts, desc="Vectorizing documents")])

# Import data
data_path = r'C:\Learning\Machine-Learning\Deep_Learning_WorkSpace\projects\sentiment-analysis-project\data\processed\cleaned_text.csv'
data = pd.read_csv(data_path)
texts = data['clean text'].tolist()

model_path = r'C:\Learning\Machine-Learning\Deep_Learning_WorkSpace\projects\sentiment-analysis-project\backend\utilis\GoogleNews-vectors-negative300.bin'  # Replace with your path

model_vect = TextVectorisation(model_path)

vectors = model_vect.vectorize_documents(texts)
vectors = pd.DataFrame(vectors)
vectors.to_csv("C:\Learning\Machine-Learning\Deep_Learning_WorkSpace\projects\sentiment-analysis-project\data\processed\\vectors.csv", index=False)