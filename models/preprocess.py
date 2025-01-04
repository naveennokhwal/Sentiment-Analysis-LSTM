import sys
import os

# Add the sibling folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utilis')))

# Now you can import from 'utilis'
from data_generator import generate_data
from text_cleanup import TextCleaning
from text_vectorization import TextVectorisation

import pandas as pd
from tqdm import tqdm


# create a class
class PreProcess(TextCleaning):
    def __init__(self):
        super().__init__()

    def pipeline(self, text):
        text = self.lowercase_text(text)
        text = self.remove_urls(text)
        text = self.expand_chat_words(text)
        text = self.correct_spelling(text)
        text = self.remove_punctuation(text)
        text = self.remove_stopwords(text)
        text = self.handle_emojis(text)
        tokens = self.tokenize_text(text)
        lemmas = self.lemmatize(tokens)

        return ' '.join(lemmas)
    
#import data 
raw_file_path = r"C:\Learning\Machine-Learning\Deep_Learning_WorkSpace\projects\text-sentiment-classifier\Sentiment-Analysis-LSTM\data\raw\train.csv"
new_data = generate_data(raw_file_path, 1000)

# create an object
model = PreProcess()


# create list of cleaned text
clean_data = [model.pipeline(text) for text in tqdm(new_data['text'], desc="Processing texts")]
label = new_data["sentiment"]

#convert it into dataframe
clean_data = pd.DataFrame(clean_data, columns=["clean_text"])


#save this new dataset
clean_file_path = r"C:\Learning\Machine-Learning\Deep_Learning_WorkSpace\projects\text-sentiment-classifier\Sentiment-Analysis-LSTM\data\processed\clean3000.csv"
label_file_path = r"C:\Learning\Machine-Learning\Deep_Learning_WorkSpace\projects\text-sentiment-classifier\Sentiment-Analysis-LSTM\data\processed\label3000.csv"

clean_data.to_csv(clean_file_path, index=False)
label.to_csv(label_file_path, index = False)


clean_data = pd.read_csv(clean_file_path)
texts = clean_data["clean_text"].tolist()
# Initialize and fit the vectorizer
vectorizer = TextVectorisation(max_features=5000, target_dims=300)
vectors = vectorizer.fit_transform(texts)

# Save the vectors
vectors_df = pd.DataFrame(vectors)
vector_file_path = r"C:\Learning\Machine-Learning\Deep_Learning_WorkSpace\projects\text-sentiment-classifier\Sentiment-Analysis-LSTM\data\processed\vectors3000.csv"
vectors_df.to_csv(vector_file_path, index=False)