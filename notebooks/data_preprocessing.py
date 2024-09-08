# import libararies
import pandas as pd
from tqdm import tqdm

# import text cleaning class
from backend.utilis.text_cleanup import TextCleaning
from backend.utilis.text_vectorization import TextVectorisation 

'''Text cleaning'''

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
data = pd.read_csv(r"C:\Learning\Machine-Learning\Deep_Learning_WorkSpace\projects\sentiment-analysis-project\data\\raw\\text10000.csv", usecols=['text'])

# create an object
model = PreProcess()

# create list of cleaned text
clean_data = [model.pipeline(text) for text in tqdm(data['text'], desc="Processing texts")]

#convert it into dataframe
clean_data = pd.DataFrame(clean_data)

#save this new dataset
clean_data.to_csv(r"C:\Learning\Machine-Learning\Deep_Learning_WorkSpace\projects\sentiment-analysis-project\data\processed\clean10000.csv")

'''Text Vectorization'''
# Import data
data_path = r'C:\Learning\Machine-Learning\Deep_Learning_WorkSpace\projects\sentiment-analysis-project\data\processed\clean10000.csv'
data = pd.read_csv(data_path)
texts = data['clean text'].tolist()

model_path = r'C:\Learning\Machine-Learning\Deep_Learning_WorkSpace\projects\sentiment-analysis-project\backend\utilis\GoogleNews-vectors-negative300.bin'  # Replace with your path

model_vect = TextVectorisation(model_path)

vectors = model_vect.vectorize_documents(texts)
vectors = pd.DataFrame(vectors)
vectors.to_csv(r"C:\Learning\Machine-Learning\Deep_Learning_WorkSpace\projects\sentiment-analysis-project\data\processed\\vector10000.csv", index=False)