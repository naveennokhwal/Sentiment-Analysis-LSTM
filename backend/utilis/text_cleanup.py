
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import emoji
from autocorrect import Speller
import spacy

nltk.download('punkt')
nltk.download('stopwords')

class TextCleaning:
    def __init__(self):
        self.spell = Speller(lang='en')
        self.nlp = spacy.load("en_core_web_sm")
        

    # Lowercasing
    def lowercase_text(self, text):
        return text.lower()

    # Remove URLs
    def remove_urls(self, text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    # Remove punctuation
    def remove_punctuation(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))

    # Chat word treatment
    def expand_chat_words(self, text):
        chat_words_dict = {
            'gn': 'good night',
            'gm': 'good morning',
            'lol': 'laugh out loud',
        }
        words = text.split()
        expanded_words = [chat_words_dict.get(word, word) for word in words]
        return ' '.join(expanded_words)

    # Remove stop words
    def remove_stopwords(self, text):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
        return ' '.join(filtered_text)

    # Handle emojis
    def handle_emojis(self, text):
        return emoji.demojize(text)

    # Spelling correction
    
    def correct_spelling(self, text):
        return ' '.join(self.spell(word) for word in text.split())


    # Tokenization
    def tokenize_text(self, text):
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        return tokens

    # Lemmatization
    def lemmatize(self, tokens):
        doc = self.nlp(" ".join(tokens))
        lemmas = [token.lemma_ for token in doc]
        return lemmas
