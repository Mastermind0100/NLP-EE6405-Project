from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle

class NLPModel:
    def __init__(self) -> None:
        pass

    def import_weights(self, dataset) -> None:
        pass

    def test_model(self, input) -> None:
        pass

    def display_stats(self) -> None:
        pass

class SVM(NLPModel):

    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

    def __init__(self) -> None:
        self.lemmatizer = WordNetLemmatizer()
        self.tv = TfidfVectorizer(stop_words='english')
        self.model = self.import_weights()
        super().__init__()
    
    def import_weights(self, dataset):
        with open(f'svm_{dataset}.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    
    def test_model(self, input):
        tv_input = np.asarray(self.tv.transform([input]).todense())
        self.prediction = self.model.predict(tv_input)
        return self.prediction
    
    def display_stats(self) -> None:
        return {
            'f1_score': 81.6,
        }
        