from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import re
from nltk.corpus import stopwords, wordnet
import tensorflow as tf

class NLPModel:
    def __init__(self) -> None:
        pass

    def import_weights(self, dataset) -> None:
        pass

    def test_model(self, input) -> None:
        pass

    def display_stats(self) -> None:
        pass

class SVM_Movie(NLPModel):

    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

    def __init__(self) -> None:
        self.lemmatizer = WordNetLemmatizer()
        with open('svm_movie_vectorizer.pkl', 'rb') as f:
            self.tv = pickle.load(f)
        self.model = self.import_weights()
        super().__init__()
    
    def import_weights(self):
        with open('svm_movie.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    
    def test_model(self, input):
        test_review_tfidf=np.asarray(self.tv.transform([self.preprocess_text(input)]).todense())
        self.prediction = self.model.predict(test_review_tfidf)[0]
        return self.prediction
    
    def display_stats(self) -> None:
        return {
            'f1_score': 0.808,
            'accuracy': 0.808,
            'precision': 0.7658,
            'recall': 0.9167,
            'auc': 0.8015536723163841
        }
    
    def preprocess_text(self, text):
        text = re.sub('(<.*?>)', ' ', text)
        text = re.sub('[,\.!?:()"]', '', text)
        text = text.strip()
        text = re.sub('[^a-zA-Z"]',' ', text)
        text = text.lower()
        text = self.tagged_lemma(text)

        words = tf.keras.preprocessing.text.text_to_word_sequence(text)
        stop_words = set(stopwords.words('english'))
        filtered_words = [w for w in words if not w in stop_words]
        text = " ".join(filtered_words)

        return text
    
    def tagged_lemma(self, string):
        pos_tagged = nltk.pos_tag(nltk.word_tokenize(string))
        wordnet_tagged = list(map(lambda x: (x[0], self.pos_tagger(x[1])), pos_tagged))
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                lemmatized_sentence.append(word)
            else:       
                lemmatized_sentence.append(self.lemmatizer.lemmatize(word, tag))
        lemmatized_sentence = " ".join(lemmatized_sentence)
        return lemmatized_sentence
    
    def pos_tagger(self, nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:         
            return None
        

class SVM_News(NLPModel):

    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

    def __init__(self) -> None:
        self.lemmatizer = WordNetLemmatizer()
        with open('svm_news_vectorizer.pkl', 'rb') as f:
            self.tv = pickle.load(f)
        self.model = self.import_weights()
        super().__init__()
    
    def import_weights(self):
        with open('svm_news.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    
    def test_model(self, input):
        test_review_tfidf=np.asarray(self.tv.transform([self.preprocess_text(input)]).todense())
        self.prediction = self.model.predict(test_review_tfidf)[0]
        return self.prediction
    
    def display_stats(self) -> None:
        return {
            'f1_score': 0.808,
            'accuracy': 0.808,
            'precision': 0.7658,
            'recall': 0.9167,
            'auc': 0.8015536723163841
        }
    
    def preprocess_text(self, text):
        text = re.sub('(<.*?>)', ' ', text)
        text = re.sub('[,\.!?:()"]', '', text)
        text = text.strip()
        text = re.sub('[^a-zA-Z"]',' ', text)
        text = text.lower()
        text = self.tagged_lemma(text)

        words = tf.keras.preprocessing.text.text_to_word_sequence(text)
        stop_words = set(stopwords.words('english'))
        filtered_words = [w for w in words if not w in stop_words]
        text = " ".join(filtered_words)

        return text
    
    def tagged_lemma(self, string):
        pos_tagged = nltk.pos_tag(nltk.word_tokenize(string))
        wordnet_tagged = list(map(lambda x: (x[0], self.pos_tagger(x[1])), pos_tagged))
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                lemmatized_sentence.append(word)
            else:       
                lemmatized_sentence.append(self.lemmatizer.lemmatize(word, tag))
        lemmatized_sentence = " ".join(lemmatized_sentence)
        return lemmatized_sentence
    
    def pos_tagger(self, nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:         
            return None

class SVM_Tweets(NLPModel):

    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

    def __init__(self) -> None:
        self.lemmatizer = WordNetLemmatizer()
        with open('svm_tweets_vectorizer.pkl', 'rb') as f:
            self.tv = pickle.load(f)
        self.model = self.import_weights()
        super().__init__()
    
    def import_weights(self):
        with open('svm_tweets.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    
    def test_model(self, input):
        test_review_tfidf=np.asarray(self.tv.transform([self.preprocess_text(input)]).todense())
        self.prediction = self.model.predict(test_review_tfidf)[0]
        return self.prediction
    
    def display_stats(self) -> None:
        return {
            'f1_score': 0.928,
            'accuracy': 0.928,
            'precision': 0.928,
            'recall': 0.928,
            'auc': 0.5263
        }
    
    def preprocess_text(self, text):
        text = re.sub('(<.*?>)', ' ', text)
        text = re.sub('[,\.!?:()"]', '', text)
        text = text.strip()
        text = re.sub('[^a-zA-Z"]',' ', text)
        text = text.lower()
        text = self.tagged_lemma(text)

        words = tf.keras.preprocessing.text.text_to_word_sequence(text)
        stop_words = set(stopwords.words('english'))
        filtered_words = [w for w in words if not w in stop_words]
        text = " ".join(filtered_words)

        return text
    
    def tagged_lemma(self, string):
        pos_tagged = nltk.pos_tag(nltk.word_tokenize(string))
        wordnet_tagged = list(map(lambda x: (x[0], self.pos_tagger(x[1])), pos_tagged))
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                lemmatized_sentence.append(word)
            else:       
                lemmatized_sentence.append(self.lemmatizer.lemmatize(word, tag))
        lemmatized_sentence = " ".join(lemmatized_sentence)
        return lemmatized_sentence
    
    def pos_tagger(self, nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:         
            return None

if __name__ == "__main__":
    svm_tweets = SVM_Tweets()
    print(svm_tweets.test_model("ouch...junior is angryð#got7 #junior #yugyoem   #omg"))
