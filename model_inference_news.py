from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import pickle

class NLPModel:
    def __init__(self) -> None:
        self.tokenizer_directory = './saved_tokenizers/'
        self.model_directory = './saved_models/'
        nltk.download('stopwords')
        nltk.download('punkt')

        with open(self.tokenizer_directory + 'news_tokenizer.pickle', 'rb') as handle:
            self.news_tokenizer = pickle.load(handle)

    def import_weights(self, dataset) -> None:
        pass

    def test_model(self, input) -> None:
        pass

    def display_stats(self) -> None:
        pass

    def remove_stopwords(text):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
        return ' '.join(filtered_text)

class NEWS_RNN(NLPModel):
    def __init__(self) -> None:
        super().__init__()
    
    def import_weights(self, dataset):
        loaded_model = load_model(self.model_directory + f'simple_rnn_{dataset}')
        return loaded_model
    
    def test_model(self, input_string):
        input_df = pd.DataFrame([input_string], columns=['text'])
        input_df['text'] = input_df['text'].apply(self.remove_stopwords)
        adhoc_test_sequences = self.news_tokenizer.texts_to_sequences(input_df['text'])
        formatted_x_test = pad_sequences(adhoc_test_sequences, maxlen=100)

        loaded_model = self.import_weights('news')

        label = loaded_model.predict(formatted_x_test).ravel()
        self.prediction = label[0].round()

        if self.prediction:
            confidence = label[0]
        else:
            confidence = 1 - label[0]
        return self.prediction, confidence
    
    def display_stats(self) -> None:
        return {
            'f1_score': 0.8824,
            'auc': 0.9374,
            'precision': 0.9059,
            'recall': 0.8600,
            'accuracy': 0.8853
        }
    
class NEWS_LSTM(NLPModel):
    def __init__(self) -> None:
        super().__init__()
    
    def import_weights(self, dataset):
        loaded_model = load_model(self.model_directory + f'lstm_{dataset}')
        return loaded_model
    
    def test_model(self, input_string):
        input_df = pd.DataFrame([input_string], columns=['text'])
        input_df['text'] = input_df['text'].apply(self.remove_stopwords)
        adhoc_test_sequences = self.news_tokenizer.texts_to_sequences(input_df['text'])
        formatted_x_test = pad_sequences(adhoc_test_sequences, maxlen=100)

        loaded_model = self.import_weights('news')

        label = loaded_model.predict(formatted_x_test).ravel()
        self.prediction = label[0].round()

        if self.prediction:
            confidence = label[0]
        else:
            confidence = 1 - label[0]
        return self.prediction, confidence
    
    def display_stats(self) -> None:
        return {
            'f1_score': 0.9108,
            'auc': 0.9721,
            'precision': 0.9233,
            'recall': 0.8987,
            'accuracy': 0.9120
        }
    

class NEWS_BiLSTM_CNN(NLPModel):
    def __init__(self) -> None:
        super().__init__()
    
    def import_weights(self, dataset):
        loaded_model = load_model(self.model_directory + f'bilstm_cnn_{dataset}')
        return loaded_model
    
    def test_model(self, input_string):
        input_df = pd.DataFrame([input_string], columns=['text'])
        input_df['text'] = input_df['text'].apply(self.remove_stopwords)
        adhoc_test_sequences = self.news_tokenizer.texts_to_sequences(input_df['text'])
        formatted_x_test = pad_sequences(adhoc_test_sequences, maxlen=100)

        loaded_model = self.import_weights('news')

        label = loaded_model.predict(formatted_x_test).ravel()
        self.prediction = label[0].round()

        if self.prediction:
            confidence = label[0]
        else:
            confidence = 1 - label[0]
        return self.prediction, confidence
    
    def display_stats(self) -> None:
        return {
            'f1_score': 0.9001,
            'auc': 0.9638,
            'precision': 0.8989,
            'recall': 0.9013,
            'accuracy': 0.9000
        }
        
if __name__ == "__main__":
    news_rnn = NEWS_RNN()
    news_lstm = NEWS_LSTM()
    news_bilstm_cnn = NEWS_BiLSTM_CNN()
    print("Testing News RNN, with Reliable")
    test_string = '''The Alberta province health minister wants to know if swine flu 
                        shots were 'inappropriately diverted' to the Calgary Flames while thousands 
                        had to stand in line for hours for the vaccine. Alberta Health Minister 
                        Ron Liepert says he doesn't know where the NHL team got the vaccine, adding 
                        that Alberta Health Services is the only supplier in the province. 
                        Team president Ken King says the club contacted the department and asked for the clinic. 
                        Health officials have begun an investigation into the special clinic, 
                        which was held for the players and their families last Friday. 
                        Liepert says the vaccine would be diverted only with the approval of the 
                        chief medical officer of health, but he doesn't know if that was the case. 
                        Alberta's opposition parties say professional ice hockey players shouldn't 
                        be getting the vaccine ahead of cancer patients and pregnant women.
                        '''
    print(news_rnn.test_model(test_string))
    print("Testing News LSTM, with Reliable")
    print(news_lstm.test_model(test_string))
    print("Testing News BiLSTM CNN, with Reliable")
    print(news_bilstm_cnn.test_model(test_string))
