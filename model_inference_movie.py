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

        with open(self.tokenizer_directory + 'movie_tokenizer.pickle', 'rb') as handle:
            self.movie_tokenizer = pickle.load(handle)

    def import_weights(self, dataset) -> None:
        pass

    def test_model(self, input) -> None:
        pass

    def display_stats(self) -> None:
        pass

    @staticmethod
    def remove_stopwords(text):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
        return ' '.join(filtered_text)

class MOVIE_RNN(NLPModel):
    def __init__(self) -> None:
        super().__init__()
    
    def import_weights(self, dataset):
        loaded_model = load_model(self.model_directory + f'simple_rnn_{dataset}')
        return loaded_model
    
    def test_model(self, input_string):
        input_df = pd.DataFrame([input_string], columns=['text'])
        input_df['text'] = input_df['text'].apply(self.remove_stopwords)
        adhoc_test_sequences = self.movie_tokenizer.texts_to_sequences(input_df['text'])
        formatted_x_test = pad_sequences(adhoc_test_sequences, maxlen=100)

        loaded_model = self.import_weights('movie')

        label = loaded_model.predict(formatted_x_test).ravel()
        self.prediction = label[0].round()

        if self.prediction:
            confidence = label[0]
        else:
            confidence = 1 - label[0]
        return self.prediction, confidence
    
    def display_stats(self) -> None:
        return {
            'f1_score': 0.8590,
            'auc': 0.9309,
            'precision': 0.8639,
            'recall': 0.8541,
            'accuracy': 0.8609
        }
    
class MOVIE_LSTM(NLPModel):
    def __init__(self) -> None:
        super().__init__()
    
    def import_weights(self, dataset):
        loaded_model = load_model(self.model_directory + f'lstm_{dataset}')
        return loaded_model
    
    def test_model(self, input_string):
        input_df = pd.DataFrame([input_string], columns=['text'])
        input_df['text'] = input_df['text'].apply(self.remove_stopwords)
        adhoc_test_sequences = self.movie_tokenizer.texts_to_sequences(input_df['text'])
        formatted_x_test = pad_sequences(adhoc_test_sequences, maxlen=100)

        loaded_model = self.import_weights('movie')

        label = loaded_model.predict(formatted_x_test).ravel()
        self.prediction = label[0].round()

        if self.prediction:
            confidence = label[0]
        else:
            confidence = 1 - label[0]
        return self.prediction, confidence
    
    def display_stats(self) -> None:
        return {
            'f1_score': 0.8684,
            'auc': 0.9392,
            'precision': 0.8475,
            'recall': 0.8904,
            'accuracy': 0.8661
        }
    

class MOVIE_BiLSTM_CNN(NLPModel):
    def __init__(self) -> None:
        super().__init__()
    
    def import_weights(self, dataset):
        loaded_model = load_model(self.model_directory + f'bilstm_cnn_{dataset}')
        return loaded_model
    
    def test_model(self, input_string):
        input_df = pd.DataFrame([input_string], columns=['text'])
        input_df['text'] = input_df['text'].apply(self.remove_stopwords)
        adhoc_test_sequences = self.movie_tokenizer.texts_to_sequences(input_df['text'])
        formatted_x_test = pad_sequences(adhoc_test_sequences, maxlen=100)

        loaded_model = self.import_weights('movie')

        label = loaded_model.predict(formatted_x_test).ravel()
        self.prediction = label[0].round()

        if self.prediction:
            confidence = label[0]
        else:
            confidence = 1 - label[0]
        return self.prediction, confidence
    
    def display_stats(self) -> None:
        return {
            'f1_score': 0.8774,
            'auc': 0.9446,
            'precision': 0.8724,
            'recall': 0.8823,
            'accuracy': 0.8776
        }
        
if __name__ == "__main__":
    movie_rnn = MOVIE_RNN()
    movie_lstm = MOVIE_LSTM()
    movie_bilstm_cnn = MOVIE_BiLSTM_CNN()

    test_string = '''first trailer film viewed , curious angle storyline would take . 
    plot one 's childhood self return present leaves open many options . 
    Bruce Willis however superb job role given . surprised see well could act part . 
    also good career move many others said seeing agree . 
    film mainly remembering kid used , coming realization n't adult planned . 
    wonderful story gripping tale makes us think . Usually scorn `` ... '' movies . 
    example , Waterworld attempted answer question `` world covered water ... ? '' 
    truthfully , nobody cared . movie however effects everyone theatre . True , 
    young children may fully grasp idea growing dreams fizzle away , leaves great 
    impact adults parents children . movie definitely worth seeing . Although , better
      second time around wo n't thinking much ( kid got , stuff ) relax fun . 
      take something leave cinema . Take piece childhood 've forgotten enjoy .'''

    print("Testing MOVIE RNN, with Reliable")
    print(movie_rnn.test_model(test_string))
    print("Testing MOVIE LSTM, with Reliable")
    print(movie_lstm.test_model(test_string))
    print("Testing MOVIE BiLSTM CNN, with Reliable")
    print(movie_bilstm_cnn.test_model(test_string))
