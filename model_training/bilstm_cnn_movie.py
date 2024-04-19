import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.utils import resample
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Conv1D, GlobalMaxPooling1D, Bidirectional, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

nltk.download('stopwords')
nltk.download('punkt')

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    stop_words.add('br')
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

def main():
    #Preprocess train data
    train_data = pd.read_csv('/content/gdrive/MyDrive/NTU mods/EE6405/Project/movie_train.csv', header=None)
    print("Class Distribution:")
    print(train_data['label'].value_counts(normalize=True))

    train_data = pd.read_csv('/content/gdrive/MyDrive/NTU mods/EE6405/Project/movie_train.csv')
    test_data = pd.read_csv('/content/gdrive/MyDrive/NTU mods/EE6405/Project/movie_train.csv')

    print(train_data.value_counts('label'))
    print(test_data.value_counts('label'))

    train_data['text'] = train_data['text'].apply(remove_stopwords)
    test_data['text'] = test_data['text'].apply(remove_stopwords)

    #limit set to not run out of ram
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(train_data['text'])
    train_sequences = tokenizer.texts_to_sequences(train_data['text'])
    test_sequences = tokenizer.texts_to_sequences(test_data['text'])

    X_train = pad_sequences(train_sequences, maxlen=100)
    X_test = pad_sequences(test_sequences, maxlen=100)

    y_train = train_data['label'].values
    y_test = test_data['label'].values

    model = Sequential([
        Embedding(input_dim=10000, output_dim=32, input_length=100),
        SpatialDropout1D(0.2),
        Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
        Bidirectional(LSTM(32)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=128, epochs=5, validation_split=0.2)
    predictions = (model.predict(X_test) > 0.5).astype("int32").flatten()

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    #only shown in notebook
    # plt.figure(figsize=(4, 3))
    # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.xticks(ticks=[0.5, 1.5], labels=['Negative', 'Positive'])
    # plt.yticks(ticks=[0.5, 1.5], labels=['Negative', 'Positive'], rotation=0)
    # plt.show()

    y_probs = model.predict(X_test).ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    # plt.figure()
    # plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC')
    # plt.legend(loc="lower right")
    # plt.show()

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")

    ### SAVE
    import pickle

    with open('./saved_tokenizers/movie_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model.save('./saved_models/bilstm_cnn_movie')

    # ### LOAD
    # from tensorflow.keras.models import load_model

    # with open('movie_tokenizer.pickle', 'rb') as handle:
    #     loaded_tokenizer = pickle.load(handle)

    # loaded_model = load_model('bilstm_cnn_movie')