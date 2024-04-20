import streamlit as st
from bert import BasicBERT
import pandas as pd
import time
from model_inference_news import NEWS_LSTM

news_lstm = NEWS_LSTM()
bert = BasicBERT("/Users/ou/Desktop/projects/nlpFinalProject/saved_model", "/Users/ou/Desktop/projects/nlpFinalProject/test_data_tweet.csv")
# 进度条
def show_progress_bar():
    progress_bar = st.empty()

    for i in range(11):
        progress_bar.progress(i / 10,)
        time.sleep(0.5)

bert_movie_result = {
    'eval_loss': 0.5846240520477295,
    'eval_accuracy': 0.8751574703955657,
    'eval_f1': 0.8760165144501438,
    'eval_precision': 0.8700298210735586,
    'eval_recall': 0.8820861678004536,
    'eval_runtime': 191.9305,
    'eval_samples_per_second': 41.359,
    'eval_steps_per_second': 5.174
}

bert_tweet_result = {
    'eval_loss': 0.5378416180610657,
    'eval_accuracy': 0.88268156424581,
    'eval_f1': 0.8834628190899,
    'eval_precision': 0.8883928571428571,
    'eval_recall': 0.8785871964679912,
    'eval_runtime': 24.8525,
    'eval_samples_per_second': 36.013,
    'eval_steps_per_second': 4.507
}

bert_news_result = {
    'eval_loss': 0.7662727236747742,
    'eval_accuracy': 0.8979319546364243,
    'eval_f1': 0.9049098819142325,
    'eval_precision': 0.8474970896391153,
    'eval_recall': 0.9706666666666667,
    'eval_runtime': 40.3585,
    'eval_samples_per_second': 37.142,
    'eval_steps_per_second': 4.658
}

# header
st.title("FINAL PROJECT OF GROUP 26")
st.subheader("by Hong, Atharva ,Xu ,Zhou ,Si ,Ou")

st.text('''
Introduction
1. You should choose a model based on your need firstly.
2. Type your text into the box and click the "predict" button then.
3. You will get the result.
''')



# choose the model
model_index = st.sidebar.selectbox(
    label='Choose your model!',
    options=('Bert', 'RoBert', 'Lstm'),
    index=0,
    format_func=str,
    help='如果您不想透露，可以选择保密'
)

if (model_index=='Bert'):
    bert_description = """
    BERT (Bidirectional Encoder Representations from Transformers) utilizes the bidirectional training mechanism of the Transformer to understand the context of language, allowing the model to capture more complex relationships between words. BERT achieved state-of-the-art performance in various NLP tasks at the time, including question-answering, language inference, and sentiment analysis.
    """

    # Display the description in the Streamlit app
    st.write(bert_description)

elif (model_index=='RoBert'):

    robert_description = """
    RoBERTa (Robustly optimized BERT approach) is a pre-trained language representation model proposed by the Facebook AI team in 2019. Built upon the foundation of the BERT model, it further improves performance by leveraging larger training data, longer training time, larger batch sizes, dynamic masking, and other optimization strategies. RoBERTa achieves state-of-the-art performance on various natural language processing tasks and has become one of the leading pre-trained language representation models at the time of its introduction.
    """
    # Display the description in the Streamlit app
    st.write(robert_description)

elif (model_index == 'Lstm'):
    lstm_description = """
    LSTM (Long Short-Term Memory) is a special type of Recurrent Neural Network (RNN) that can learn long-term dependencies, making it suitable for processing and predicting important events in time series with long intervals and delays. LSTM introduces three gates (input gate, forget gate, output gate) to control the memory of neurons, effectively solving the gradient vanishing problem common in traditional RNNs. This makes LSTM highly applicable to complex sequence learning tasks, such as speech recognition, language modeling, and text generation.
    """
    st.write(lstm_description)




dataName= st.sidebar.selectbox(
    label='name of dataset',
    options=('tweet', 'movie', 'news'),
    index=0,
    format_func=str,
)


ifShow = st.sidebar.radio(
    label = 'show the performance of model?',
    options = ('not show', 'show'),
    index = 0,
    format_func = str,
    )


text = st.text_area(label='sentence prediction',
                    value='',
                    height=150,
                    max_chars=200,
                    help='最大长度限制为200')


if st.button('Predict'):
    # 使用所选模型进行预测
    if model_index == 'Bert' and dataName == 'news':
        show_progress_bar()
        result = bert.test_string(text)
        if result[1] == 1:
            st.write("## result : reliable")
        elif result[1] == 0:
            st.write("## result : satire")
    elif model_index == 'Lstm' and dataName == 'news':
        result = news_lstm.test_model(text)
        print(result)
        if result[0] == 1.0:
            st.write("## result : reliable")
        else:
            st.write("## result : satire")


#
if ifShow == 'show':
    if model_index == 'Bert' and dataName == 'tweet':
        # 将字典转换为 DataFrame
        df_news_result = pd.DataFrame.from_dict(bert_tweet_result, orient='index', columns=['Value'])
        df_news_result.index.name = 'Metric'

        # 显示 DataFrame
        st.sidebar.dataframe(df_news_result)

    elif model_index == 'Bert' and dataName == 'news':
        # 将字典转换为 DataFrame
        df_news_result = pd.DataFrame.from_dict(bert_news_result, orient='index', columns=['Value'])
        df_news_result.index.name = 'Metric'

        # 显示 DataFrame
        st.sidebar.dataframe(df_news_result)

    elif model_index == 'Bert' and dataName == 'movie':
        # 将字典转换为 DataFrame
        df_news_result = pd.DataFrame.from_dict(bert_movie_result, orient='index', columns=['Value'])
        df_news_result.index.name = 'Metric'

        # 显示 DataFrame
        st.sidebar.dataframe(df_news_result)

    elif model_index == 'Lstm' and dataName == 'news':
        bert_movie_result = news_lstm.display_stats()
        df_news_result = pd.DataFrame.from_dict(bert_movie_result, orient='index', columns=['Value'])
        df_news_result.index.name = 'Metric'

        # 显示 DataFrame
        st.sidebar.dataframe(df_news_result)








