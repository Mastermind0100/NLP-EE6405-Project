from bert import BasicBERT
import warnings
warnings.filterwarnings("ignore")

bert = BasicBERT(model_path="/Users/kiriharari/Desktop/EE6405/BERT_NEWS/saved_model", data_path="/Users/kiriharari/Desktop/EE6405/BERT_NEWS/final_news_balancedtest.csv")
# bert.test_model()
bert.test_string("A man revive after his death was confirmed.")