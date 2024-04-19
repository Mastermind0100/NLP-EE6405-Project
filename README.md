# NLP-EE6405-Project

The code you can use in interface is "bert.py"

## How to use
BasicBERT is the object I created to use BERT model. It has the following methods:

Let's say you create an instance of BasicBERT as bert.
You need to create it like this:
```python
from bert import BasicBERT
bert = BasicBERT("saved_model_path_in_your_local_machine", "test_data_path_in_your_local_machine")

#Then you can use the following methods:
bert.test_model() # This method will test the model with the test data you provided
bert.test_string("text") # This method will predict the label of the text you provided
```
You can see the example in bert.py file.

To be specific:

test_model method will return as follows:

```python
result = {
    "eval_loss": loss,
    "accuracy": accuracy,
    "f1": f1
    "precision": precision,
    "recall": recall,
    "eval_runtime": eval_runtime,
    "eval_samples_per_second": eval_samples_per_second,
    "eval_steps_per_second": eval_steps_per_second,
    "eval_steps_per_second": eval_steps_per_second,
}
```
This is a dictionary object. You can only pick what you need.

The test_string method will return as follows:
```python
result = {
    "Probabilities": prob, # This is a tensor has two elements. The first one is the probability of the text to be 0, the second one is the probability of the text to be 1.
    "Predictions": prediction # This is model's predictin. It will be 0 or 1.
}
```

If you want to show the result by yourself, just comment the print line in bert.py and use a variable to receive the result.
