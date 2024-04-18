import datasets
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


class BasicBERT:
    def __init__(self, model_path, data_path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        data = pd.read_csv(data_path)
        data.columns = ["labels", "text"]
        self.dataset = datasets.Dataset.from_pandas(data)

    def tokenize(self, data):
        tokenized_data = self.tokenizer(data["text"],
                                        padding="max_length",
                                        truncation=True,
                                        max_length=128,
                                        return_tensors='pt')
        tokenized_data['labels'] = torch.tensor(data["labels"])
        return tokenized_data

    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        f1 = datasets.load_metric("f1")
        precision = datasets.load_metric("precision")
        recall = datasets.load_metric("recall")
        accuracy = datasets.load_metric("accuracy")

        f1_score = f1.compute(predictions=predictions, references=labels, average='binary')
        precision_score = precision.compute(predictions=predictions, references=labels, average='binary')
        recall_score = recall.compute(predictions=predictions, references=labels, average='binary')
        accuracy_score = accuracy.compute(predictions=predictions, references=labels)

        return {
            "accuracy": accuracy_score['accuracy'],
            "f1": f1_score['f1'],
            "precision": precision_score['precision'],
            "recall": recall_score['recall']
        }

    def test_model(self):
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="epoch",
            learning_rate=5e-5,
        )
        tokenized_test = self.dataset.map(self.tokenize, batched=True)
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            eval_dataset=tokenized_test,
            compute_metrics=self.compute_metrics,
            args=training_args,
        )
        result = trainer.evaluate()
        print(result)
        return result

    def test_string(self, input):
        training_args = TrainingArguments(
            output_dir="./results",
            per_device_eval_batch_size=1
        )
        trainer = Trainer(
            model=self.model,
            args=training_args
        )
        data = pd.DataFrame({'labels': [-1], 'text': input})
        data = datasets.Dataset.from_pandas(data)
        tokenized_input = data.map(self.tokenize, batched=True)
        predictions = trainer.predict(tokenized_input)
        logits = predictions.predictions
        probs = torch.softmax(torch.tensor(logits), dim=-1)
        predictions = torch.argmax(probs, dim=-1)
        predictions = predictions.item()

        print("Probabilities:", probs)
        print("Predictions:", predictions)
        return [probs, predictions]
