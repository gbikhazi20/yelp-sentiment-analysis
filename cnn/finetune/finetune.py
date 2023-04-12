#!/usr/bin/python3

from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from datasets import load_dataset

print('Imports successful')

# inspired by HF finetuning demo


def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def launch_train():
    print('Loading dataset...')#, end=' ')
    dataset = load_dataset("yelp_review_full")
    print('Done!')
	
    print('Tokenizing data...')#, end=' ')
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    print('Done!')

    #train_dataset = tokenized_datasets["train"]
    #eval_dataset = tokenized_datasets["test"]

    train_dataset = tokenized_datasets["train"].shuffle(seed=123).select(range(3000))
    eval_dataset = tokenized_datasets["test"].shuffle(seed=123).select(range(1500))

    print('Loading model...')#, end=' ')
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=5)
    print('Done!')
	

    training_args = TrainingArguments(
        output_dir="test_trainer", evaluation_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        #compute_metrics=compute_metrics,
    )

    print('Training model...')#, end=' ')
    trainer.train()

    trainer.save_model('finetuned_bert')
    print('Saved model')

if __name__ == '__main__':
    launch_train()
