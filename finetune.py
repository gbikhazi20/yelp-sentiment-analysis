from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from datasets import load_dataset

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
    dataset = load_dataset("yelp_review_full")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=5)

    training_args = TrainingArguments(
        output_dir="test_trainer", evaluation_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model('finetuned_bert')


if __name__ == '__main__':
    launch_train()
