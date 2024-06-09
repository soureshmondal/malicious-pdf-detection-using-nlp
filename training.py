#!/usr/bin/env python3
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments

def preprocess_data(data_path):
    train = pd.read_csv(data_path)
    train.drop("id", axis=1, inplace=True)
    return train

def tokenize_data(data, tokenizer):
    inputs = tokenizer(data['contents'].tolist(), padding="max_length", truncation=True)
    return inputs

class PDFDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['label'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

def train_model(model, training_args, train_dataset):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()
    return model

def save_model_weights(model, file_path):
    torch.save(model.state_dict(), file_path)

def main():
    data_path = "./data/training.csv"
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    train = preprocess_data(data_path)
    inputs = tokenize_data(train, tokenizer)
    train_dataset = PDFDataset(inputs, train['label'].tolist())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    trained_model = train_model(model, training_args, train_dataset)

    save_model_weights(trained_model, 'model_weights.pth')

if __name__ == "__main__":
    main()
