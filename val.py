#!/usr/bin/env python3
import torch
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizerFast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_model(model_path):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def encode_data(data, tokenizer, device):
    encoded_data = tokenizer(data, truncation=True, padding=True, return_tensors='pt')
    input_ids = encoded_data['input_ids'].to(device)
    attention_mask = encoded_data['attention_mask'].to(device)
    return input_ids, attention_mask

def evaluate(model, test_data, tokenizer, device):
    predictions = []
    true_labels = []

    for content, label in zip(test_data['contents'], test_data['label']):
        input_ids, attention_mask = encode_data(content, tokenizer, device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]
        predictions.append(prediction)
        true_labels.append(label)

    return predictions, true_labels

def print_evaluation(predictions, true_labels):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    confusion = confusion_matrix(true_labels, predictions)

    print("Confusion Matrix:")
    print(confusion)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

def main():
    model_path = "./results/model_weights.pth"
    test_data_path = "./data/testing.csv"

    model = load_model(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    test_data = pd.read_csv(test_data_path)
    test_data.drop("id", axis=1, inplace=True)

    predictions, true_labels = evaluate(model, test_data, tokenizer, device)

    print_evaluation(predictions, true_labels)

if __name__ == "__main__":
    main()
