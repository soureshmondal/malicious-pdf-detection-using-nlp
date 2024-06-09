import numpy as np
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
model.load_state_dict(torch.load("./results/model_weights.pth"))
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

from preprocessing import get_file_byte_string

def pdf_check(test_encoding):
    input_ids = torch.tensor(test_encoding["input_ids"]).to(device)
    attention_mask = torch.tensor(test_encoding["attention_mask"]).to(device)
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
    y = np.argmax(outputs[0].to('cpu').numpy())

    return y


test_encoding1 = tokenizer(str(get_file_byte_string("./data/malicious.pdf")), truncation=True, padding=True)
input_ids = torch.tensor(test_encoding1['input_ids']).to(device)
attention_mask = torch.tensor(test_encoding1['attention_mask']).to(device)
output = pdf_check(test_encoding1)

print("If predicted result is '0' then PDF is Benign if predicted result is '1' then PDF is Malicious ")
print("Predicted Result:", output)
