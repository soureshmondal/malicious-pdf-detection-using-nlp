# Malicious-pdf-detection



The threat posed by malware embedded in Portable Document Formats (PDFs) is a significant concern for the average Internet user. PDFs have the capability to execute JavaScript, which can be exploited to conceal malicious scripts and data. Although several Machine Learning-based models exist for detecting PDF malware, the use of transformers for static analysis of PDFs remains unexplored. Transformers, with their attention mechanisms and parallel data processing capabilities, offer great potential for detailed analysis of large datasets without excessive computational demands. 
The data used for this project was from ***CIC-Evasive-PDFMal2022*** which can be requested [here](https://www.unb.ca/cic/datasets/pdfmal-2022.html).
For preprocessing the PDFs first step is to split the entire dataset and then create the csv file with 4 columns and storing the byte strings of PDFs in the contents column generating meaningful word embeddings through one-hot encoding and variable n-grams, and utilizing these inputs in different fine-tuned transformer models,and then comparing them to finally select BERT as more preferred model.
Then I had used that trained model to check if it is detecting some real life malicious PDFs. My evaluation indicates that this approach is effective for successfully detecting malicious PDF files. Nonetheless, it is crucial to further refine the current model and investigate additional methods to enhance the model's accuracy and precision across diverse datasets.

 
## Installation

First set up a virtual environment using the following command.

```
python3 -m virtualenv venv
```

In order to activate the virtual environment, run the following command.

```
source venv/bin/activate
```

Once the virtualn environment is activated, you can install all of the necessary dependencies using the following command.

```
pip install -r requirements.txt
```

Additionally, two directories by the names of `data` and `results` should be placed at the root of the repository.


## Preparation of data to be trained

In order to perform preprocessing of the ***CIC-Evasive-PDFMal2022*** dataset,run `data_prepare.py` on the relevant zip files to generate the training-validation data split required. Use the following command.This will split the data into training and testing format in the required ratio and also generates the training and testing CSV files required for the model. 

```
python3 data_prepare.py -t 90
```

This will output a `training.csv` and a `testing.csv` which you can place in the `data` directory.

## Training

To run the training script, make sure that training.csv is placed in `data/training.csv`. You can then begin training using the following command.

```
python3 training.py
```

## Validation

To run the validation script, make sure that both `data/testing.csv` is created and `results/model_weights.pth` are placed correctly.
You can then begin validation using the following command.

```
python3 val.py
```

## Check

In order to check that if the trained model is actually detecting any real life malware or not , ensure that `data/test.pdf` (replace with whichever PDF you want to perform inferencing on) and `results/model_weights.pth` are placed correctly.You can then run the following command to perform the check operation.

```
python3 test.py
```

