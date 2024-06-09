# Malicious-pdf-detection



PDFs, widely used for sharing documents due to their portability and consistent formatting across different devices, can unfortunately be weaponized by cybercriminals. One of the primary risks associated with PDFs is their capability to execute JavaScript. This feature, designed to enhance the functionality of PDFs by enabling interactive elements, can be exploited to conceal malicious scripts and data. When a user opens an infected PDF, the embedded JavaScript can execute automatically, potentially compromising the user's system by installing malware, stealing sensitive information, or performing other harmful activities.

To combat this, several Machine Learning (ML)-based models have been developed for detecting PDF malware. These models typically rely on analyzing the structure and content of PDFs to identify anomalies or patterns indicative of malicious activity. However, despite their effectiveness, these traditional ML models have limitations, particularly in handling the complex and varied nature of PDF files.

One promising but largely unexplored area is the use of transformers for the static analysis of PDFs. Transformers, a type of deep learning model, have revolutionized natural language processing (NLP) due to their advanced attention mechanisms and ability to process data in parallel. These features make transformers well-suited for detailed analysis of large datasets, offering the potential to accurately detect malware within PDFs without imposing excessive computational demands.

Attention mechanisms allow transformers to focus on the most relevant parts of the input data, effectively identifying subtle indicators of malicious activity that might be missed by other models. Additionally, the parallel processing capabilities of transformers enable them to handle large volumes of data efficiently, making them ideal for real-time or near-real-time analysis of PDFs.

Exploring the application of transformers in this context could lead to significant advancements in cybersecurity, providing a more robust and scalable solution for protecting users from the hidden dangers lurking in seemingly benign PDF files. 
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

