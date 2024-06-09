#!/usr/bin/env python3
import argparse
import os
import csv
import random
import shutil
import sys  
import zipfile
import numpy as np
from preprocessing import get_file_byte_string

def extract_zip(zip_file, extract_path):
    try:
        with zipfile.ZipFile(zip_file) as zip_ref:
            zip_ref.extractall(extract_path)
    except zipfile.BadZipfile as e:
        print("BAD ZIP: " + zip_file)
        try:
            os.remove(zip_file)
        except OSError as e:
            print(f"Failed to remove {zip_file}: {e}")

def create_directories():
    directories = ["Training/Benign", "Testing/Benign", "Training/Malicious", "Testing/Malicious"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def split_files(source, dest_train, dest_test, files, percentage):
    num_train = round(len(files) * (percentage / 100))
    for file_name in random.sample(files, num_train):
        shutil.move(os.path.join(source, file_name), dest_train)
    for f in os.listdir(source):
        os.rename(os.path.join(source, f), os.path.join(dest_test, f))

def main(args):
    parser = argparse.ArgumentParser(description='Split dataset')
    parser.add_argument('-t', '--training-percentage', type=int, required=False, default=50, help='Percentage of data for training')
    options = parser.parse_args(args)

    if not 5 <= options.training_percentage <= 95:
        print("Percentage must be between 5 and 95.")
        return

    create_directories()

    for filename in os.listdir():
        if filename == "Benign.zip":
            name = os.path.splitext(os.path.basename(filename))[0]
            if not os.path.isdir(name):
                extract_zip(filename, "Benign")
        elif filename == "Malicious.zip":
            name = os.path.splitext(os.path.basename(filename))[0]
            if not os.path.isdir(name):
                extract_zip(filename, "Malicious")

    benign_files = os.listdir("Benign")
    malicious_files = os.listdir("Malicious")

    split_files("Benign", "Training/Benign", "Testing/Benign", benign_files, options.training_percentage)
    split_files("Malicious", "Training/Malicious", "Testing/Malicious", malicious_files, options.training_percentage)

if __name__ == "__main__":
    main(sys.argv[1:])

    
    
def create_row(filetype, file):
    global id_counter
    file_data = np.zeros(4, dtype=object)
    file_data[0] = id_counter
    file_data[1] = filetype
    file_data[2] = os.path.basename(os.path.normpath(file))
    bytecode = get_file_byte_string(file)
    truncated_bytecode = bytecode[:MAX_BYTECODE_LENGTH]
    file_data[3] = truncated_bytecode
    print("Length of file_data:", len(file_data))  
    id_counter += 1
    return file_data

MAX_BYTECODE_LENGTH = 1000  
header = ['id', 'label', 'name', 'contents']
id_counter = 1  

with open('testing.csv', 'w', newline='') as testing_csv:
    writer = csv.writer(testing_csv)
    writer.writerow(header)
    for benign_file in os.listdir(os.path.join('Testing', 'Benign')):
        row_data = create_row(0, os.path.join('Testing', 'Benign', benign_file))
        writer.writerow(row_data)

    for malicious_file in os.listdir(os.path.join('Testing', 'Malicious')):
        row_data = create_row(1, os.path.join('Testing', 'Malicious', malicious_file))
        writer.writerow(row_data)

with open('training.csv', 'w', newline='') as training_csv:
    writer = csv.writer(training_csv)
    writer.writerow(header)
    for benign_file in os.listdir(os.path.join('Training', 'Benign')):
        row_data = create_row(0, os.path.join('Training', 'Benign', benign_file))
        writer.writerow(row_data)

    for malicious_file in os.listdir(os.path.join('Training', 'Malicious')):
        row_data = create_row(1, os.path.join('Training', 'Malicious', malicious_file))
        writer.writerow(row_data)
