#!/usr/bin/env python3
import glob

def get_file_byte_string(file):
    curr_file = open(file, "rb")
    data = curr_file.read()
    data_str = str(data)
    data_delim = ' '.join(data_str[i:i+4] for i in range(0, len(data_str), 4)) 
    data_bytes = bytes(data_delim, 'utf-8')
    curr_file.close()
    return data_bytes
