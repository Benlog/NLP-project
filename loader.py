import os
import json

def tokenize_packed(s : str) -> "list(list(int))" :
    '''
        return an array of array of 0 or 1 encoding the utf8 code of the caracter, if the caracter encoding is no longger than 1 byte, else encode 0xFF

        f'{ord(i):8b}'
        give the string representing ord(i) in binary of length 8
    '''
    return [[1 if digit=='1' else 0 for digit in f'{ord(i):8b}'] if ord(i) <= 0x7F else [1,1,1,1,1,1,1,1] for i in s]

def tokenize_one_hot(s : str) -> "list(list(int))" :
    '''
        return an array of array of 0 or 1 in one hot encoding the position of the caracter in utf8 encoding, if the caracter encoding is no longger than 1 byte, else encode as last position 
    '''
    return [[1 if i == (ord(c) if ord(c) <= 0x7F else 0x80) else 0 for i in range(0x81)] for c in s]

def load_json(path : "PathLike") -> "list(dict)" :
    r = []
    with os.scandir(path) as it:
        for entry in it:
            if entry.name[-5:] == ".json" and entry.is_file():
                with open(entry.path, "r", encoding="utf-8") as file :
                    r.append(json.load(file))
    return r

def load_json_iter(path : "PathLike") -> "iter(dict)" :
    with os.scandir(path) as it:
        for entry in it:
            if entry.name[-5:] == ".json" and entry.is_file():
                with open(entry.path, "r", encoding="utf-8") as file :
                    yield json.load(file)
    