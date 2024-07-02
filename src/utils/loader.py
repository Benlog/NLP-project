import json
import math
from os import PathLike, scandir
from pathlib import Path
from random import choice, getrandbits
from typing import Callable, Generator, Iterator, Optional, Sequence, Union
from unittest.util import strclass

import torch
from torch.utils.data import IterableDataset


def tokenize_packed(s: str) -> list[list[int]]:
    """
    return an array of array of 0 or 1 encoding the utf8 code of the caracter, if the
    caracter encoding is no longger than 1 byte, else encode 0xFF

    f'{ord(i):8b}'
    give the string representing ord(i) in binary of length 8
    """
    return [
        [1 if digit == "1" else 0 for digit in f"{ord(i):8b}"]
        if ord(i) <= 0x7F
        else [1, 1, 1, 1, 1, 1, 1, 1]
        for i in s
    ]


def tokenize_one_hot(s: str) -> "list[list[int]]":    
    """
    return an array of array of 0 or 1 in one hot encoding the position of the caracter
    in utf8 encoding, if the caracter encoding is no longger than 1 byte, else encode as
    last position
    """
    #return torch.tensor([
    #    [1 if i == (ord(c) if ord(c) <= 0x7F else 0x80) else 0 for i in range(0x81)]
    #    for c in s
    #], dtype=torch.float)
    return [
        [1 if i == (ord(c) if ord(c) <= 0x7F else 0x80) else 0 for i in range(0x81)]
        for c in s
    ]


def load_json(path: "PathLike") -> list[dict]:
    r = []
    with scandir(path) as it:
        for entry in it:
            if entry.name[-5:] == ".json" and entry.is_file():
                with open(entry.path, "r", encoding="utf-8") as file:
                    r.append(json.load(file))
    return r


def load_json_iter(path: Path) -> Iterator[dict]:
    # with scandir(path) as it:
    #     for entry in it:
    #         if entry.name[-5:] == ".json" and entry.is_file():
    #             with open(entry.path, "r", encoding="utf-8") as file :
    #                 yield json.load(file)

    for item in path.iterdir():
        if item.name[-5:] == ".json" and item.is_file():
            with item.open("r", encoding="utf-8") as file:

                yield json.load(file)


def filter_length_json_iter(
    iterable, min_length=0, max_length=math.inf
) -> Iterator[str]:
    for i in iterable:
        t = i["text"]
        length = len(t)
        if length > min_length and length < max_length:
            yield t


class JSONDataset(IterableDataset):
    def __init__(
        self,
        dataset_folder: Union[Path, str],
        min_length: int,
        vector_length :int,
        tokenize_fn: Callable = tokenize_one_hot,
    ):
        super().__init__()
        self.source = Path(dataset_folder)
        self.tokenize_fn = tokenize_fn
        self.min_length = min_length
        self.vector_length = vector_length

    def __iter__(self):
        return (
            self.tokenize_fn(i.ljust(self.vector_length, ' '))
            for i in filter_length_json_iter(
                load_json_iter(self.source), self.min_length, self.vector_length
            )
        )

class FakeDataset(IterableDataset):
    def __init__(
        self,
        dataset_folder: Union[Path, str],
        generator: Generator,
        min_length: int,
        vector_length: int,
        tokenize_fn: Callable = tokenize_one_hot,
    ):
        super().__init__()
        self.source = Path(dataset_folder)
        self.generator = generator
        self.tokenize_fn = tokenize_fn
        self.min_length = min_length
        self.vector_length = vector_length

    def __iter__(self):

        # TODO: put bool rand in the returned generator

        # Fake
        text: str = next(self.generator)
        return (
            self.tokenize_fn(text.ljust(self.vector_length, ' '))
        )
        
        # if getrandbits(1):
        #     # Truth

        #     return (
        #         self.tokenize_fn(i.ljust(self.vector_length, ' '))
        #         for i in filter_length_json_iter(
        #             load_json_iter(self.source), self.min_length, self.vector_length
        #         )
        #     )

        # else:
        #     # Fake
        #     text: str = next(choice(self.dummy_generators))
        #     return (
        #         self.tokenize_fn(text.ljust(self.vector_length, ' '))
        #     )
