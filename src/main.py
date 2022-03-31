from os import environ
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from .RRMC import discrimNN
from .utils import JSONDataset, load_json_iter


def main():
    root = Path(".")
    load_dotenv(root / Path(".env"))

    data = root / Path(str(environ.get("DATA_PATH"))) / Path("french_articles")

    # max_length = max(len(doc["text"]) for doc in load_json_iter(data))
    # min_length = min(len(doc["text"]) for doc in load_json_iter(data))

    dataset = JSONDataset(data, min_length=50, vector_length=100)

    loader = DataLoader(dataset, batch_size=10, collate_fn=lambda x: torch.tensor(x, dtype=torch.float), pin_memory=True)

    loader_iter = iter(loader)

    data1 = next(loader_iter)
    #print(data1)
    print(data1.size())

    nn = discrimNN(129, 10)
    ft = nn.forward(data1)

    print(ft)
    print(ft.size())

    print("Done")
