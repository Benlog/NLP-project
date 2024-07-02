from os import environ
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch import nn
from torch.utils.data import DataLoader

from .RRMC import discrimNN
from .utils import JSONDataset, load_json_iter

learning_rate = 1e-3
batch_size = 64
epochs = 5

def main():
    root = Path(".")
    load_dotenv(root / Path(".env"))

    data = root / Path(str(environ.get("DATA_PATH"))) / Path("french_articles")

    # max_length = max(len(doc["text"]) for doc in load_json_iter(data))
    # min_length = min(len(doc["text"]) for doc in load_json_iter(data))

    dataset = JSONDataset(data, min_length=50, vector_length=100)

    #loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=lambda x: torch.tensor(x, dtype=torch.float), pin_memory=True)
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, collate_fn=lambda x: torch.tensor(x, dtype=torch.float), pin_memory=True)

    """ data1 = next(loader_iter)
    #print(data1)
    print(data1.size())
 """
    model = discrimNN(129, 10)
    loss = nn.MSELoss()
    epoch = 5

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    model.train()

    for epo in range(epoch):
        for i, x in enumerate(loader):
            y = torch.ones(64)
            optimizer.zero_grad()

            output = model(x)
            output = loss(output, y)

            print(f"Epoch {epo}, batch {i} : loss {output}")

            output.backward()
            optimizer.step()

            #print([sum(p.data) for p in optimizer.param_groups[0]['params']])

    print("Done")
