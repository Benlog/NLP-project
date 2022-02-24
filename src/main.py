from .utils import JSONDataset
from dotenv import load_dotenv
from pathlib import Path
from os import environ
from torch.utils.data import DataLoader

root = Path("..")
load_dotenv(root)

data = root / Path(environ.get("DATA_PATH"))

dataset = JSONDataset()

loader = DataLoader(dataset, batch_size=10, pin_memory=True)