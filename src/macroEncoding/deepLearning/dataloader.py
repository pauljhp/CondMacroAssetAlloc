from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Sequence, Union, Optional, Generator
import pandas as pd
import torch
from utils import PreProcessor, sliding_window_iter


class MacroData(Dataset):
    def __init__(self, 
            window_size: int, 
            data: pd.DataFrame,
            dtype: torch.dtype=torch.float32,
            pad_val: float=-1e10) -> None:
        super().__init__()
        assert len(data) >= window_size, "data must be at least as logn as window_size!"
        self.data = data
        self.data = self.data.fillna(pad_val)
        self.window_size = window_size

    def __iter__(self) -> Generator: 
        yield sliding_window_iter(self.data, self.window_size)

    def __len__(self) -> int:
        return len(self.data) - self.window_size
    
    def __getitem__(self, idx) -> np.array:
        return self.data.iloc[idx:idx+self.window_size].values




