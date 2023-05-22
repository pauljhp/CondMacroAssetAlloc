from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Sequence, Union, Optional, Generator, Tuple
import pandas as pd
import torch
from utils import PreProcessor, sliding_window_iter


class MacroData(Dataset):
    def __init__(self, 
            window_size: int, 
            data: pd.DataFrame,
            # dtype: torchdtype=torch.float32,
            pad_val: float=-1e10) -> None:
        super().__init__()
        assert len(data) >= window_size, "data must be at least as logn as window_size!"
        self.data = data
        self.pad_val = pad_val
        self.window_size = window_size

    def __iter__(self) -> Generator: 
        yield sliding_window_iter(self.data, self.window_size)

    def __len__(self) -> int:
        return len(self.data) - self.window_size
    
    def __getitem__(self, idx) -> Tuple[int, np.array, np.array]:
        data = self.data.iloc[idx:idx+self.window_size]
        
        padding_mask = data.isna().sum(axis=1).apply(lambda x: True if x >=1 else False)
        data = data.fillna(self.pad_val)
        return idx, data.values, padding_mask.values


class AssetClassReturn(Dataset):
    def __inti__(self,
                #  window_size: int,
                 lookahead_period: int,
                 data: pd.DataFrame):
        """
        :param lookahead_period: # of months ahead for calculating the return
        :param data: dataset passed to the loader
        """
        super().__init__()
        self.data = data
        self.lookahead_period = lookahead_period

    def __iter__(self) -> Generator:
        """returns tuples of return and std of the return"""
        assert len(self.data) > self.lookahead_period, "num of periods in the data must be longer than the specified lookahead period!"
        for i in range(len(self.data) - self.lookahead_period):
            returns = self.data.iloc[i + self.lookahead_period] / self.data.iloc[i]
            yield tuple(returns.values)


    def __getitem__(self, idx: int) -> Tuple[float]:
        assert idx + self.lookahead_period <= len(self.data), "idx outside of the index range!"
        returns = self.data.iloc[idx + self.lookahead_period] \
            / self.data.iloc[idx]
        return tuple(returns.values)

