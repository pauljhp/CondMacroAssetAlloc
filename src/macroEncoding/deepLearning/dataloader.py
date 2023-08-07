from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Sequence, Union, Optional, Generator, Tuple, Literal
import pandas as pd
import torch
from utils import PreProcessor, sliding_window_iter
from torch.nn.utils.rnn import pad_sequence

DAYS_IN_A_YEAR = 365.25

class MacroData(Dataset):
    def __init__(self, 
            data: pd.DataFrame,
            window_size: int=96, 
            # dtype: torchdtype=torch.float32,
            pad_val: float=-1e10) -> None:
        super().__init__()

        self.data = data
        assert len(data) >= window_size, "data must be at least as logn as window_size!"
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
    
    @classmethod
    def get_item_from_period(
        cls, 
        data: pd.DataFrame, 
        window_size: int,
        pad_val: float, 
        period: pd.offsets) -> Tuple[int, np.array, np.array]:
        inst = cls(data, window_size, pad_val)
        if period in inst.data.index:
            idx = inst.data.index.get_loc(period)
        else:
            left_mask = inst.data.index.asfreq("D") < period.asfreq("D")
            idx = left_mask.sum()
        return inst.__getitem__(idx)


class AssetClassReturn(Dataset):

    @staticmethod
    def to_period_index(
        df: pd.DataFrame, 
        dayfirst: bool=True, 
        freq: Literal["D", "W", "M", "Q", "Y"]="D"):
        df_ = df.copy(deep=True)
        df_.index = pd.to_datetime(df_.index, dayfirst=dayfirst)
        df_.index = df_.index.to_period(freq=freq)
        return df_

    def __init__(
            self,
            lookahead_period: int,
            lookback_period: int,
            price_data: pd.DataFrame,
            macro_window_size: int,
            macro_data: pd.DataFrame,
            dtype: torch.dtype,
            pad_val: float=-1e10,
            macro_dims: int=60,
            train_val_test_split: Tuple[float, float, float]=(.7, .1, .2),
            freq: Literal["D", "W", "M", "Q", "Y"]="D",
            mode: Literal["train", "val", "test"]="train",
            ignore_return: bool=False
            ):
        """
        :param lookahead_period: # of months ahead for calculating the return
        :param data: dataset passed to the loader
        """
        super().__init__()
        self._price_data_obj = PreProcessor(price_data, train_val_test_split=train_val_test_split)
        self._macro_data_obj = PreProcessor(macro_data, train_val_test_split=train_val_test_split)
        self._price_data = eval(f"self._price_data_obj._{mode}") # original price data
        self.price_data_ = eval(f"self._price_data_obj.{mode}") # normalized price data
        self.macro_data = eval(f"self._macro_data_obj.{mode}")
        self.price_data_ = self.to_period_index(self.price_data_, freq=freq)
        self._price_data = self.to_period_index(self._price_data, freq=freq)
        self.lookahead_period = lookahead_period
        self.lookback_period = lookback_period
        self._dtype = dtype
        self.pad_val = pad_val
        self.macro_window_size = macro_window_size
        self.macro_dims = macro_dims
        self.ignore_return = ignore_return
        

    def get_return(self, idx0: int, idx1: int):
        """idx0 and idx1 must be in the index"""
        days = (self.price_data_.index[idx1].asfreq("D") - self.price_data_.index[idx0].asfreq("D")).n
        returns = (self._price_data.iloc[idx1] / 
                self._price_data.iloc[idx0])
        returns = np.log(returns ** (DAYS_IN_A_YEAR / days))
        returns = returns.replace(float("inf"), 1e5).replace(float("-inf"), -1e5).values
        return returns

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype
    
    @dtype.setter
    def dtype(self, newdtype: torch.dtype) -> None:
        self._dtype = newdtype

    def __iter__(self) -> Generator[Tuple[torch.tensor, torch.tensor], None, None]:
        # FIXME - fix macro series to be the same as __getitem__
        """returns tuples of return and std of the return"""
        for i in range(len(self.price_data_) - self.lookahead_period):
            price_series = self.price_data_.iloc[i : i + self.lookahead_period].values
            price_padding = pd.DataFrame(np.isnan(price_series)).iloc[:, 0].values
            returns = self.get_return(i, i + self.lookahead_period)
            period = self.price_data_.index[i]
            _, macro_series, macro_padding = MacroData.get_item_from_period(self.macro_data, period=period, window_size=self.macro_window_size, pad_val=self.pad_val)
            yield (price_series, price_padding, returns, macro_series, macro_padding)

    def __getitem__(self, idx: int) -> Tuple[np.array, np.array, np.array, np.array]:
        assert idx + self.lookahead_period <= len(self.price_data_), "idx outside of the index range!"
        price_series = self.price_data_.iloc[idx : idx + self.lookback_period].fillna(-1e-10).values
        price_padding = pd.DataFrame(np.isnan(price_series)).iloc[:, 0].values
        if not self.ignore_return:
            returns = self.get_return(idx + self.lookback_period, idx + self.lookahead_period + self.lookback_period)
        period = self.price_data_.index[idx]
        _, macro_series, macro_padding = MacroData.get_item_from_period(
            self.macro_data, period=period, 
            window_size=self.macro_window_size, pad_val=self.pad_val)
        gap_size = self.macro_window_size - macro_series.shape[0]
        if gap_size > 0:
            macro_series_padding = np.array(
                [[self.pad_val for _ in range(self.macro_dims)] 
                 for _ in range(gap_size)])
            padding_mask = np.array([True for _ in range(gap_size)])
            macro_series = np.concatenate([macro_series_padding, macro_series])
            macro_padding_ = np.concatenate([macro_padding, padding_mask])
        else: macro_padding_ = macro_padding
        if not self.ignore_return:
            return (price_series, price_padding, returns, macro_series, macro_padding_) # FIXME - fix padding
        else:
            return (price_series, price_padding, macro_series, macro_padding_)


    def __len__(self) -> int:
        return self.price_data_.shape[0] - self.lookahead_period - self.lookback_period
