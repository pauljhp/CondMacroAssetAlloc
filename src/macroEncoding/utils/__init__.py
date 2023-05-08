"""utility functions for macro encoding"""
import scipy
import numpy as np
import pandas as pd
from typing import Literal, Union, Optional, Sequence, Tuple, Generator, Any
import itertools
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from copy import deepcopy
import torch


Number = Union[int, float, bool, np.int_, np.float_]

def stationarity_test(
        sequence: Sequence[Number], 
        test_mode: Literal["ADF", "KPSS"]="ADF",
        pvalue_threshold: float=0.05,
        verbose: bool=False) -> bool:
    """conducts stationarity test"""
    match test_mode:
        case "ADF" | "adf":
            test_fn = lambda x: adfuller(x)
        case "KPSS" | "kpss":
            test_fn = lambda x: kpss(x)
        case _:
            raise ValueError(f"test_mode {test_mode} not supported")
    res = test_fn(sequence)
    if verbose:
        print(f"{test_mode} statistics:\n{res[0]};\np-value = {res[1]};\ncritical values: {res[2]}")
    if res[1] <= pvalue_threshold:
        return True
    else: return False

def get_difference(input: Sequence[Number], offset: int=1):
    res = deepcopy(input)
    res[offset:] = np.log(input[offset:] / input[:-offset])
    res[:offset] = [np.nan for _ in res[:offset]]
    return res



class PreProcessor:
    def __stationarize(self, data: pd.DataFrame, replace_inf: Number=1e10) -> Tuple[pd.DataFrame, pd.Series]:
        stationarized = deepcopy(data)
        original_starts_ = data.iloc[:, 0]
        for colid, col in data.items():
            is_stationary = self.stationarity_mask[colid]
            if not is_stationary[0]:
                stationarized[colid] = get_difference(col.values, self.differencing_offset)
        stationarized = stationarized.replace(float("inf"), replace_inf)
        stationarized = stationarized.replace(float("-inf"), -replace_inf)
        return stationarized, original_starts_

    
    def __normalize(self, data: pd.DataFrame, fit_data: bool=False) -> pd.DataFrame:
        """:param data: must be subset of self.data"""
        if fit_data:
            normalized = pd.DataFrame(self.normalizer.fit_transform(data),
                                    index=data.index,
                                    columns=data.columns)
        else: 
            normalized = pd.DataFrame(self.normalizer.transform(data),
                                    index=data.index,
                                    columns=data.columns)
        
        return normalized
    
    @staticmethod
    def train_val_test_split(
            split: Tuple[float, float, float], 
            data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_ratio = split[0]
        val_ratio = split[0] + split[1]
        def get_idx(portion: float, length: int=len(data)) -> int:
            return int(length * portion // 1)
        train_idx, val_idx = get_idx(train_ratio), get_idx(val_ratio)
        train, val, test = data.iloc[:train_idx], data.iloc[train_idx: val_idx], data.iloc[val_idx:]
        return (train, val, test)

    def __init__(self, 
            data: pd.DataFrame,
            differencing_offset: int=1,
            train_val_test_split: Tuple[float, float, float]=(0.7, 0.1, 0.2)):
        self.data = deepcopy(data)
        self.data = self.data.dropna()
        self.normalizer = StandardScaler(with_mean=True, with_std=True)
        self.stationarity_mask = self.data.apply(lambda x: stationarity_test(x), axis=0)
        self.differencing_offset = differencing_offset

        train, val, test = self.train_val_test_split(split=train_val_test_split, data=self.data)
        train_, original_starts_ = self.__stationarize(train)
        self.original_starts = original_starts_
        train_ = self.__normalize(train_, fit_data=True)
        self.mean_ = self.normalizer.mean_
        self.var_ = self.normalizer.var_

        val_, _ = self.__stationarize(val)
        val_ = self.__normalize(val_)

        test_, _ = self.__stationarize(test)
        test_ = self.__normalize(test_)

        self.train, self.val, self.test = train_, val_, test_

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (self.train, self.val, self.test)

def iter_by_chunk(iterable: Any, chunk_size: int):
    """iterate by chunk size"""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, chunk_size))
        if not chunk:
            break
        yield chunk

def padding(x: torch.tensor, direction: str='left',
    pad_value: Union[float, int, str]=0.,
    repeat: int=1):
    """padd a tensor with a value on one direction
    :param x: input to be padded
    :param direction: takes 'left' or 'right'. For 'up' or 'down', just tranpose
        the data
    :param pad_value: value to pad data with
    :param repeat: number of times to repeat the padding
    """
    if direction == 'left':
        return torch.cat([torch.tensor(pad_value).repeat(
                x.shape[0], repeat, ), x], dim=1)
    elif direction == 'right':
        return torch.cat([x, torch.tensor(pad_value).repeat(
                x.shape[0], repeat, )], dim=1)
    

def sliding_window_iter(data, window_size: int, enumeration: bool=False) -> Generator:
    """iterate with a sliding window"""
    assert len(data) >= window_size, \
        "window size cannot be longer than self.data!"
    for i, _ in enumerate(data):
        if len(self.data) - i >= window_size:
            if enumeration:
                yield i, data[i: i + window_size]
            else:
                yield data[i: i + window_size]