import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as smapi
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer

from scipy.stats.mstats import winsorize as c_winsorize
from scipy.signal import savgol_filter


def pca(df, target_ratio=0.8, **pca_kwargs) -> (pd.DataFrame, PCA, int):
    """
        Returns the PCA fit-transformed data. Suggested to perform normalization before fitting.
        Can call pca.explained_variance_ratio_ and pca.singular_values_ to obtain the parameters.
    """

    _pca = PCA(**pca_kwargs)
    _tmp = _pca.fit_transform(df.values)

    n_vars, target_remaining = 0, target_ratio
    while target_remaining > 0 and n_vars < len(df.columns):
        target_remaining -= _pca.explained_variance_ratio_[n_vars]
        n_vars += 1

    print(f"[Decomposition.PCA] It requires {n_vars} pc to explain {target_ratio - target_remaining:.3f} variance.")
    return pd.DataFrame(_tmp, index=df.index).iloc[:, :n_vars], _pca, n_vars


def dfm(df_normalized, k_factors=1, factor_order=1, error_order=1, plot_rsq=False):
    mod = smapi.tsa.DynamicFactor(endog=df_normalized,
                                  k_factors=k_factors,
                                  factor_order=factor_order,
                                  error_order=error_order)

    # initial maximization
    res_i = mod.fit(method='powell', disp=False)

    # recursive maximization
    res_f = mod.fit(res_i.params, disp=False)

    if plot_rsq:
        res_f.plot_coefficients_of_determination()
        plt.tight_layout()
        plt.show()

    return pd.Series(res_f.factors.filtered[0], index=df_normalized.index).rename('dfm').to_frame(), res_f

def winsorize_(self, q_lb=0.01, q_ub=0.99):
    clip_winsorize = self.quantile(q_lb), self.quantile(q_ub)
    clean_col = self.clip(*clip_winsorize)
    return clean_col


def normalize(df,
              ptr=None,
              winsorize=True,
              clip_range=None,
              renormalize=False,
              skip_col=None,
              verbose=False):
    """
        Returns the fit-transformed data using Yeo-Johnson Power Transform to tackle hetero-skedasticity.
        Then, clipping is applied to remove effect of extreme values and to reduce dimensionality.

        Some other transforms are coded but not used in the data, as the performance of the resulting model
        is not ideal.

        Input:
            @DataFrame df: the input dataframe,

        Returns:
            @DataFrame dat: the transformed dataframe.
    """

    if not clip_range:
        clip_range = (-99, 99)

    if skip_col is None:
        skip_col = []

    if not ptr:
        if verbose:
            print("[PowerTransformer] Creating new power transformer.")

        _ptr = PowerTransformer(method="yeo-johnson")
        _df = df.replace([np.inf, -np.inf], np.nan)
        _arr = _df.values

        if winsorize:
            _arr = c_winsorize(_arr, limits=[0.01, 0.01], axis=0)

        _arr_ptr = _ptr.fit_transform(_arr).clip(*clip_range)

        if verbose:
            print(f"[PowerTransformer] Fit and applied power transform.")

    else:
        if verbose:
            print("[PowerTransformer] Using existing transformer.")

        _df = df.replace([np.inf, -np.inf], np.nan)
        _arr = _df.values

        if winsorize:
            _arr = c_winsorize(_arr, limits=[0.01, 0.01], axis=0)

        _arr_ptr = ptr.transform(_arr).clip(*clip_range)

        if verbose:
            print(f"[PowerTransformer] Applied existing power transformer.")

        _ptr = ptr

    df_normalized = pd.DataFrame(data=_arr_ptr, index=df.index, columns=df.columns)
    return df_normalized, _ptr


def get_curve(ser, deg=2, plot_crv=False):
    assert isinstance(ser, pd.DataFrame), \
        "@ser should be a pandas DataFrame. Use .to_frame() to convert to df for a one-variable series."

    x = np.arange(len(ser))
    y = ser.values.reshape(-1)
    z = np.polyfit(x, y, deg)
    f = np.poly1d(z)

    y_new = f(x)
    y_out = pd.Series(y - y_new, index=ser.index).to_frame()

    if plot_crv:
        ax = pd.Series(y_new, index=ser.index).plot()
        pd.Series(y, index=ser.index).plot(ax=ax)
        plt.tight_layout()
        plt.show()

    return y_out


def identity(series, **kwargs):
    return series


def digitize(series, cutoff):
    return 1 if series > cutoff else 0


def nth(_iter, _i, preserve_none=True):
    if preserve_none:
        return [_item[_i] for _item in _iter]
    else:
        return [_item[_i] for _item in _iter if _item[_i] is not None]


def last_loc(items, pivot):
    return min([i for i in items if i <= pivot], key=lambda x: abs(x - pivot))


def savgol_pd(x, window_length, poly_order, deriv_order):
    """ Wrapper for the scipy.signal.savgol_filter for pandas object. """
    val = savgol_filter(np.ravel(x.dropna()), window_length, poly_order, deriv_order)
    return pd.DataFrame(val, index=x.dropna().index)
