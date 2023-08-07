import logging
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from statsmodels.api import OLS, add_constant

from utils.generic import verbose_func_call, ROOT_PATH
from utils.preprocessing import fred_get, from_csv
from utils.transform import normalize

# Options
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

# Global Variables
os.chdir(ROOT_PATH)

data_path = os.path.join(os.getcwd(), "data")
out_path = os.path.join(os.getcwd(), "out")


# Driver Functions
@verbose_func_call
def load_fred_data(list_input, io_fn, load_local=False, save_local=True):
    """
        Read data from FRED and returning metadata on data read
        Set @load_local to True to bypass load time (alleviate network load on tests).
        Set @save_local to True to save data to local ./data.
    """
    if load_local:
        fred_res = from_csv(os.path.join(data_path, io_fn))
        return fred_res[list_input]

    fred_res = fred_get(list_input)
    data_freq_inv = {var_input: pd.infer_freq(fred_res[var_input].dropna().index) for var_input in list_input}
    data_freq = {}
    for k, v in data_freq_inv.items(): data_freq[v] = data_freq.get(v, []) + [k]

    data_freq = {k: ', '.join(v) for k, v in data_freq.items()}
    print("[Data] --------- Data imported ---------")
    _ = [print(f"[Data] Freq {k}: Series {v}") for k, v in data_freq.items()]
    print("[Data] Preprocessing completed.")

    if save_local:
        fred_res.to_csv(os.path.join(data_path, io_fn), index=True)

    return fred_res


def data_train_val_split(data_diffed,
                         train_val_split_date=None,
                         get_truth=False,
                         skip_col=None,
                         **aug_kwargs):
    '''
    Perform training/validation split on data.
    Use get_truth to return the complete data transformed on the entire universe.
    Pass aug_kwargs to modify normalization behavior such as clip range.
    Pass skip_target as a list to skip certain items.
    '''
    # Generate normalized first difference (the normalization fit-transform is tuned on training data only)
    if not get_truth:
        assert train_val_split_date, "Train/validation split date cannot be None when doing training"
        data_train_pre, ptr = normalize(data_diffed.loc[data_diffed.index < train_val_split_date],
                                        skip_col=skip_col, **aug_kwargs)
        data_train_post, _ = normalize(data_diffed.loc[data_diffed.index >= train_val_split_date],
                                       ptr=ptr, skip_col=skip_col, **aug_kwargs)
        return data_train_pre, data_train_post
    else:
        data_truth, _ = normalize(data_diffed, ptr=None, **aug_kwargs)
        return data_truth


@verbose_func_call
def model_create(data_enriched,
                 params_inp,
                 target_var,
                 model_inception_date,
                 clip_range,
                 training_cutoff_date=None,
                 shift=0,
                 skip_norm_col=None,
                 renormalize=False,
                 **kwargs):
    """
    Create OLS model and returning metrics such as predictive power,
    and parameters such as coefficients.
    """
    assert params_inp == 'all' or (isinstance(params_inp, list) or isinstance(params_inp, tuple)), \
        "@params_inp must be an iterable base class like list of tuple."

    if "do_diff" in kwargs.keys():
        logging.warning('keyword argument @do_diff is deprecated.')

    if "period" in kwargs.keys():
        logging.warning('keyword argument @period is deprecated.')

    if params_inp == 'all':
        params_inp = [v for v in data_enriched.columns if v != target_var]

    if training_cutoff_date is not None:
        loc = (data_enriched.index >= model_inception_date) & (data_enriched.index <= training_cutoff_date)
    else:
        loc = (data_enriched.index >= model_inception_date)

    data_enriched = data_enriched[params_inp + [target_var, ]]
    inc_period = data_enriched[loc].index

    imse, omse, pred, pred_index, lrs, smapi_lr = [], [], [], [], [], []
    coef = pd.DataFrame(columns=params_inp, index=inc_period)

    data_truth = data_train_val_split(data_enriched, clip_range=clip_range, renormalize=renormalize, get_truth=True)
    data_truth.loc[:, 'res'] = data_truth[target_var].shift(-shift)

    # Incremental Training
    for inc_dt in inc_period:
        data_train_pre, data_train_post = data_train_val_split(data_enriched,
                                                               train_val_split_date=inc_dt,
                                                               clip_range=clip_range, 
                                                               skip_col=skip_norm_col,
                                                               renormalize=renormalize)

        # if fill_nan:
        #     data_train_pre = data_train_pre.fillna(0.)

        data_train_pre.loc[:, 'res'] = data_train_pre[target_var].shift(-shift)
        x_train, y_train = data_train_pre.dropna()[params_inp], data_train_pre.dropna()['res']
        x_train_na = data_train_pre[params_inp].dropna()

        # model creation
        lr = LinearRegression().fit(x_train, y_train)
        # y_train_pred = list(lr.predict(x_train_na))
        y_train_pred = lr.predict(x_train_na)

        ols = OLS(y_train, add_constant(x_train, prepend=False))
        lr_ = ols.fit()
        summary = lr_.summary()
        # y_train_pred = list(lr.predict(add_constant(x_train_na, prepend=False)))

        imse.append(mse(y_train, y_train_pred[:len(y_train)]))
        lrs.append(lr)
        smapi_lr.append(lr_)

        # prepend data for predi for first iteration for evaluation
        if not pred:
            pred = list(y_train_pred)
            pred_index = list(x_train_na.index)

        # coefficients savedown
        coef.loc[inc_dt] = list(lr.coef_)

        # model validation
        data_train_post.loc[:, 'res'] = data_train_post[target_var].shift(-shift)
        x_vali, y_vali = data_train_post.dropna()[params_inp], data_train_post.dropna()['res']
        x_vali_na = data_train_post[params_inp].dropna()
        y_pred = lr.predict(x_vali_na)
        pred.append(y_pred[0])
        pred_index.append(inc_dt + pd.DateOffset(months=shift))

        data_truth_inc = data_truth.loc[data_truth.index >= inc_dt]
        x_truth, y_truth = data_truth_inc.dropna()[params_inp], data_truth_inc.dropna()['res']
        x_truth_na = data_truth_inc[params_inp].dropna()

        # get out of sample prediction results
        y_vali_pred = lr.predict(x_vali_na)
        if len(y_truth):
            omse.append(mse(y_truth, y_vali_pred[:len(y_truth)]))

    last_snapshot = pd.concat([data_train_pre, data_train_post], axis=0)[params_inp].dropna(axis=0)
    pred_out = pd.Series(pred, index=pred_index, name="predicted_z-score").to_frame()
    trsq_out = pd.Series([lr.score(data_truth.dropna()[params_inp], data_truth.dropna()['res']) for lr in lrs],
                         index=inc_period, name='r-sq_of_model_at_time_t').to_frame()
    imse_out = pd.Series(imse, index=inc_period, name="mse_in_sample").to_frame()
    omse_out = pd.Series(omse, index=inc_period[:len(omse)], name="mse_out_of_sample").to_frame()
    attr_out = pd.DataFrame(
        ((last_snapshot * coef.iloc[-1]).values)[:len(pred_index)],
        columns=params_inp,
        index=pred_index
    )
    late_out = pd.Series(
        data=np.hstack([y_train_pred, y_pred]),
        index=(last_snapshot.index + pd.DateOffset(months=shift))[-len(np.hstack([y_train_pred, y_pred])):]
    )

    dict_out = {
        "coef": coef,
        "linear_model": lrs,
        "mse_in_sample": imse_out,
        "mse_out_of_sample": omse_out,
        "summary": summary,
        "ground_truth": data_truth,
        "smapi_model": smapi_lr,
        "attribution": attr_out,
        "last_sample": late_out,
    }

    return pred_out, trsq_out, dict_out
