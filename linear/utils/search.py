import itertools
from functools import partial, reduce
from typing import Dict, Iterable, Any, Callable

import numpy as np
import pandas as pd
import statsmodels.api as smapi
from pathos.helpers import cpu_count
from pathos.multiprocessing import _ProcessPool as Pool
from statsmodels.stats.stattools import durbin_watson

from utils.transform import nth

CPU_COUNT = cpu_count()


def forward_selection(data_train,
                      target_var,
                      max_num_vars=8,
                      shift=None,
                      exclusion_list=None,
                      n_stop=2000,
                      threshold=0.01,
                      verbose=False):

    from queue import PriorityQueue

    df = data_train.copy()
    if exclusion_list:
        df = df.loc[~df.columns.isin(exclusion_list)]

    if shift:
        df.loc[:, target_var] = df.loc[:, target_var].shift(-abs(shift))

    X_train, y_train = df.loc[:, ~df.columns.isin([target_var])], df.loc[:, [target_var]]
    X_train = smapi.add_constant(X_train)
    frontier = PriorityQueue()
    frontier.put((0, ['const', ], [i for i in X_train.columns.to_list() if i != 'const']))
    results = []
    explored = []

    while frontier.queue:
        if verbose:
            print(f"[MODEL SELECTOR] frontier len: {len(frontier.queue)}, results len: {len(results)}")

        old_score, used_vars, remaining_vars = frontier.get()
        explored.append((used_vars, remaining_vars))

        for new_var in remaining_vars:
            training_vars = sorted(used_vars + [new_var, ])
            remaining_vars_new = sorted([v for v in remaining_vars if v != new_var])

            y_train_inst = y_train.dropna()
            X_train_inst = X_train.loc[:, training_vars].dropna()
            set_index = sorted(set(y_train_inst.index).intersection(X_train_inst.index))

            y_train_inst = y_train_inst.loc[set_index]
            X_train_inst = X_train_inst.loc[set_index, :]

            ols = smapi.OLS(y_train_inst, X_train_inst).fit(cov_type='HC1')
            score = np.round(ols.rsquared_adj, 6)# - ((durbin_watson(ols.resid) - 2) ** 2) * 0.1

            if (score, training_vars) not in results:
                results.append((score, training_vars))

            if len(training_vars) <= max_num_vars + 1:
                if (training_vars, remaining_vars_new) not in explored and \
                   (-score, training_vars, remaining_vars_new) not in frontier.queue and \
                   (score - -old_score) > threshold:
                    frontier.put((-score, training_vars, remaining_vars_new))

        if len(results) >= n_stop:
            break

    return results


def ensemble_execute(func: Callable, cases=None, *args, **kwargs):
    """
    Create partial function from args and kwargs.
    Call partial function with additional args & kwargs in cases and return the result.
    """
    assert cases is not None, "cases must not be None when executing ensemble"
    efunc = partial(func, *args, **kwargs)

    print(f"[Ensemble executor] Ensemble initiated on {len(cases)} cases.")

    with Pool(CPU_COUNT) as pool:
        async_fut = [pool.apply_async(efunc, args=(), kwds=_c) for _c in cases]
        async_res = [_fut.get() for _fut in async_fut]

    print(f"[Ensemble executor] Ensemble completed.")

    return async_res


def ensemble_create_cases(dicts: Dict[str, Iterable[Any]]):
    """
    Create combinations of every element in every iterables.

    >> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1}, {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1}, {'character': 'b', 'number': 2}]
    """
    return list(dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def ensemble_extract_results(ensemble_result, lb=None, ub=None):
    """
    A shorthand to concat of [nth(arr, i) for i in range(lb, ub)].
    Concat on axis=1 an iterable of pd.Series into pd.DataFrame.
    """
    res_ = []

    if lb is None or ub is None:
        lb, ub = 0, len(ensemble_result[0])

    for rdim_index in range(lb, ub):
        objs = nth(ensemble_result, rdim_index, preserve_none=False)
        if len(objs):
            res = pd.concat(objs, axis=1)
            res.columns = list(map(str, range(len(objs))))
        else:
            res = pd.DataFrame()
        # res = reduce(lambda x, y: x.join(y.iloc[:, 0].rename(str(len(x.columns))).to_frame(), how='outer'),
        #              nth(ensemble_result, rdim_index))
        # res.columns = ["0", ] + list(res.columns[1:])
        res_.append(res)

    return res_


def ensemble_get_weighted_average(ensemble_scores, ensemble_acc, cutoff=0.8):
    # select ensemble results with persistent zscore > 0.8
    # get numer := weighted zscore by acc, denom := sum of weights

    def nan_map(i):
        return 1 if not pd.isna(i) else np.nan

    last_acc = ensemble_acc.iloc[-1]
    numerator = (ensemble_scores
                 .apply(lambda x: x * last_acc.map(lambda i: i if i >= cutoff else np.nan), axis=1)
                 .sum(axis=1))

    denominator = (ensemble_scores.apply(
        lambda x: (x.apply(nan_map) * last_acc.map(lambda i: i if i >= cutoff else np.nan)), axis=1).sum(axis=1))
    df_z_score = (numerator / denominator).dropna().rename('predicted').to_frame()
    return df_z_score


def ensemble_get_average(ensemble_result, idx):
    return reduce(lambda x, y: x + y, nth(ensemble_result, idx)) / len(nth(ensemble_result, idx))


def ensemble_get_weight_config_view(ensemble_result, ensemble_cases: dict, n_top=5):
    """
    Return the ensemble weight, related configs and the Black-Litterman view of top @n_top items.
    """
    if not len(ensemble_result):
        return

    e_return = ensemble_extract_results(ensemble_result, lb=0, ub=1)[0]
    cumulative_return = e_return.cumsum()
    r_loc = list(cumulative_return.iloc[-1].sort_values(ascending=False).iloc[:n_top].index)

    e_wgt = [wgt for (i, wgt) in enumerate(nth(ensemble_result, 1)) if str(i) in r_loc]
    e_blv = [view for (i, view) in enumerate(nth(ensemble_result, 2)) if str(i) in r_loc]
    e_opt_wgt = [opt_wgt for (i, opt_wgt) in enumerate(nth(ensemble_result, 3))
                 if str(i) in r_loc and opt_wgt is not None]

    e_wgt = list(map(lambda x: x.div(x.sum(axis=1), axis=0), e_wgt))
    e_wgt = sum(e_wgt) / len(e_wgt)

    if len(e_opt_wgt):
        e_opt_wgt = sum(e_opt_wgt) / len(e_opt_wgt)
    else:
        e_opt_wgt = None

    e_top_cases = [(i, f"{cumulative_return.iloc[-1, i]:.4f}", case) for (i, case) in enumerate(ensemble_cases)
                   if str(i) in r_loc]

    e_blv_out = sum(e_blv) / len(e_blv)
    # e_blv_out = e_blv_out.resample("MS").first()

    return e_wgt, e_top_cases, e_blv_out, e_opt_wgt


def ensemble_get_attribution(ensemble_result):

    attribution = list(map(lambda x: x[2]['attribution'], ensemble_result, ))
    attribution = [i for i in attribution if i.columns.to_list() == attribution[0].columns.to_list()]
    union_index = reduce(pd.DatetimeIndex.union, list(map(lambda x: x.index, attribution)))
    attribution = list(map(lambda x: x.reindex(union_index), attribution))
    count_mask = sum(list(map(lambda x: ~x.applymap(pd.isna), attribution)))
    attribution = sum(list(map(lambda x: x.fillna(0), attribution))) / count_mask

    # direction
    direction = list(map(lambda x: x[2]['coef'].iloc[-1], ensemble_result, ))[0].map(np.sign)

    return attribution, direction


def feature_to_index(x, feature_to_index_map):
    list_repr = [feature_to_index_map[i] for i in x if i != 'const']
    one_hot = [0, ] * len(feature_to_index_map)
    for i in list_repr:
        one_hot[i] = 1

    return np.array(one_hot)


if __name__ == '__main__':
    pass