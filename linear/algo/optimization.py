import time

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize


def _cpsd(mat: np.array):
    X = mat.copy()
    Y = (X + X.T) / 2
    D, Q = np.linalg.eigh(Y)
    D_pos = np.diag(D.clip(0))
    Z = Q @ D_pos @ Q.T
    return Z


def max_sharpe_cvxpy(price=None, roll_price=None, cov_base=None, upper_bound_default=0.3, upper_bound_dict=None, **kwargs):
    sel_ = roll_price.columns[(price.count() >= 3)]
    force_psd = kwargs.get('force_psd', False)

    sym = sel_.copy()
    ret_ = price[sel_].mean().values
    rret_ = roll_price[sel_].mean().values
    # rret_[np.where(ret_ < 0)] = rret_[np.where(ret_ < 0)].clip(max=0)
    cov_ = cov_base[sel_].cov(ddof=1).values
    dp_ = kwargs.get('dp', 1.)

    if upper_bound_dict is None:
        upper_bound_dict = dict()

    ub_arr = [upper_bound_dict.get(k, upper_bound_default) for k in sel_]

    x = cp.Variable(len(rret_))

    if force_psd:
        form = cp.Maximize(x.T @ rret_ - dp_ * cp.quad_form(x, _cpsd(cov_)))
    else:
        form = cp.Maximize(x.T @ rret_ - dp_ * cp.quad_form(x, cov_))

    prob = cp.Problem(
        form,
        [
            cp.sum(x) == 1,
            x >= 0.,
            x <= ub_arr,
        ],
    )
    prob.solve(
        solver=cp.ECOS,
        warm_start=True,
        abstol=1e-4,
        reltol=1e-3,
        feastol=1e-4,
        max_iters=20
    )

    if prob.status.startswith('optimal'):
        prob.success = True
    else:
        prob.success = False

    return dict(zip(list(sym), list((x.value / x.value.sum()).round(6)))), prob


def weight_func_sharpe(date=None,
                       price=None,
                       roll_price=None,
                       upper_bound_default: float = 0.3,
                       upper_bound_dict: dict = None,
                       **kwargs):
    # default args
    dp = kwargs.get("dp", 0)
    if upper_bound_dict is None:
        upper_bound_dict = dict()

    # data loading
    _sel = roll_price.columns[(price.count() >= 3)]
    ub_arr = [(0., upper_bound_dict.get(k, upper_bound_default)) for k in _sel]

    nm = roll_price[_sel].apply(np.mean, axis=0)
    nv = price[_sel].apply(np.var, axis=0)
    lw = len(_sel)
    ig = np.random.rand(lw)
    ig /= np.sum(ig)

    res = minimize(
        # lambda w: -10 * (w @ nm) - np.min([(w @ nm) / np.sqrt(w @ nv) / np.sqrt(52), dp]),
        lambda w: (w @ nm) / np.sqrt(w @ nv) - np.sqrt(dp) * (w @ nm),
        x0=ig,
        constraints=({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.}),
        bounds=ub_arr,
    )

    return dict(zip(list(_sel), list(res.x))), res


def weight_func_min_var(date=None,
                        price=None,
                        roll_price=None,
                        upper_bound_default: float = 0.3,
                        upper_bound_dict: dict = None,
                        **kwargs):

    # default args
    seed = int(time.time())
    rng = np.random.default_rng(seed=seed)
    dp = kwargs.get("dp", 0)
    if upper_bound_dict is None:
        upper_bound_dict = dict()

    # data loading
    _sel = roll_price.columns[(price.count() >= 3)]
    ub_arr = [(0., upper_bound_dict.get(k, upper_bound_default)) for k in _sel]

    nm = roll_price[_sel].apply(np.mean, axis=0)
    nv = price[_sel].apply(np.var, axis=0)
    lw = len(_sel)
    ig = rng.random(lw)
    ig /= np.sum(ig)

    res = minimize(
        lambda w: np.sqrt(w @ nv) - dp * (w @ nm),
        x0=ig,
        constraints=({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.}),
        bounds=ub_arr,
    )

    return dict(zip(list(_sel), list(res.x))), res

