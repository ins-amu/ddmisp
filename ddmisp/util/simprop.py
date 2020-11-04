
import numpy as np
import scipy.optimize as opt

def expbilin(y, c, q):
    c1, c2 = -1, 1
    y1, y2 = 0, 1
    q11, q12, qa21, qa22 = q
    q21 = q11 + qa21
    q22 = q12 + qa22

    return np.exp(1./((c2 - c1)*(y2 - y1)) * (  q11*(c2 - c)*(y2 - y) + q21*(c - c1)*(y2 - y)
                                              + q12*(c2 - c)*(y - y1) + q22*(c - c1)*(y - y1)))

def prop(c, w, q):
    nreg = len(c)

    t = np.zeros(nreg)

    x = np.zeros(nreg, dtype=int)
    z = np.zeros(nreg, dtype=float)

    regs = np.r_[:nreg]

    time = 0.0
    for i in range(nreg):
        y = np.dot(w, x)
        fy = expbilin(y, c, q)

        # First to switch
        dts = np.divide(1.0 - z, fy, out=np.Inf * np.ones_like(fy), where=fy!=0)

        mask = x < 0.5
        dt, reg_to_switch = min(zip(dts[mask], regs[mask]))

        if np.isinf(dt):
            raise ValueError("Numerical error")

        z += dt * fy

        time += dt
        x[reg_to_switch] = 1
        t[reg_to_switch] = time

    return t


def propinv(w, q, obsmask, t_obs, c_hid):
    t_obs = np.array(t_obs)
    c_hid = np.array(c_hid)

    nreg = w.shape[0]
    nobs = sum(obsmask)
    nhid = sum(~obsmask)

    assert len(obsmask) == nreg
    assert nobs == len(t_obs)
    assert nhid == len(c_hid)

    # Sort the onset times
    inds = np.argsort(t_obs)
    t_obs_sorted = t_obs[inds]
    reg_obs = np.where(obsmask)[0]
    reg_obs_sorted = reg_obs[inds]

    # Prepare everything...
    c = np.zeros(nreg)
    t = np.ones(nreg)
    c[~obsmask] = c_hid

    x = np.zeros(nreg, dtype=int)
    z = np.zeros(nreg, dtype=float)
    ys = np.zeros((nreg, nreg), dtype=float)
    dts = np.zeros(nreg, dtype=float)

    mask = np.zeros(nreg, dtype=bool)
    mask[~obsmask] = True
    nobs_switched = 0
    nhid_switched = 0

    # Go for it
    time = 0.0
    for i in range(nreg):
        ys[i, :] = np.dot(w, x)
        fy = expbilin(ys[i], c, q)

        dt_ = np.full(nreg, np.inf)
        dt_[mask] = (1. - z[mask]) / fy[mask]

        reg_to_switch = np.argmin(dt_)
        dt = dt_[reg_to_switch]

        if (nobs_switched == nobs) or (time + dt < t_obs_sorted[nobs_switched]):
            # Hidden region is switching
            z += dt * fy
            mask[reg_to_switch] = False
            time += dt
            dts[i] = dt

        else:
            # Observed region is switching
            reg_to_switch = reg_obs_sorted[nobs_switched]
            dt = t_obs_sorted[nobs_switched] - time
            dts[i] = dt
            z += dt * fy
            c[reg_to_switch] = get_c(ys[:i+1, reg_to_switch], dts[:i+1], q)
            time += dt
            nobs_switched += 1

        x[reg_to_switch] = 1
        t[reg_to_switch] = time

    return c, t


def get_c(ys, dts, q):
    cost = lambda c: (np.sum(dts * expbilin(ys, c, q)) - 1.)**2
    res = opt.minimize(cost, x0=2.0, tol=1e-3, method='BFGS')
    if res.success:
        return res.x
    else:
        print("failure of minization")
        return np.nan
