import os
import numpy as np
import h5py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.multioutput import MultiOutputRegressor 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib


def read_pod(pod_path):
    with h5py.File(pod_path, "r") as f:
        Sigma_U = f["Sigma_U"][...]   # shape: (nmode, 1, 3)
        A_U     = f["A_U"][...]       # shape: (nmode, nsnap, 3)
        Sigma_P = f["Sigma_P"][...]   # shape: (nmode, 1) or (nmode, 1)
        A_P     = f["A_P"][...]       # shape: (nmode, nsnap)
    return Sigma_U, A_U, Sigma_P, A_P

def truncate(mode_cutoff, dir_):
    Sigma_U, A_U, Sigma_P, A_P = read_pod(os.path.join(dir_, "POD.h5"))
    _, nsnap, nmode = A_U.shape
    r = min(mode_cutoff, nmode)

    A_Ur = A_U[:, :, :r]          # (3, nsnap, r)
    A_Pr = A_P[:, :r]             # (nsnap, r)
    Sigma_Ur = Sigma_U[:, :, :r]  # (3, 1, r)
    Sigma_Pr = Sigma_P[:, :r]     # (1, r)

    c_Ux, c_Uy, c_Uz = A_Ur[0, :, :], A_Ur[1, :, :], A_Ur[2, :, :]
    C_Ux, C_Uy, C_Uz,  = c_Ux.T, c_Uy.T, c_Uz.T # (r,nsnap)
    c_P = A_Pr
    C_P = c_P.T # (r,nsnap)
    return r, nsnap, C_Ux, C_Uy, C_Uz, C_P

def snapshot_params(lambda1s, lambda2s, thetas):
    P = []
    for l1 in lambda1s:
        for l2 in lambda2s:
            for th in thetas:
                P.append([float(l1), float(l2), float(th)])
    X = np.array(P, dtype=np.float32).T  # shape: (3, nsnap)
    return X

def snapshot_col_index(lambda1s, lambda2s, thetas, l1, l2, th):
    """
    Matches snapshot_params where loops are:
        for l1 in lambda1s:
            for l2 in lambda2s:
                for th in thetas:
                    ...
    i.e., θ varies fastest, then λ2, and λ1 varies slowest.
    Returns 0-based column index.
    """
    i1 = lambda1s.index(l1)   # λ1 (slowest)
    i2 = lambda2s.index(l2)   # λ2
    iθ = thetas.index(th)     # θ (fastest)
    return i1 * (len(lambda2s) * len(thetas)) + i2 * len(thetas) + iθ

def split_data(training, nsnap, Xparams, C, seed=123):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(nsnap)
    ntrain = int(np.floor(training * nsnap))
    itrain, ival = idx[:ntrain], idx[ntrain:]
    Xtr,  Xval  = Xparams[:, itrain], Xparams[:, ival]
    Ytr, Yval = C[:, itrain], C[:, ival]
    return Xtr, Xval, Ytr, Yval

def make_model(deg):
    return Pipeline([
        ("poly",   PolynomialFeatures(degree=deg, include_bias=False, interaction_only=False)),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("bayes",  MultiOutputRegressor(BayesianRidge(compute_score=True)))
    ])

def train_test_split_keep_edges(X, Y, train_frac=0.8, random_state=123):
    n = len(X)
    assert n >= 4, "Need at least 4 samples to keep first/last two in training."

    # Indices forced into the training set
    forced_idx = np.array([0, 1, n-2, n-1])

    # Indices eligible for shuffling / validation
    other_idx = np.array([i for i in range(n) if i not in forced_idx])

    X_other = X[other_idx]
    Y_other = Y[other_idx]

    # What would train/val sizes be on the FULL data?
    n_train_nominal = int(round(train_frac * n))
    n_val_nominal   = n - n_train_nominal

    # We will take exactly n_val_nominal validation samples from the inner ones
    if n_val_nominal > len(other_idx):
        raise ValueError(
            f"Not enough non-edge samples ({len(other_idx)}) for desired "
            f"validation size {n_val_nominal}."
        )

    # Split ONLY the "other" indices, taking an *absolute* number of val samples
    Xtr_o, Xval, Ytr_o, Yval = train_test_split(
        X_other, Y_other,
        test_size=n_val_nominal,        # absolute count, NOT fraction
        random_state=random_state,
        shuffle=True
    )

    # Final training set = forced edges + the remaining inner samples
    Xtr = np.vstack([X[forced_idx], Xtr_o])
    Ytr = np.vstack([Y[forced_idx], Ytr_o])

    return Xtr, Xval, Ytr, Yval

def fit_field_model(Xparams, C_field, deg, train_frac=0.8, random_state=456, keep_edges=False):
    """
    Xparams: (3, nsnap)
    C_field: (r, nsnap)  -> we train Y = C_field.T as (nsnap, r)
    """
    X = Xparams.T                     # (nsnap, r)
    Y = C_field.T                     # (nsnap, r)

    if keep_edges:
        Xtr, Xval, Ytr, Yval = train_test_split_keep_edges(
            X, Y, train_frac=train_frac, random_state=random_state
        )
    else:
        Xtr, Xval, Ytr, Yval = train_test_split(
            X, Y, train_size=train_frac, random_state=random_state, shuffle=True
        )

    model = make_model(deg)
    model.fit(Xtr, Ytr)

    Ŷtr = model.predict(Xtr)
    Ŷval = model.predict(Xval)

    rmse_tr  = np.sqrt(mean_squared_error(Ytr,  Ŷtr))
    rmse_val = np.sqrt(mean_squared_error(Yval, Ŷval))

    # optional normalized MSE (per-field)
    def _nmse(y, yhat):
        denom = np.maximum(np.mean(y**2), np.finfo(np.float32).eps)
        return mean_squared_error(y, yhat) / denom

    nmse_tr  = _nmse(Ytr,  Ŷtr)
    nmse_val = _nmse(Yval, Ŷval)

    metrics = dict(rmse_tr=rmse_tr, rmse_val=rmse_val, nmse_tr=nmse_tr, nmse_val=nmse_val)
    return model, metrics

def train_all_fields(Xparams, C_Ux, C_Uy, C_Uz, C_P, deg, keep_edges=True, outdir=None):
    models = {}
    metrics = {}

    for name, C in [("Ux", C_Ux), ("Uy", C_Uy), ("Uz", C_Uz), ("P", C_P)]:
        m, met = fit_field_model(Xparams, C, deg, keep_edges=keep_edges)  # (r, nsnap)
        models[name]  = m
        metrics[name] = met
        print(f"[{name}] RMSE(tr)={met['rmse_tr']:.4g}, RMSE(val)={met['rmse_val']:.4g}, "
              f"NMSE(tr)={met['nmse_tr']:.4g}, NMSE(val)={met['nmse_val']:.4g}")

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        for k, m in models.items():
            joblib.dump(m, os.path.join(outdir, f"BayesRidge_{k}.joblib"))
        print(f"✅ Saved models to: {outdir}")

    return models, metrics

# --------------------
# Prediction utilities
# --------------------
def predict_coeffs(models, λ1, λ2, θ):
    """
    Returns 4 vectors of length r:
      aUx, aUy, aUz, aP  (predicted POD coeffs for that parameter)
    """
    x = np.array([[float(λ1), float(λ2), float(θ)]], dtype=np.float32)
    aUx = models["Ux"].predict(x).ravel()
    aUy = models["Uy"].predict(x).ravel()
    aUz = models["Uz"].predict(x).ravel()
    aP  = models["P"].predict(x).ravel()
    return aUx, aUy, aUz, aP

def load_models(outdir):
    models = {}
    for k in ["Ux","Uy","Uz","P"]:
        models[k] = joblib.load(os.path.join(outdir, f"BayesRidge_{k}.joblib"))
    return models

def parity_plot(yhat, ytrue, path, xlabel, ylabel):
    yhat = np.asarray(yhat); ytrue = np.asarray(ytrue)
    lo = min(yhat.min(), ytrue.min()); hi = max(yhat.max(), ytrue.max())
    plt.figure(figsize=(6,6))
    plt.scatter(yhat, ytrue, s=10)
    plt.plot([lo,hi],[lo,hi], 'k-', lw=2)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()

def save_predictions_h5(
    out_path,
    aUx_pred, aUy_pred, aUz_pred, aP_pred,
    pred_params,
    λ1_recs, λ2_recs, θ_recs
):  
    """
    Writes predictions & metadata to an HDF5 file that Julia can read.
    Shapes:
      a*_pred: (r, Nmax)
      pred_params: (Nmax, 3) as float32 [λ1, λ2, θ] per column
      λ1_recs, λ2_recs, θ_recs: vectors for reference
      pred_on_grid / true_on_grid (optional): dict with keys "Ux","Uy","Uz","P", each (r, nsnap)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with h5py.File(out_path, "w") as f:
        grp_pred = f.create_group("pred")
        grp_pred.create_dataset("aUx", data=aUx_pred, compression="gzip")
        grp_pred.create_dataset("aUy", data=aUy_pred, compression="gzip")
        grp_pred.create_dataset("aUz", data=aUz_pred, compression="gzip")
        grp_pred.create_dataset("aP",  data=aP_pred,  compression="gzip")

        # pack params as (Nmax,3)
        params = np.array(pred_params, dtype=np.float32)
        grp_pred.create_dataset("params", data=params, compression="gzip")

        grp_grid = f.create_group("grid")
        grp_grid.create_dataset("λ1_recs", data=np.asarray(λ1_recs, dtype=np.float32))
        grp_grid.create_dataset("λ2_recs", data=np.asarray(λ2_recs, dtype=np.float32))
        grp_grid.create_dataset("θ_recs",  data=np.asarray(θ_recs,  dtype=np.float32))


# ========
#  main()
# ========
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    poddir     = os.path.join(script_dir, "data", "rotor_BiotSimulation", "POD")
    models_dir = os.path.join(script_dir, "data", "rotor_BiotSimulation", "surrogate")
    parity_dir = os.path.join(script_dir, "tex", "errors")
    datadir    = os.path.join(script_dir, "data", "rotor_BiotSimulation", "prediction")
    λ1s, λ2s, θs  = [3,4,5], [3,4,5], [0,15,30]
    λ1_recs, λ2_recs, θ_recs = [3,4,5], [3,4,5], [0,15,30]
    mode_cutoff   = 5
    degree        = 4
    train         = True   # << your switch: False = load models; True = train & save

    # ---- load/truncate POD ----
    print("Loading POD coefficients")
    r, nsnap, C_Ux, C_Uy, C_Uz, C_P = truncate(mode_cutoff, poddir)
    Xparams = snapshot_params(λ1s, λ2s, θs)  # (3, nsnap)
    print("Xparams shape:", Xparams.shape)
    print("Xparams:", Xparams)

    # ---- train or load BRR models ----
    if train:
        print("Training BRR models")
        models, _ = train_all_fields(Xparams, C_Ux, C_Uy, C_Uz, C_P, degree, keep_edges=True, outdir=models_dir)
    else:
        print("Loading BRR models")
        models = load_models(models_dir)

    # ---- prediction loop over (λ1_recs × λ2_recs × θ_recs) ----
    Nmax = len(λ1_recs)*len(λ2_recs)*len(θ_recs)
    aUx_pred = np.empty((r, Nmax), dtype=np.float32)
    aUy_pred = np.empty((r, Nmax), dtype=np.float32)
    aUz_pred = np.empty((r, Nmax), dtype=np.float32)
    aP_pred  = np.empty((r, Nmax), dtype=np.float32)
    pred_params = []

    all_pred = {"Ux": [], "Uy": [], "Uz": [], "P": []}
    all_true = {"Ux": [], "Uy": [], "Uz": [], "P": []}

    col = 0
    for l1 in λ1_recs:
        for l2 in λ2_recs:
            for th in θ_recs:
                aUx, aUy, aUz, aP = predict_coeffs(models, l1, l2, th)
                aUx_pred[:, col] = aUx
                aUy_pred[:, col] = aUy
                aUz_pred[:, col] = aUz
                aP_pred[:,  col] = aP
                pred_params.append((l1,l2,th))

                # if this (l1,l2,th) exists in the original snapshot grid, collect for parity
                if (l1 in λ1s) and (l2 in λ2s) and (th in θs):
                    j = snapshot_col_index(λ1s, λ2s, θs, l1, l2, th)  # column in C_* (r,nsnap)
                    all_pred["Ux"].extend(aUx.tolist()); all_true["Ux"].extend(C_Ux[:, j].tolist())
                    all_pred["Uy"].extend(aUy.tolist()); all_true["Uy"].extend(C_Uy[:, j].tolist())
                    all_pred["Uz"].extend(aUz.tolist()); all_true["Uz"].extend(C_Uz[:, j].tolist())
                    all_pred["P"] .extend(aP.tolist());  all_true["P"] .extend(C_P[:,  j].tolist())
                col += 1
    out_h5 = os.path.join(datadir, f"PredCoeffs.h5")
    save_predictions_h5(
        out_h5,
        aUx_pred, aUy_pred, aUz_pred, aP_pred,
        pred_params,
        λ1_recs, λ2_recs, θ_recs
    )
    print(f"💾 Saved predictions to: {out_h5}")

    # ---- parity plots (pred vs true on-grid) ----
    print("Making parity plots")
    parity_plot(all_pred["Ux"], all_true["Ux"], os.path.join(parity_dir, "Parity_Ux_Bayesian.pdf"),
                r"$a_{\star,i}^{(\overline{u}_x)}$", r"$a_{i}^{(\overline{u}_x)}$")
    parity_plot(all_pred["Uy"], all_true["Uy"], os.path.join(parity_dir, "Parity_Uy_Bayesian.pdf"),
                r"$a_{\star,i}^{(\overline{u}_y)}$", r"$a_{i}^{(\overline{u}_y)}$")
    parity_plot(all_pred["Uz"], all_true["Uz"], os.path.join(parity_dir, "Parity_Uz_Bayesian.pdf"),
                r"$a_{\star,i}^{(\overline{u}_z)}$", r"$a_{i}^{(\overline{u}_z)}$")
    parity_plot(all_pred["P"],  all_true["P"],  os.path.join(parity_dir, "Parity_P_Bayesian.pdf"),
                r"$a_{\star,i}^{(\overline{P})}$",  r"$a_{i}^{(\overline{P})}$")

    # print("✅ Done.")
    # print("aUx_pred shape:", aUx_pred.shape)
    # np.set_printoptions(precision=4, suppress=True, linewidth=200)
    # print("aUx_pred matrix:\n", aUx_pred)
    # return (aUx_pred, aUy_pred, aUz_pred, aP_pred, pred_params)

if __name__ == "__main__":
    main()
