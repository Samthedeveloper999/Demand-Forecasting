import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from metrics import rmse, mape
import warnings

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def load_series(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    df = df.set_index("date").asfreq("D")
    df["sales"] = df["sales"].interpolate("linear").round().astype(int)
    return df

def grid_search_aic(y, pdq_list, seasonal_period=None):
    best = {"aic": np.inf, "order": None, "seasonal_order": None}
    for (p,d,q) in pdq_list:
        if seasonal_period:
            for P in [0,1]:
                for D in [0,1]:
                    for Q in [0,1]:
                        order_s = (P,D,Q,seasonal_period)
                        try:
                            model = SARIMAX(y, order=(p,d,q), seasonal_order=order_s, enforce_stationarity=False, enforce_invertibility=False)
                            res = model.fit(disp=False, maxiter=500)
                            if res.aic < best["aic"]:
                                best = {"aic": float(res.aic), "order": (p,d,q), "seasonal_order": order_s}
                        except Exception:
                            continue
        else:
            try:
                model = SARIMAX(y, order=(p,d,q), enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(disp=False, maxiter=500)
                if res.aic < best["aic"]:
                    best = {"aic": float(res.aic), "order": (p,d,q), "seasonal_order": None}
            except Exception:
                continue
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to daily_sales.csv")
    ap.add_argument("--horizon", type=int, default=90, help="forecast horizon in days")
    ap.add_argument("--val_days", type=int, default=60, help="validation window size")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    ensure_outdir(args.outdir)
    df = load_series(args.input)
    y = df["sales"]
    train = y.iloc[:-args.val_days]
    val = y.iloc[-args.val_days:]

    pdq = [(p,d,q) for p in [0,1,2] for d in [0,1] for q in [0,1,2]]
    best = grid_search_aic(train, pdq_list=pdq, seasonal_period=7)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model = SARIMAX(y, order=best["order"], seasonal_order=best["seasonal_order"],
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False, maxiter=500)

    res_val = res.get_prediction(start=val.index[0], end=val.index[-1], dynamic=False)
    val_pred = res_val.predicted_mean.reindex(val.index)
    val_rmse = rmse(val.values, val_pred.values)
    val_mape = mape(val.values, val_pred.values)

    fut = res.get_forecast(steps=args.horizon)
    f_mean = fut.predicted_mean
    f_ci = fut.conf_int(alpha=0.05)
    f = pd.DataFrame({
        "date": f_mean.index,
        "forecast": f_mean.values,
        "lower": f_ci.iloc[:,0].values,
        "upper": f_ci.iloc[:,1].values
    })

    f.to_csv(os.path.join(args.outdir, "forecast.csv"), index=False)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as fp:
        json.dump({"rmse": val_rmse, "mape": val_mape, "arima_order": best["order"], "seasonal_order": best["seasonal_order"], "aic": best["aic"]}, fp, indent=2)

    # Plot history + forecast
    fig, ax = plt.subplots(figsize=(12,4))
    y.plot(ax=ax, label="history")
    f_mean.plot(ax=ax, label="forecast")

    # Convert datetime to numeric for fill_between
    x = mdates.date2num(pd.to_datetime(f["date"]).dt.to_pydatetime())
    lower = f["lower"].astype(float).to_numpy()
    upper = f["upper"].astype(float).to_numpy()
    ax.fill_between(x, lower, upper, alpha=0.2, label="95% CI")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    ax.set_title("Daily Sales: History & Forecast")
    ax.set_xlabel("Date"); ax.set_ylabel("Units")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "fig_history_forecast.png"), dpi=160)
    plt.close(fig)

    # Residual diagnostics
    resid = res.resid
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(resid.index, resid.values)
    ax.set_title("Residuals (model fit on full data)")
    ax.set_xlabel("Date"); ax.set_ylabel("Residual")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "fig_residuals.png"), dpi=160)
    plt.close(fig)

    print("[OK] Forecasting complete.")
    print(f"Best order: {best}")
    print(f"Validation RMSE={val_rmse:.3f}, MAPE={val_mape:.2f}%")
    print(f"Outputs saved to: {args.outdir}")

if __name__ == "__main__":
    main()
