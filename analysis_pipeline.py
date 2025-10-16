
import pandas as pd
import numpy as np
import math
import re
from statistics import multimode
from pathlib import Path

# ---- Config paths ----
SURVEY_CSV = "editorial_reading_survey_clean.csv"
CONTESTS_CSV = "clean_data.csv"
OUT_PER_CONTEST = "per_contest_clean.csv"
OUT_PER_USER = "per_user_summary.csv"
OUT_GROUP = "group_summary.csv"

def map_reader_bucket(x):
    if pd.isna(x): return np.nan
    x = int(x)
    if x in (1,2): return "non-readers (1–2)"
    if x == 3:     return "sometimes (3)"
    if x in (4,5): return "readers (4–5)"
    return np.nan

def trimmed_mean(arr, p=0.10):
    a = np.sort(arr[~np.isnan(arr)])
    n = len(a)
    if n == 0:
        return np.nan
    k = int(np.floor(p * n))
    if n - 2*k <= 0:
        return np.nan
    return a[k:n-k].mean()

def normalize_columns(df):
    def norm(s):
        return re.sub(r'[^a-z0-9]+', '', str(s).lower())
    return {norm(c): c for c in df.columns}

def load_survey(path):
    s = pd.read_csv(path)
    s.columns = [c.strip().lower().replace(" ", "_") for c in s.columns]
    s["reader_bucket"] = s["editorial_reading"].apply(map_reader_bucket)
    return s

def load_contests(path):
    raw = pd.read_csv(path)
    nm = normalize_columns(raw)
    def pick(cands):
        for k in cands:
            if k in nm: return nm[k]
        return None
    handle_col = pick(["handle","user","username","account","codeforceshandle"])
    contest_id_col = pick(["contestid","id","cid"])
    contest_name_col = pick(["contestname","name","contest"])
    start_time_col = pick(["starttime","start","startdate","ratingupdatetime","updatetime","date"])
    rank_col = pick(["rank","place","position"])
    old_rating_col = pick(["oldrating","old","prev","prerating","ratingbefore"])
    new_rating_col = pick(["newrating","new","post","postrating","ratingafter"])
    delta_col = pick(["ratingchange","delta","change","drating","diff"])

    df = pd.DataFrame()
    df["handle"] = raw[handle_col] if handle_col else np.nan
    df["contest_id"] = raw[contest_id_col] if contest_id_col else np.arange(len(raw))
    df["contest_name"] = raw[contest_name_col] if contest_name_col else np.nan
    df["start_time"] = pd.to_datetime(raw[start_time_col], errors="coerce") if start_time_col else pd.NaT
    df["rank"] = pd.to_numeric(raw[rank_col], errors="coerce") if rank_col else np.nan
    df["old_rating"] = pd.to_numeric(raw[old_rating_col], errors="coerce") if old_rating_col else np.nan
    df["new_rating"] = pd.to_numeric(raw[new_rating_col], errors="coerce") if new_rating_col else np.nan
    if delta_col:
        df["delta"] = pd.to_numeric(raw[delta_col], errors="coerce")
    else:
        df["delta"] = df["new_rating"] - df["old_rating"]
    return df

def analyze():
    survey = load_survey(SURVEY_CSV)
    contests = load_contests(CONTESTS_CSV)

    # Sort and exclude first 5 contests per handle
    contests = contests.sort_values(["handle","start_time","contest_id"], na_position="last").copy()
    contests["startup_rank"] = contests.groupby("handle")["start_time"].rank(method="first", ascending=True)
    contests["startup_outlier"] = contests["startup_rank"] <= 5

    main = contests[~contests["startup_outlier"]].copy()
    main = main[~main["delta"].isna()].copy()

    # Global IQR on remaining deltas
    if len(main) >= 4:
        q1 = main["delta"].quantile(0.25)
        q3 = main["delta"].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    else:
        lower = upper = np.nan

    main["iqr_outlier"] = False
    if not (np.isnan(lower) or np.isnan(upper)):
        main["iqr_outlier"] = (main["delta"] < lower) | (main["delta"] > upper)

    # Save per-contest clean
    main.to_csv(OUT_PER_CONTEST, index=False)

    # Per-user summaries
    rows = []
    for handle, g in main.groupby("handle"):
        arr_all = g["delta"].to_numpy()
        arr_rob = g.loc[~g["iqr_outlier"], "delta"].to_numpy()
        rows.append({
            "handle": handle,
            "n_contests_after_startup": len(g),
            "n_iqr_outliers": int(g["iqr_outlier"].sum()),
            "mean_delta_all": float(np.nanmean(arr_all)) if len(arr_all) else np.nan,
            "median_delta_all": float(np.nanmedian(arr_all)) if len(arr_all) else np.nan,
            "trimmed_mean_10_all": float(trimmed_mean(arr_all, 0.10)),
            "mean_delta_robust": float(np.nanmean(arr_rob)) if len(arr_rob) else np.nan,
            "median_delta_robust": float(np.nanmedian(arr_rob)) if len(arr_rob) else np.nan,
            "trimmed_mean_10_robust": float(trimmed_mean(arr_rob, 0.10)),
        })
    per_user = pd.DataFrame(rows)

    survey_small = survey[["handle","editorial_reading","reader_bucket","experience","weekly_time_on_codeforces"]]
    merged = per_user.merge(survey_small, on="handle", how="left")

    def summarize_group(x):
        vec = x["mean_delta_robust"].dropna().to_numpy()
        return pd.Series({
            "n_users": len(vec),
            "mean_of_means": np.nanmean(vec) if len(vec) else np.nan,
            "median_of_means": np.nanmedian(vec) if len(vec) else np.nan,
            "std_of_means": np.nanstd(vec, ddof=1) if len(vec) > 1 else np.nan
        })

    group_summary = merged.groupby("reader_bucket", dropna=False).apply(summarize_group).reset_index()

    merged.to_csv(OUT_PER_USER, index=False)
    group_summary.to_csv(OUT_GROUP, index=False)

if __name__ == "__main__":
    analyze()
