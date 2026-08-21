"""
Generates a simulated credit card transaction dataset for fraud analysis.

Real transaction-level data isn't publicly available for privacy reasons, and the
usual public fraud dataset (Kaggle's ULB dataset) has PCA-anonymized features which
can't be interpreted in business terms. This script builds a synthetic dataset with
named, interpretable features instead, with fraud probability driven by a mix of
known real-world fraud signals (spend anomalies, distance/travel patterns, odd hours,
risky merchant categories, velocity) plus noise, so the signal is discoverable but
not perfectly separable.

Amounts are in INR. Run with: python generate_data.py
"""

import numpy as np
import pandas as pd
from datetime import datetime

rng = np.random.default_rng(42)

N_CUSTOMERS = 8000
N_TRANSACTIONS = 100000
START_DATE = pd.Timestamp("2025-09-01")
END_DATE = pd.Timestamp("2026-02-28")

CATEGORIES = ["grocery", "restaurant", "gas_station", "online_retail", "entertainment",
              "utilities", "healthcare", "travel", "electronics", "jewelry", "gift_cards"]
CATEGORY_WEIGHTS = [0.25, 0.18, 0.12, 0.15, 0.08, 0.05, 0.05, 0.04, 0.04, 0.015, 0.025]
HIGH_RISK_CATEGORIES = {"online_retail", "electronics", "jewelry", "gift_cards", "travel"}

CATEGORY_AMOUNT_SCALE = {
    "grocery": 0.5, "restaurant": 0.4, "gas_station": 0.35, "online_retail": 1.1,
    "entertainment": 0.6, "utilities": 0.7, "healthcare": 1.3, "travel": 3.5,
    "electronics": 2.8, "jewelry": 4.5, "gift_cards": 1.0,
}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_customers():
    age = rng.normal(38, 12, N_CUSTOMERS).clip(18, 80).astype(int)
    tenure = rng.exponential(28, N_CUSTOMERS).clip(1, 180).astype(int)
    home_lat = rng.uniform(9.0, 33.0, N_CUSTOMERS)
    home_lon = rng.uniform(70.0, 96.0, N_CUSTOMERS)
    avg_amount = rng.lognormal(6.7, 0.55, N_CUSTOMERS).clip(200, 20000)
    activity_weight = rng.gamma(2.0, 1.0, N_CUSTOMERS)
    return pd.DataFrame({
        "customer_id": np.arange(1, N_CUSTOMERS + 1),
        "customer_age": age,
        "account_tenure_months": tenure,
        "home_lat": home_lat,
        "home_lon": home_lon,
        "avg_amount_30d": avg_amount,
        "activity_weight": activity_weight,
    })


def assign_customers(customers):
    p = customers["activity_weight"].values
    p = p / p.sum()
    idx = rng.choice(len(customers), size=N_TRANSACTIONS, p=p)
    return customers.iloc[idx].reset_index(drop=True)


def sample_hour():
    hour_weights = np.array([1, 1, 1, 1, 1, 2, 4, 7, 9, 9, 9, 9,
                              10, 10, 9, 9, 9, 10, 11, 11, 9, 6, 3, 2], dtype=float)
    hour_weights /= hour_weights.sum()
    return rng.choice(24, size=N_TRANSACTIONS, p=hour_weights)


def sample_timestamps():
    total_days = (END_DATE - START_DATE).days
    day_offset = rng.uniform(0, total_days, N_TRANSACTIONS)
    hour = sample_hour()
    minute = rng.integers(0, 60, N_TRANSACTIONS)
    second = rng.integers(0, 60, N_TRANSACTIONS)
    ts = START_DATE + pd.to_timedelta(day_offset, unit="D")
    ts = ts.floor("D") + pd.to_timedelta(hour, unit="h") + \
        pd.to_timedelta(minute, unit="m") + pd.to_timedelta(second, unit="s")
    return ts, hour


def build():
    customers = make_customers()
    txns = assign_customers(customers)
    n = len(txns)

    timestamp, hour = sample_timestamps()
    txns["timestamp"] = timestamp

    category = rng.choice(CATEGORIES, size=n, p=CATEGORY_WEIGHTS)
    txns["merchant_category"] = category

    # channel probabilities shift a bit by category
    channel = np.empty(n, dtype=object)
    online_leaning = np.isin(category, ["online_retail", "gift_cards", "entertainment"])
    for leaning, probs in [(True, [0.35, 0.60, 0.05]), (False, [0.78, 0.17, 0.05])]:
        mask = online_leaning == leaning
        channel[mask] = rng.choice(["card_present", "online", "atm_withdrawal"],
                                    size=mask.sum(), p=probs)
    txns["channel"] = channel

    # transaction location: usually near home, occasionally far (travel or fraud)
    far_draw = rng.random(n) < 0.06
    lat_jitter = np.where(far_draw, rng.normal(0, 4.5, n), rng.normal(0, 0.35, n))
    lon_jitter = np.where(far_draw, rng.normal(0, 4.5, n), rng.normal(0, 0.35, n))
    txns["txn_lat"] = (txns["home_lat"] + lat_jitter).clip(6, 37)
    txns["txn_lon"] = (txns["home_lon"] + lon_jitter).clip(66, 99)

    scale = category.astype(object)
    scale = np.array([CATEGORY_AMOUNT_SCALE[c] for c in category])
    noise = rng.lognormal(0, 0.5, n)
    txns["amount"] = (txns["avg_amount_30d"] * scale * noise).clip(50, 300000).round(2)

    txns["is_foreign_txn"] = (rng.random(n) < 0.02).astype(int)

    txns["transaction_id"] = np.arange(1, n + 1)

    # sort per customer by time to compute sequential features (used only to build
    # the fraud signal here -- the notebook re-derives these from raw columns itself)
    txns = txns.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
    grp = txns.groupby("customer_id")

    prev_ts = grp["timestamp"].shift(1)
    time_since_last_hrs = (txns["timestamp"] - prev_ts).dt.total_seconds() / 3600
    time_since_last_hrs = time_since_last_hrs.fillna(9999)

    prev_lat, prev_lon = grp["txn_lat"].shift(1), grp["txn_lon"].shift(1)
    dist_last = haversine_km(prev_lat, prev_lon, txns["txn_lat"], txns["txn_lon"])
    dist_last = dist_last.fillna(0)

    dist_home = haversine_km(txns["home_lat"], txns["home_lon"], txns["txn_lat"], txns["txn_lon"])

    # velocity: transactions by the same customer in the preceding 24h
    tmp = txns.set_index("timestamp")
    velocity = tmp.groupby("customer_id")["transaction_id"] \
        .rolling("24h", closed="left").count().fillna(0).reset_index(drop=True)

    amount_ratio = txns["amount"] / txns["avg_amount_30d"]
    implied_speed = dist_last / time_since_last_hrs.clip(lower=0.1)
    late_night = txns["timestamp"].dt.hour.isin([1, 2, 3, 4]).astype(int)
    is_online = (txns["channel"] == "online").astype(int)
    high_risk_cat = txns["merchant_category"].isin(HIGH_RISK_CATEGORIES).astype(int)
    new_account = (txns["account_tenure_months"] < 6).astype(int)

    def z(s):
        s = np.asarray(s, dtype=float)
        return (s - s.mean()) / (s.std() + 1e-9)

    logit = (
        -8.0
        + 1.15 * z(np.log1p(amount_ratio))
        + 0.85 * z(np.log1p(dist_home))
        + 0.95 * z(np.log1p(dist_last))
        + 0.55 * z(np.log1p(implied_speed))
        + 0.75 * late_night
        + 0.65 * is_online
        + 0.60 * high_risk_cat
        + 1.30 * txns["is_foreign_txn"]
        + 0.50 * z(velocity)
        + 0.35 * new_account
        + rng.normal(0, 0.9, n)
    )
    fraud_prob = sigmoid(logit)
    is_fraud = (rng.random(n) < fraud_prob).astype(int)
    txns["is_fraud"] = is_fraud

    # re-shuffle to chronological order globally, like a real processing log would be
    txns = txns.sort_values("timestamp").reset_index(drop=True)
    txns["transaction_id"] = np.arange(1, n + 1)

    out_cols = ["transaction_id", "customer_id", "timestamp", "amount", "merchant_category",
                "channel", "home_lat", "home_lon", "txn_lat", "txn_lon", "customer_age",
                "account_tenure_months", "avg_amount_30d", "is_foreign_txn", "is_fraud"]
    out = txns[out_cols].copy()
    for c in ["home_lat", "home_lon", "txn_lat", "txn_lon"]:
        out[c] = out[c].round(5)
    out["avg_amount_30d"] = out["avg_amount_30d"].round(2)

    return out


if __name__ == "__main__":
    df = build()
    df.to_csv("transactions.csv", index=False)
    print(f"rows: {len(df)}")
    print(f"fraud rate: {df['is_fraud'].mean():.4%}")
    print(f"date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"unique customers: {df['customer_id'].nunique()}")
