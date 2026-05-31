import numpy as np
import pandas as pd


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def build_features(df):
    data = df.copy()

    data["transaction_time"] = pd.to_datetime(data["transaction_time"], errors="coerce")
    data["hour"] = data["transaction_time"].dt.hour.fillna(0).astype(int)
    data["dayofweek"] = data["transaction_time"].dt.dayofweek.fillna(0).astype(int)
    data["month"] = data["transaction_time"].dt.month.fillna(0).astype(int)
    data["is_night"] = data["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(int)

    data["amount_log"] = np.log1p(data["amount"])
    data["distance_km"] = haversine(
        data["lat"],
        data["lon"],
        data["merchant_lat"],
        data["merchant_lon"],
    )

    data["city_pop_log"] = np.log1p(data["population_city"])

    data["full_name"] = data["name_1"].astype(str) + "_" + data["name_2"].astype(str)

    features = [
        "amount",
        "amount_log",
        "population_city",
        "city_pop_log",
        "lat",
        "lon",
        "merchant_lat",
        "merchant_lon",
        "distance_km",
        "hour",
        "dayofweek",
        "month",
        "is_night",
        "is_weekend",
        "merch",
        "cat_id",
        "gender",
        "us_state",
        "jobs",
        "full_name",
    ]

    return data[features]
