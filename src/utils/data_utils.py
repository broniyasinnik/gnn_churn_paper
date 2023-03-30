from functools import partial
from pathlib import Path
from typing import List, Union, Dict

import pandas as pd
import os

from ..settings import DATA_ROOT, SCRIPTS_ROOT

from ..db.orm import Users
from ..db.db_utils import get_engine
from ..db.queries import data_query
from tqdm import trange
import subprocess


def download_users_data(
    save_path: str, batch: int = 1_000_000, total: int = 156_493_941
):
    """
    Download full table from Vertica by multiple queries and chunking the data into smaller
    pieces
    :param save_path:
    :param batch:
    :param total:
    :return:
    """
    engine = get_engine()
    with engine.connect() as connection, open(save_path, "at") as f:
        for i in trange(total // batch + 1):
            query = data_query(offset=i * batch, limit=batch)
            res = connection.execute(query)
            records = res.fetchall()
            df = pd.DataFrame.from_records(records)
            df.to_csv(f, header=False, index=False)


def build_small_dataset(
    size=10, small_dir: str = "small", download_sessions: bool = True
):
    dataset = pd.read_csv(DATA_ROOT / "full" / "users.csv")
    users = dataset.groupby("user_id")["label"].first()
    sampled_users = pd.concat(
        [
            users[users == 1].sample(size // 2),
            users[users == 0].sample(size - size // 2),
        ]
    )
    small_dataset = dataset[dataset["user_id"].isin(sampled_users.index)]
    # Save small dataset of users
    os.makedirs(DATA_ROOT / small_dir, exist_ok=True)
    small_dataset.to_csv(DATA_ROOT / small_dir / "users.csv", index=False)

    # Save the sessions of each user in dataset
    if download_sessions:
        subprocess.run(str(SCRIPTS_ROOT / "download_sessions.sh"), check=True)

    return small_dataset


def read_user_sessions_spark(user_id: str, sessions_dir: str) -> pd.DataFrame:
    path_lst = (DATA_ROOT / sessions_dir / f"user_id={user_id}").glob("*.csv")
    df = pd.concat(
        map(partial(pd.read_csv, header=0, names=Users.columns.keys()[1:]), path_lst)
    )
    df.event_id = df.event_id.fillna(0).astype("int")
    df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True)
    df = df.sort_values("event_ts")

    return df


def read_user_data(user_id: str, install_date: str, users_dir: str, horizon: int = 2):
    user_df = pd.read_parquet(
        DATA_ROOT / users_dir / install_date.replace("-", "") / f"{user_id}.parquet"
    )
    horizon = user_df["event_ts"].min()+pd.Timedelta(days=2)
    user_df = user_df.query("event_ts <= @horizon")
    return user_df


