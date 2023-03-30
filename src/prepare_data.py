import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from .settings import DATA_ROOT, SQL_ROOT
from omegaconf import DictConfig
from src.db.db_utils import read_sql, get_engine


def download_users_file(cfg: DictConfig):
    engine = get_engine(SQL_ROOT/"bb_users.sql")
    query = read_sql()
    df = pd.read_sql(query, con=engine)
    df.to_csv(DATA_ROOT/cfg.users_file, index=False)


def create_train_test(users_df: pd.DataFrame, cfg: DictConfig):
    df = users_df[["user_id", "label"]].groupby(["user_id", "label"], as_index=False).first()
    X = df["user_id"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
    df_train = users_df[users_df["user_id"].isin(X_train)]
    df_test = users_df[users_df["user_id"].isin(X_test)]
    df_train.to_csv(DATA_ROOT / cfg.train_file, index=False)
    df_test.to_csv(DATA_ROOT / cfg.test_file, index=False)


def create_train_folds(train_df: pd.DataFrame, cfg: DictConfig):
    train_df["kfold"] = -1
    df = train_df[["user_id", "label"]].groupby(["user_id", "label"], as_index=False).first()
    X = df["user_id"]
    y = df["label"]

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    for fold, (train_indicies, valid_indicies) in enumerate(skf.split(X, y)):
        row_indexer = train_df["user_id"].isin(X[valid_indicies])
        train_df.loc[row_indexer, "kfold"] = fold
        # df_train.loc[valid_indicies, "kfold"] = fold

    train_df.to_csv(DATA_ROOT / cfg.train_folds_file, index=False)


def get_targets(users_df: pd.DataFrame):
    targets = users_df.groupby("user_id").label.first()
    targets = targets.map({"churn": 0, "active": 1})
    return targets


def build_basic_features(users_df: pd.DataFrame):
    # df = read_users_data(horizon=2)
    features = users_df.groupby(["user_id", "install_date"]).agg(
        sessions=pd.NamedAgg("new_session_id", "count"),
        duration_sum=pd.NamedAgg("length", "sum"),
        duration_mean=pd.NamedAgg("length", "mean"),
        events_sum=pd.NamedAgg("num_events", "sum"),
        events_mean=pd.NamedAgg("num_events", "mean"),
    )
    features = features.astype(float)

    return features


