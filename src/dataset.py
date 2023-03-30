import os.path
from typing import Optional

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .utils.data_utils import read_user_data

from .prepare_data import (
    build_basic_features,
    create_train_folds,
    create_train_test,
    download_users_file,
    get_targets,
)
from .settings import DATA_ROOT
from .utils.graph_utils import build_user_activity_sequence, build_user_session_graphs


def pad_collate(batch):
    out_batch = dict()
    if "features" in batch[0]:
        features = torch.stack([elem["features"] for elem in batch])
        out_batch["features"] = features
    if "target" in batch[0]:
        target = torch.stack([elem["target"] for elem in batch])
        out_batch["target"] = target
    if "activity_seq" in batch[0]:
        act_seqs = list(elem["activity_seq"] for elem in batch)
        act_seqs_lens = [seq.shape[0] for seq in act_seqs]
        act_seqs_pad = pad_sequence(act_seqs, batch_first=True, padding_value=0)
        out_batch["activity_seq"] = act_seqs_pad
        out_batch["activity_seq_lens"] = act_seqs_lens
    if "action_graphs" in batch[0]:
        graphs_seq = [elem["action_graphs"] for elem in batch]
        max_graph_seq_len = max(map(len, graphs_seq))
        num_nodes = graphs_seq[0][0].num_nodes()
        pad_graph = dgl.add_self_loop(dgl.graph([], num_nodes=num_nodes))
        pad_graph.ndata["w"] = torch.zeros((num_nodes, num_nodes))
        out_batch["action_graphs"] = [
            dgl.batch([seq[i] if i < len(seq) else pad_graph for seq in graphs_seq])
            for i in range(max_graph_seq_len)
        ]
        if "graph_patterns" in batch[0]:
            graph_patterns_seq = [b["graph_patterns"] for b in batch]
            # Pad patterns to match the maximal length of sequence
            out_batch["graph_patterns"] = [
                [seq[i] if i < len(seq) else pd.Series([], dtype=float) for seq in graph_patterns_seq]
                for i in range(max_graph_seq_len)]
            # Offset nodes in patterns to align with batch
            out_batch["graph_patterns"] = [
                pd.concat([s[i].apply(lambda n_seq: [n + i * num_nodes for n in n_seq]) for i in range(len(s))]) for s
                in
                out_batch["graph_patterns"]]

            out_batch["num_patterns"] = [[len(b[i]) if i < len(b) else 0 for b in graph_patterns_seq] for i in
                                         range(max_graph_seq_len)]

    if "patterns" in batch[0]:
        out_batch["patterns"] = torch.stack([elem["patterns"] for elem in batch])

    return out_batch


class UsersData(Dataset):
    def __init__(
            self,
            users_ind: pd.Series,
            features: np.array,
            targets: np.array,
            data_dir: str = None,
            users_dir: str = None,
            include_macro_features: bool = False,
            include_activity_seq: bool = False,
            include_action_graphs: bool = False,
            include_patterns_seq: bool = False,
            use_pattern_trajectories: bool = False,
            node_label: Optional[str] = None,
            node_features: Optional[str] = None,
    ):
        super().__init__()
        self.users_ind = users_ind
        self.features = features
        self.targets = targets
        self.users_dir = users_dir
        self.data_dir = data_dir

        # components to include in dataset
        self.include_macro_features = include_macro_features
        self.include_activity_seq = include_activity_seq
        self.include_action_graphs = include_action_graphs
        self.include_patterns_seq = include_patterns_seq

        # action graphs settings
        self.use_patterns_trajectories = use_pattern_trajectories
        self.node_label = node_label
        self.node_features = node_features

        if node_label == "event_id":
            bb_game_stats_file = "bb_game_event_stats.csv"
        else:
            bb_game_stats_file = "bb_game_intent_stats.csv"

        self.game_graph = pd.read_csv(
            os.path.join(DATA_ROOT, "auxiliary", bb_game_stats_file)
        )
        self.events_arr = pd.read_csv(
            os.path.join(DATA_ROOT, "auxiliary", "events_id_to_intent_originator.csv")
        )[node_label].unique()

        self.patterns_df = pd.read_csv(
            os.path.join(DATA_ROOT, self.data_dir, "patterns.csv")
        ).set_index(["user_id", "new_session_id"])

        self.patterns_mapping = pd.read_csv(
            os.path.join(DATA_ROOT, "auxiliary", "patterns_mapping.csv")
        ).set_index("pattern_index")

    def get_graphs_patterns_trajectories(self, user_id, graphs):
        graph_patterns_seq = list()
        for new_session_id, _ in graphs.items():
            patterns_index = (
                self.patterns_df.loc[(user_id, new_session_id)]
                .filter(regex="pattern")
                .loc[lambda x: x == 1]
                .index.str.extract("(\d+)")
                .astype("int")
                .rename(columns={0: "index"})["index"]
            )
            patterns_sess = (
                self.patterns_mapping.loc[patterns_index]["event_seq"]
                .str.split(",")
                .apply(
                    lambda x: [
                        np.argwhere(self.events_arr == int(n)).item() for n in x
                    ]
                )
            )
            graph_patterns_seq.append(patterns_sess)
        return graph_patterns_seq

    def __getitem__(self, idx):
        user_id, install_date, features, target = (
            self.users_ind.get_level_values("user_id")[idx],
            self.users_ind.get_level_values("install_date")[idx],
            self.features[idx],
            self.targets[idx],
        )
        user_df = read_user_data(
            user_id=user_id,
            install_date=install_date,
            users_dir=self.users_dir,
            horizon=2,
        )
        item = dict(target=torch.tensor(target, dtype=torch.float32))
        if self.include_macro_features:
            item["features"] = torch.tensor(features, dtype=torch.float32)
        if self.include_activity_seq:
            activity_seq = build_user_activity_sequence(user_df)
            item["activity_seq"] = torch.tensor(activity_seq, dtype=torch.float32)
        if self.include_action_graphs:
            graphs = build_user_session_graphs(
                user_df,
                weight_type=self.node_features,
                node_label=self.node_label,
                game_graph_df=self.game_graph,
                events_arr=self.events_arr,
            )
            n1 = [
                [np.argwhere(self.events_arr == e[0]).item() for e in g.edges]
                for g in graphs.values()
            ]
            n2 = [
                [np.argwhere(self.events_arr == e[1]).item() for e in g.edges]
                for g in graphs.values()
            ]
            w = [nx.to_numpy_array(g) for g in graphs.values()]
            item["action_graphs"] = [
                dgl.add_self_loop(dgl.graph((n1[i], n2[i]), num_nodes=w[0].shape[0]))
                for i in range(len(n1))
            ]
            for i, g in enumerate(item["action_graphs"]):
                g.ndata["w"] = torch.tensor(w[i], dtype=torch.float)

            if self.use_patterns_trajectories:
                item["graph_patterns"] = self.get_graphs_patterns_trajectories(user_id, graphs)

        if self.include_patterns_seq:
            patterns = torch.tensor(
                self.patterns_df.loc[[user_id]].filter(regex="pattern").max(axis=0),
                dtype=torch.float,
            )
            item["patterns"] = patterns

        return item

    def __len__(self):
        return self.users_ind.shape[0]


class UsersDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.users_df = (
            pd.read_csv(
                DATA_ROOT / self.cfg.data.users_file,
                parse_dates=["start_ts", "end_ts", "install_date"],
            )
            .assign(
                horizon=lambda x: (x["start_ts"] - x["install_date"]).dt.days,
                label=lambda x: np.where(x.label, "active", "churn"),
            )
            .query("horizon < 2")
        )
        self.train_data: UsersData = None
        self.val_data: UsersData = None
        self.test_data: UsersData = None

    def prepare_data(self):
        if not os.path.exists(DATA_ROOT / self.cfg.data.users_file):
            logger.error(
                f"File in {DATA_ROOT / self.cfg.data.users_file} doesn't exist!"
            )
            return
        train_test_files = [
            DATA_ROOT / self.cfg.data.train_file,
            DATA_ROOT / self.cfg.data.test_file,
            DATA_ROOT / self.cfg.data.train_folds_file,
        ]
        all_files_exist = all(map(os.path.exists, train_test_files))
        if not all_files_exist:
            create_train_test(self.users_df, self.cfg.data)
            df_train = pd.read_csv(DATA_ROOT / self.cfg.data.train_file)
            create_train_folds(df_train, self.cfg.data)

    def setup(self, fold: int, stage: Optional[str] = None):
        df_folds = pd.read_csv(DATA_ROOT / self.cfg.data.train_folds_file).rename(
            columns=lambda x: x.strip()
        )
        df_test = pd.read_csv(DATA_ROOT / self.cfg.data.test_file).rename(
            columns=lambda x: x.strip()
        )

        train_data = df_folds[df_folds.kfold != fold]
        valid_data = df_folds[df_folds.kfold == fold]
        test_data = df_test.copy()

        users_trn = build_basic_features(train_data)
        users_val = build_basic_features(valid_data)
        users_tst = build_basic_features(test_data)

        scale_transformer = MinMaxScaler()

        # Training data
        train_ind = users_trn.index
        feat_trn = scale_transformer.fit_transform(users_trn)
        y_trn = get_targets(train_data).to_numpy()

        # Validation data
        valid_ind = users_val.index
        feat_val = scale_transformer.transform(users_val)
        y_val = get_targets(valid_data).to_numpy()

        # Test data
        test_ind = users_tst.index
        feat_tst = scale_transformer.transform(users_tst)
        y_tst = get_targets(test_data).to_numpy()

        include_macro_features: bool = self.cfg.components.macro_features.include
        include_activity_seq: bool = self.cfg.components.activity_seq.include
        include_action_graphs: bool = self.cfg.components.action_graphs.include
        include_patterns_seq: bool = self.cfg.components.patterns_seq.include

        use_patterns_trajectories: bool = self.cfg.components.action_graphs.pattern_trajectory
        node_label: str = self.cfg.components.action_graphs.node_label
        node_features: str = self.cfg.components.action_graphs.node_features

        self.train_data = UsersData(
            train_ind,
            feat_trn,
            y_trn,
            users_dir=self.cfg.data.users_dir,
            data_dir=self.cfg.data.data_dir,
            include_macro_features=include_macro_features,
            include_activity_seq=include_activity_seq,
            include_action_graphs=include_action_graphs,
            include_patterns_seq=include_patterns_seq,
            use_pattern_trajectories=use_patterns_trajectories,
            node_label=node_label,
            node_features=node_features,
        )
        self.val_data = UsersData(
            valid_ind,
            feat_val,
            y_val,
            users_dir=self.cfg.data.users_dir,
            data_dir=self.cfg.data.data_dir,
            include_macro_features=include_macro_features,
            include_activity_seq=include_activity_seq,
            include_action_graphs=include_action_graphs,
            include_patterns_seq=include_patterns_seq,
            use_pattern_trajectories=use_patterns_trajectories,
            node_label=node_label,
            node_features=node_features,
        )
        self.test_data = UsersData(
            test_ind,
            feat_tst,
            y_tst,
            users_dir=self.cfg.data.users_dir,
            data_dir=self.cfg.data.data_dir,
            include_macro_features=include_macro_features,
            include_activity_seq=include_activity_seq,
            include_action_graphs=include_action_graphs,
            include_patterns_seq=include_patterns_seq,
            use_pattern_trajectories=use_patterns_trajectories,
            node_label=node_label,
            node_features=node_features,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            collate_fn=pad_collate,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.cfg.batch_size,
            collate_fn=pad_collate,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
            pin_memory=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.cfg.batch_size,
            collate_fn=pad_collate,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
            pin_memory=False,
        )

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size)
    #
    # def teardown(self, stage: Optional[str] = None):
    #     # Used to clean-up when the run is finished
    #     ...
