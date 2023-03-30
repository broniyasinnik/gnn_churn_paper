from collections import Counter, OrderedDict

import os
import networkx as nx
import numpy as np
import pandas as pd
from itertools import groupby, chain
from typing import Dict
from ..settings import DATA_ROOT

from src.utils.data_utils import read_user_sessions_spark

EVENTS_DIM = 142


def create_activity_seq(x):
    vec = np.zeros(EVENTS_DIM + 1)
    vec[x["event_id"]] = x["count"]
    vec /= vec.sum()
    return vec[1:]


def build_user_activity_sequence(user_df: pd.DataFrame):
    activity_seq = (
        user_df.groupby("new_session_id")
        .event_id.value_counts()
        .rename("count")
        .to_frame()
        .reset_index()
    )
    activity_seq = activity_seq.groupby("new_session_id").apply(create_activity_seq)
    return np.stack(activity_seq)


def markov_session_graphs(usr_sessions_df: pd.DataFrame):
    graphs = OrderedDict.fromkeys(usr_sessions_df.index)
    for session_id, events in usr_sessions_df.groupby("new_session_id"):
        G = nx.DiGraph()
        # G.add_nodes_from(set(events.event_id))
        G.add_nodes_from(np.arange(EVENTS_DIM))
        events = events.sort_values("event_ts")
        edges = list(zip(events.event_id.iloc[:-1], events.event_id.iloc[1:]))
        edges = [
            (node1, node2, weight) for (node1, node2), weight in Counter(edges).items()
        ]
        G.add_weighted_edges_from(edges)
        graphs[session_id] = G
    return graphs


def build_user_markov_graphs(user_id: str, sessions_dir: str, days: int = 3):
    graphs = list(build_user_session_graphs(user_id, sessions_dir).items())
    sessions = pd.read_parquet(
        os.path.join(DATA_ROOT, sessions_dir, f"{user_id}.parquet")
    )
    sess_start = (
        sessions.groupby("new_session_id")["event_ts"].min()
        - sessions["event_ts"].min()
    ).dt.days
    temporal_graphs = [
        nx.DiGraph(1 / EVENTS_DIM * np.eye(EVENTS_DIM)) for _ in range(days)
    ]
    for day, sess_gs in groupby(graphs, lambda g: sess_start[g[0]]):
        G = nx.DiGraph()
        edges = chain.from_iterable([g.edges(data="weight") for _, g in sess_gs])
        self_edges = [(n, n, 1) for n in range(EVENTS_DIM)]
        edges = chain(edges, self_edges)
        df = pd.DataFrame(edges, columns=["n1", "n2", "w"])
        edges = df.groupby(["n1", "n2"])["w"].sum() / df["w"].sum()
        edges = [(n1, n2, w) for (n1, n2), w in edges.items()]
        G.add_weighted_edges_from(edges)
        temporal_graphs[day] = G
    return temporal_graphs


def build_user_markov_graphs_in_window(
    user_id: str, sessions_dir: str, window: int = 2
):
    graphs = list(build_user_session_graphs(user_id, sessions_dir).items())
    temporal_graphs = []
    for i in range(len(graphs) - window + 1):
        g_agg = nx.DiGraph()
        # Aggregation of graphs in window
        for idx, g in graphs[i : i + window]:
            g_agg.add_nodes_from(g.nodes)
            for u, v, w in g.edges.data("weight"):
                if (u, v) not in g_agg.edges:
                    g_agg.add_weighted_edges_from([(u, v, w)])
                else:
                    g_agg.edges[u, v]["weight"] += w
        # Probability weights for nodes.
        for node in g_agg.nodes:
            out_edges = g_agg.out_edges(node)
            total_weight = sum([g_agg.edges[u, v]["weight"] for u, v in out_edges])
            if total_weight != 0:
                for u, v in out_edges:
                    g_agg.edges[u, v]["weight"] /= total_weight

        temporal_graphs.append(g_agg)
    return temporal_graphs


def temporal_graph_from_session(session_df: pd.DataFrame) -> pd.DataFrame:
    nodes = session_df["node_label"].unique()
    edges = pd.DataFrame(
        np.transpose([np.tile(nodes, len(nodes)), np.repeat(nodes, len(nodes))]),
        columns=["node1", "node2"],
    )
    nodes_1 = (
        edges.merge(
            session_df[["node_label", "event_ts"]],
            how="left",
            left_on="node1",
            right_on="node_label",
        )
        .assign(node=1)
        .drop(columns="node_label")
    )
    nodes_2 = (
        edges.merge(
            session_df[["node_label", "event_ts"]],
            how="left",
            left_on="node2",
            right_on="node_label",
        )
        .assign(node=2)
        .drop(columns="node_label")
    )
    graph_raw = (
        pd.concat([nodes_1, nodes_2], axis=0)
        .drop_duplicates(subset=["node1", "node2", "event_ts"])
        .sort_values(["node1", "node2", "event_ts", "node"])
    )
    graph_processed = (
        graph_raw.assign(
            next_event_ts=graph_raw["event_ts"].shift(-1),
            next_node=graph_raw["node"].shift(-1),
            next_node1=graph_raw["node1"].shift(-1),
            next_node2=graph_raw["node2"].shift(-1),
        )
        .query("node==1 and next_node==2 and node1==next_node1 and node2==next_node2")
        .astype({"event_ts": "datetime64[ns]", "next_event_ts": "datetime64[ns]"})
        .assign(ts_diff=lambda x: (x["next_event_ts"] - x["event_ts"]).dt.seconds)[
            ["node1", "node2", "ts_diff"]
        ]
        .groupby(["node1", "node2"])
        .agg({"ts_diff": "mean"})
        .reset_index()
        .rename(columns={"node1": "u", "node2": "v", "ts_diff": "w"})
    )
    return graph_processed


def temporal_session_graphs(
    usr_sessions_df: pd.DataFrame,
    game_graph_df: pd.DataFrame,
    events_arr: np.ndarray,
    node_label: str = "event_id",
) -> OrderedDict[nx.Graph]:
    graphs = OrderedDict.fromkeys(usr_sessions_df.index)
    usr_sessions_df["node_label"] = usr_sessions_df[node_label]
    for session_id, session_df in usr_sessions_df.groupby("new_session_id"):
        G = nx.DiGraph()
        G.add_nodes_from(events_arr)

        edges = temporal_graph_from_session(session_df)
        edges = edges.merge(
            game_graph_df[["event_1", "event_2", "avg_ts_diff"]],
            how="left",
            left_on=["u", "v"],
            right_on=["event_1", "event_2"],
        ).drop(columns=["event_1", "event_2"])
        edges_norm = edges.assign(
            w=(edges["w"] / edges["avg_ts_diff"]).replace([np.inf, -np.inf, np.nan], 0)
        )[["u", "v", "w"]]
        G.add_weighted_edges_from(edges_norm.to_records(index=False))
        graphs[session_id] = G

    return graphs


def build_user_session_graphs(
    user_df: pd.DataFrame,
    weight_type: str = "markov",
    node_label: str = "event_id",
    game_graph_df: pd.DataFrame = None,
    events_arr: np.ndarray = None,
) -> Dict[str, nx.Graph]:
    usr_sessions_df = (
        user_df.merge(
            user_df.groupby("new_session_id", as_index=False).agg(
                start_ts=pd.NamedAgg(column="event_ts", aggfunc="min"),
            )[["new_session_id", "start_ts"]],
            how="left",
            on="new_session_id",
        ).sort_values("start_ts")
    ).set_index("new_session_id")

    if weight_type == "markov":
        graphs_dict = markov_session_graphs(usr_sessions_df)
    elif weight_type == "temporal":
        graphs_dict = temporal_session_graphs(
            usr_sessions_df,
            game_graph_df=game_graph_df,
            events_arr=events_arr,
            node_label=node_label,
        )

    return graphs_dict
