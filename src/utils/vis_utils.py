import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .data_utils import read_user_sessions_spark


def user_session_gantt(user_id: str) -> go.Figure:
    session = read_user_sessions_spark(user_id).reset_index()
    events_count = session.groupby("session_id").event_id.count().rename("events_count")
    session_start = session.groupby("session_id").event_ts.apply(min).rename("start_ts")
    session_end = session.groupby("session_id").event_ts.apply(max).rename("end_ts")
    gantt_df = pd.DataFrame(index=session_start.index)
    gantt_df["events_count"] = events_count
    gantt_df["start_ts"] = session_start
    gantt_df["end_ts"] = session_end
    gantt_df = gantt_df.reset_index()
    fig = px.timeline(gantt_df, x_start="start_ts", x_end="end_ts", y="session_id", text="events_count")
    return fig
