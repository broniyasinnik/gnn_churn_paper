from src.graph_utils import (
    read_user_sessions,
    build_user_session_graphs,
    build_user_temporal_graphs
)


def test_read_user_session():
    df = read_user_sessions('420007117')
    assert True


def test_build_user_session_graphs():
    graphs = build_user_session_graphs('420007117')
    assert False


def test_build_user_temporal_graphs():
    graphs = build_user_temporal_graphs('420007117')
    assert False
