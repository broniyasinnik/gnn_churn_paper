from src.query import session_query, users_query, events_query, labels_query
from src.data_utils import execute_query


def test_session_query():
    q = session_query()
    params = {'start_date': '2021-12-01',
              'end_date': '2021-12-15'}
    result = execute_query(q, params)
    assert True


def test_users_query():
    q = users_query()
    print(q)


def test_events_query():
    q = events_query()
    print(q)


def test_labels_query():
    q = labels_query()
    print(q)
