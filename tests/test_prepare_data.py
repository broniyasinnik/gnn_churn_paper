from src.prepare_data import split_train_test


def test_split_train_test():
    split_train_test(n_splits=5)
    assert False
