from dataclasses import dataclass


@dataclass
class DataConfig:
    users_file: str
    users_dir: str
    train_file: str
    test_file: str
    train_folds_file: str

@dataclass
class ExperimentConfig:
    ...