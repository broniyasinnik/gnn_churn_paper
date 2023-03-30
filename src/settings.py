import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(os.path.dirname(__file__)).joinpath("..").absolute()
ASSETS_ROOT = PROJECT_ROOT.joinpath("assets")
DATA_ROOT = ASSETS_ROOT.joinpath("data")
LOGS_ROOT = ASSETS_ROOT.joinpath("logs")
SCRIPTS_ROOT = PROJECT_ROOT.joinpath("scripts")
SQL_ROOT = ASSETS_ROOT.joinpath("sql")
CONFIG_ROOT = ASSETS_ROOT.joinpath("config")
UTCNOW = datetime.utcnow().strftime("%y%m%d.%H%M%S")
REMOTE_DATA_ROOT = "/alluxio/training/datalake/gnn_churn/full_data.csv/"
