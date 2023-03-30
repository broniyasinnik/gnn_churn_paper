import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from .prepare_data import build_basic_features, get_targets
from src.settings import DATA_ROOT


def calculate_metrics(ytrue, ypred):
    precision = precision_score(ytrue, ypred)
    recall = recall_score(ytrue, ypred)
    acc = accuracy_score(ytrue, ypred)
    f1 = f1_score(ytrue, ypred)
    return dict(precision=precision, recall=recall, f1=f1, accuracy=acc)


def train_baseline():
    df_folds = pd.read_csv(DATA_ROOT / "full" / "users_train_folds.csv")
    df_test = pd.read_csv(DATA_ROOT / "full" / "users_test.csv")
    # useful_features = [c for c in df_folds.columns if c not in ("id", "target", "kfold")]
    # object_cols = [col for col in useful_features if 'cat' in col]
    # df_test = df_test[useful_features]
    final_predictions = []
    for fold in range(4):
        xtrain = df_folds[df_folds.kfold != fold].reset_index(drop=True)
        xvalid = df_folds[df_folds.kfold == fold].reset_index(drop=True)
        xtest = df_test.copy()

        ytrain = get_targets(xtrain)
        yvalid = get_targets(xvalid)
        ytest = get_targets(xtest)

        xtrain = build_basic_features(xtrain).reset_index(drop=True)
        xvalid = build_basic_features(xvalid).reset_index(drop=True)
        xtest = build_basic_features(xtest).reset_index(drop=True)

        scale_transformer = MinMaxScaler()
        xtrain = scale_transformer.fit_transform(xtrain)
        xvalid = scale_transformer.transform(xvalid)
        xtest = scale_transformer.transform(xtest)

        # ordinal_encoder = OrdinalEncoder()
        # xtrain[object_cols] = ordinal_encoder.fit_transform(xtrain[object_cols])
        # xvalid[object_cols] = ordinal_encoder.transform(xvalid[object_cols])
        # xtest[object_cols] = ordinal_encoder.transform(xtest[object_cols])

        model = LogisticRegression(random_state=fold, n_jobs=4)
        model.fit(xtrain, ytrain)
        preds_valid = model.predict(xvalid)
        test_preds = model.predict(xtest)
        final_predictions.append(calculate_metrics(ytest, test_preds))

        precision = precision_score(yvalid, preds_valid)
        recall = recall_score(yvalid, preds_valid)
        f1 = f1_score(yvalid, preds_valid)
        acc = accuracy_score(yvalid, preds_valid)
        print(f"Fold #{fold}: Precision {precision}, Recall {recall}, f1 {f1}, accuracy {acc}")

    print(final_predictions)


if __name__ == "__main__":
    # The first stage is to split the data to train and test
    # create_train_test()
    # Create 5 folds of the trainning data
    # create_train_folds()
    train_baseline()
