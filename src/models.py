import torch
import dgl
from dgl.nn.pytorch import GraphConv
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torchmetrics import Accuracy, MetricCollection, Precision, Recall, F1Score


class MLP(nn.Module):

    def __init__(self, input_dims=64, hidden_dims=[128, 256], output_dims=10):
        super().__init__()
        hidden_dims = [input_dims] + hidden_dims + [output_dims]
        layers = []
        for idx in range(len(hidden_dims) - 1):
            layers += [
                nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]),
                nn.ReLU(inplace=True)
            ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LSTM(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=cfg["seq_dim"],
            hidden_size=cfg["hidden_dim"],
            num_layers=cfg["layers"],
            dropout=cfg["dropout"],
            batch_first=True,
        )
        self.linear = nn.Linear(cfg["hidden_dim"], cfg["linear_out_dim"])

    def forward(self, x):
        out, (ht, ct) = self.lstm(x)
        # output_padded, output_lengths = pad_packed_sequence(out, batch_first=True)
        last_hidden_out = ht[-1]
        linear_out = self.linear(last_hidden_out)
        return linear_out


class GNN(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(GNN, self).__init__()
        in_feats = cfg.gcn_in
        hidden_size = cfg.gcn_hid
        gcn_out = cfg.gcn_out
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, gcn_out)

    def forward(self, g, inputs, patterns=None, num_patterns=None):
        # g = dgl.add_self_loop(g)
        h = self.conv1(g, inputs)
        # print("First Conv:", h)
        h = F.relu(h)
        # print("RELU:", h)
        h = self.conv2(g, h)
        # print("Second Conv:", h)
        with g.local_scope():
            g.ndata['h'] = h
            if patterns is None:
                msgs = dgl.mean_nodes(g, 'h')
                return self.linear(msgs)
            else:
                msgs = [dgl.mean_nodes(g.subgraph(s), 'h') for s in patterns.values]
                if len(msgs) != 0:
                    msgs = torch.stack(msgs).split_with_sizes(num_patterns)
                    msgs = torch.stack([torch.sum(t, axis=0) for t in msgs]).reshape((g.batch_size, -1))
                    return self.linear(msgs)
                else:
                    return torch.zeros((g.batch_size, self.linear.out_features)).to(g.device)
            # print("msgs:", h)


class MultiChannelModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.macro_features = cfg.components.macro_features.include
        self.activity_seq = cfg.components.activity_seq.include
        self.action_graphs = cfg.components.action_graphs.include
        self.patterns = cfg.components.patterns_seq.include

        self.use_pattern_trajectories = cfg.components.action_graphs.pattern_trajectory

        out_dim = 0
        if self.macro_features:
            self.macro_features = True
            out_dim += cfg.model.macro_features_dim
        if self.activity_seq:
            self.lstm = LSTM(self.cfg.model.lstm_activity_seq)
            out_dim += cfg.model.lstm_activity_seq["linear_out_dim"]
        if self.action_graphs:
            self.gnn = GNN(cfg.model.gnn)
            self.gnn_lstm = LSTM(cfg.model.gcn_lstm)
            out_dim += cfg.model.gcn_lstm["linear_out_dim"]
        if self.patterns:
            self.patterns_mlp = MLP(**self.cfg.model.patterns_mlp)
            out_dim += cfg.model.patterns_mlp.output_dims

        self.classifier = nn.Linear(out_dim, 1)
        # Define metrics
        metrics = MetricCollection([Accuracy(task='binary'),
                                    Precision(task='binary'),
                                    Recall(task='binary'),
                                    F1Score(task='binary')
                                    ])
        self.train_metrics = metrics.clone()
        self.valid_metrics = metrics.clone()
        self.test_metrics = metrics.clone()

    def forward(self, batch, mode: str = "train"):
        targets = batch["target"].unsqueeze(1)
        outputs = list()
        if self.macro_features:
            features = batch["features"]
            outputs.append(features)
        if self.activity_seq:
            activity_seq = batch["activity_seq"]
            lstm_out = self.lstm(activity_seq)
            outputs.append(lstm_out)
        if self.action_graphs:
            graphs = batch["action_graphs"]
            if self.use_pattern_trajectories:
                pattern_graphs = batch["graph_patterns"]
                num_patterns = batch["num_patterns"]
                gnn_sequence = torch.stack([self.gnn(graphs[i],
                                                     graphs[i].ndata['w'],
                                                     pattern_graphs[i],
                                                     num_patterns[i],
                                                     ) for i in range(len(graphs))], 1)
            else:
                gnn_sequence = torch.stack([self.gnn(graphs[i], graphs[i].ndata['w']) for i in range(len(graphs))], 1)
            gnn_out = self.gnn_lstm(gnn_sequence)
            outputs.append(gnn_out)
        if self.patterns:
            patterns = batch["patterns"]
            patterns_out = self.patterns_mlp(patterns)
            outputs.append(patterns_out)

        logits = self.classifier(torch.cat(outputs, axis=1))
        loss = F.binary_cross_entropy_with_logits(logits, targets.reshape_as(logits))
        return dict(loss=loss, logits=logits, targets=targets)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        result = self.forward(batch, "train")
        loss, logits, targets = result["loss"], result["logits"], result["targets"]
        metrics_vals = self.train_metrics(logits, targets.type(torch.int))
        tb_logger = self.loggers[0].experiment
        tb_logger.add_scalar("Loss/batch", loss, self.global_step)
        self.log(name="Acc",
                 value=metrics_vals["BinaryAccuracy"],
                 logger=False,
                 prog_bar=True)

        return loss

    def logging_metrics_end_epoch(self, outputs, stage="train"):
        if stage == "train":
            metrics = self.train_metrics
            csv_logger = self.loggers[1].experiment
        elif stage == "valid":
            metrics = self.valid_metrics
            csv_logger = self.loggers[2].experiment
        tb_logger = self.loggers[0].experiment
        loss_epoch = torch.stack(outputs).mean()
        epoch_metrics = metrics.compute()
        epoch_metrics.update({"Loss/epoch": loss_epoch})
        csv_logger.log_metrics(epoch_metrics, step=self.current_epoch)
        for metric, val in epoch_metrics.items():
            tb_logger.add_scalars(metric, {stage: val}, self.current_epoch)
        metrics.reset()
        return epoch_metrics

    def training_epoch_end(self, training_step_outputs):
        outputs = [out["loss"] for out in training_step_outputs]
        self.logging_metrics_end_epoch(outputs)

    def validation_step(self, batch, batch_idx):
        result = self.forward(batch, "val")
        loss, logits, targets = result["loss"], result["logits"], result["targets"]
        self.valid_metrics.update(logits, targets.type(torch.int))
        return loss

    def validation_epoch_end(self, outputs):
        epoch_metrics = self.logging_metrics_end_epoch(outputs, stage='valid')
        self.log("val_loss", epoch_metrics["Loss/epoch"], logger=False)

    # TODO: Think How to implement the testing of the model
    def predict_step(self, batch, batch_idx):
        result = self.forward(batch, "test")
        logits, targets = result["logits"], result["targets"]
        result_dict = {"batch_idx": batch_idx,
                       "logits": logits,
                       "targets": targets}
        return result_dict

    def configure_optimizers(self):
        if self.cfg.optimizer.mode == "adam":
            lr = self.cfg.optimizer.adam.lr
            beta1 = self.cfg.optimizer.adam.betas[0]
            beta2 = self.cfg.optimizer.adam.betas[1]
            weight_decay = self.cfg.optimizer.adam.weight_decay
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=lr,
                                         betas=(beta1, beta2),
                                         weight_decay=weight_decay)
        elif self.cfg.optimizer.mode == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.cfg.optimizer.sgd.lr,
                momentum=self.cfg.optimizer.sgd.momentum)
        return optimizer
