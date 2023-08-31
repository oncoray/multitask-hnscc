import pytorch_lightning as pl
import torch

from survival_plus_x.models.cox_lightning import CoxHead
from survival_plus_x.models.cox_lightning import metrics_from_step_outputs as cox_metrics

from survival_plus_x.models.lognormal import LognormalHead
from survival_plus_x.models.lognormal import metrics_from_step_outputs as lognormal_metrics

from survival_plus_x.models.gensheimer import GensheimerHead
from survival_plus_x.models.gensheimer import metrics_from_step_outputs as gensheimer_metrics

from survival_plus_x.models.cindex import CindexHead
# from survival_plus_x.models.model_head import SurvivalModelHead


class MultitaskHead(pl.LightningModule):
    def __init__(self,
                 input_shape,
                 # for the lognormal and the gensheimer head
                 timepoints_cindex,
                 timepoints_brier,
                 training_labels,
                 # for the gensheimer
                 gensheimer_interval_breaks,
                 # for the cox head
                 cox_output_activation="tanh",
                 cox_nll_on_batch_only=True,
                 cox_average_over_events_only=False,
                 cox_training_labels=None,
                 cox_validation_labels=None,
                 cox_memory_bank_init_dict=None,
                 cox_memory_bank_decay_factor=1.,
                 cox_bias=False,
                 # choose which heads and which weights
                 heads_to_use=["cox", "lognormal", "gensheimer", "cindex"],
                 loss_weights={'cox': .25, 'lognormal': .25,
                               'gensheimer': .25, 'cindex': .25},
                 ):

        super().__init__()
        for h in heads_to_use:
            assert h in loss_weights

        # self.save_hyperparameters()

        self.heads = torch.nn.ModuleDict()
        self.weights = {}
        if "cox" in heads_to_use:
            self.heads["cox"] = CoxHead(
                input_shape=input_shape,
                output_activation=cox_output_activation,
                nll_on_batch_only=cox_nll_on_batch_only,
                average_over_events_only=cox_average_over_events_only,
                training_labels=cox_training_labels,
                validation_labels=cox_validation_labels,
                memory_bank_init_dict=cox_memory_bank_init_dict,
                memory_bank_decay_factor=cox_memory_bank_decay_factor,
                bias=cox_bias,
            )
            self.weights["cox"] = loss_weights['cox']

        if "lognormal" in heads_to_use:
            self.heads["lognormal"] = LognormalHead(
                input_shape=input_shape,
                timepoints_cindex=timepoints_cindex,
                timepoints_brier=timepoints_brier,
                training_labels=training_labels,
            )
            self.weights["lognormal"] = loss_weights['lognormal']

        if "gensheimer" in heads_to_use:
            self.heads["gensheimer"] = GensheimerHead(
                input_shape=input_shape,
                timepoints_cindex=timepoints_cindex,
                timepoints_brier=timepoints_brier,
                training_labels=training_labels,
                interval_breaks=gensheimer_interval_breaks,
            )
            self.weights["gensheimer"] = loss_weights["gensheimer"]

        if "cindex" in heads_to_use:
            self.heads["cindex"] = CindexHead(
                input_shape=input_shape,
                output_activation=cox_output_activation,
            )
            self.weights["cindex"] = loss_weights["cindex"]

    def configure_optimizers(self):
        # note: lr is then used for all params, but again
        # is not directly applicable since we will set it
        # on a module that uses this one
        return torch.optim.AdamW(self.parameters(), lr=1.e-4, weight_decay=0.)

    def forward(self, x):

        out_dict = {}
        for head_name, head in self.heads.items():
            out_dict[head_name] = head(x)

        return out_dict

    def _call_step_fn_on_heads(self, batch, batch_idx, step_name):
        assert step_name in ["training_step",
                             "validation_step", "predict_step"]

        batch_data = {}
        for head_name, head in self.heads.items():
            head_fn = getattr(head, step_name)
            batch_data[head_name] = head_fn(batch, batch_idx)

        return batch_data

    def _shared_step_trainphase(self, batch, batch_idx, train_or_val):
        step_name = {"train": "training_step",
                     "val": "validation_step"}[train_or_val]

        batch_data = self._call_step_fn_on_heads(
            batch, batch_idx,
            step_name=step_name)
        losses = []
        weights = []
        for head_name in batch_data:
            losses.append(batch_data[head_name]["loss"])
            weights.append(self.weights[head_name])
        weights = torch.tensor(weights).to(self.device)

        loss = (torch.stack(losses) * weights).sum()

        batch_data["loss"] = loss

        # detach everything properly (except the loss we need for training)
        for k1 in batch_data:
            v1 = batch_data[k1]
            if not isinstance(v1, dict):
                # this skips the 'loss' key
                continue
            for k2 in v1:
                v = v1[k2]
                if isinstance(v, torch.Tensor):
                    if v.requires_grad:
                        # print(f"{k1}: {k2} should be detached!")
                        batch_data[k1][k2] = batch_data[k1][k2].detach()

        return batch_data

    def training_step(self, batch, batch_idx):
        batch_data = self._shared_step_trainphase(batch, batch_idx, "train")

        return batch_data

    def validation_step(self, batch, batch_idx):
        batch_data = self._shared_step_trainphase(batch, batch_idx, "val")

        return batch_data

    def predict_step(self, batch, batch_idx):
        return self._call_step_fn_on_heads(
            batch, batch_idx, step_name="predict_step")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = CoxHead.add_model_specific_args(parent_parser)

        parent_parser.add_argument(
            "--heads_to_use",
            nargs="+",
            choices=["cox", "lognormal", "cindex", "gensheimer"],
            default=["cox", "lognormal", "cindex", "gensheimer"]
        )
        parent_parser.add_argument(
            "--gensheimer_interval_breaks",
            type=float,
            nargs="+",
            required=True,
            help="Definition of interval borders for gensheimer loss. Needs to start with 0.!"
        )
        parent_parser.add_argument(
            "--timepoints_cindex",
            type=float,
            nargs="+",
            required=True,
            help="Time points at which C-index metric is computed for gensheimer and lognormal model."
        )
        parent_parser.add_argument(
            "--timepoints_brier",
            type=float,
            nargs="+",
            required=True,
            help="Time points over which integrated_brier_score metric is computed for gensheimer and lognormal model."
        )

        return parent_parser


def multitask_metrics_from_step_outputs(
        step_output_list,
        task_names,
        timepoints_cindex,
        timepoints_brier,
        training_labels,
        gensheimer_interval_breaks):

    metrics = {}
    predictions = {}
    if "cox" in task_names:
        metrics["cox"], predictions["cox"] = cox_metrics(
            [d["cox"] for d in step_output_list])

    if "lognormal" in task_names:
        metrics["lognormal"], predictions["lognormal"] = lognormal_metrics(
            [d["lognormal"] for d in step_output_list],
            timepoints_cindex=timepoints_cindex,
            timepoints_brier=timepoints_brier,
            training_labels=training_labels)
    if "gensheimer" in task_names:
        metrics["gensheimer"], predictions["gensheimer"] = gensheimer_metrics(
            [d["gensheimer"] for d in step_output_list],
            timepoints_cindex=timepoints_cindex,
            timepoints_brier=timepoints_brier,
            training_labels=training_labels,
            interval_breaks=gensheimer_interval_breaks)
    if "cindex" in task_names:
        # We can evaluate the cindex model in the same way as the cox model, no need
        # for redefining the function
        metrics["cindex"], predictions["cindex"] = cox_metrics(
            [d["cindex"] for d in step_output_list])

    return metrics, predictions
