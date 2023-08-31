import torch

from survival_plus_x.models.vit import vit_output_head
from survival_plus_x.models.cox_lightning import metrics_from_step_outputs
from survival_plus_x.losses.cindex_boosting import cindex_approx
from survival_plus_x.models.model_head import SurvivalModelHead


class CindexHead(SurvivalModelHead):
    def __init__(self,
                 input_shape,
                 output_activation="linear",
                 bias=False
                 ):
        self.bias = bias
        super().__init__(input_shape=input_shape)
        # self.save_hyperparameters()

        if output_activation == "linear":
            self.output_activation = torch.nn.Identity()
        elif output_activation == "tanh":
            self.output_activation = torch.nn.Tanh()
        else:
            raise ValueError(
                "output_activation can only be 'linear' or 'tanh', "
                f"not {output_activation}")

    def _create_network(self, input_shape):
        return vit_output_head(
            dim_input=input_shape,
            n_outputs=1,
            bias=self.bias)

    def _latent_to_final_output(self, latent):
        return self.output_activation(latent)

    def _compute_loss(self, batch_data):
        return cindex_approx(
            labels=batch_data["label"],
            risk=batch_data["prediction"],
            reduction_fn=torch.sum)

    def _compute_metrics(self, step_output_list):
        metrics, _ = metrics_from_step_outputs(step_output_list)
        return metrics
