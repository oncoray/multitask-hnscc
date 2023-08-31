import pytorch_lightning as pl
import torch


class SurvivalModelHead(pl.LightningModule):
    def __init__(self,
                 input_shape):

        super().__init__()
        self.mlp_head = self._create_network(input_shape)

    def _create_network(self, input_shape):
        raise NotImplementedError

    def configure_optimizers(self):
        # NOTE: we just set a dummy learning rate here
        # since this function is required by lightning
        # but we will not train it as a lightning module
        # but only use it as building block in some other
        # module, which then is responsible for setting
        # the learning rate directly.
        return torch.optim.AdamW(
            self.parameters(), lr=1.e-4)

    def _latent_to_final_output(self, latent):
        """
        Convert the latent output of the network to the
        final form needed for predictions
        """
        raise NotImplementedError

    def forward(self, x):
        latent = self.mlp_head(x)
        return self._latent_to_final_output(latent)

    def _predict_and_return_dict(self, batch):
        assert "img_features" in batch
        assert "label" in batch
        assert "patient" in batch

        # tensor with shape B input_shape
        batch_img_feats = batch["img_features"]
        batch_labels = batch["label"]  # tensor with
        batch_patients = batch["patient"]  # list of str

        batch_predictions = self.forward(batch_img_feats)

        return {
            "patient": batch_patients,
            "label": batch_labels,
            "prediction": batch_predictions
        }

    def _compute_loss(self, batch_data):
        raise NotImplementedError

    def _shared_step_trainphase(self, batch, batch_idx):
        batch_data = self._predict_and_return_dict(batch)

        loss = self._compute_loss(batch_data)
        batch_data["loss"] = loss

        # NOTE: logging might not be needed at all since we will not
        # directly train such a Head, but will use it as a building
        # block in other lightning models, for which we will then log
        # instead.

        # detach everything properly (except the loss we need for training)
        for k in batch_data:
            v = batch_data[k]
            if k == "loss":
                continue
            if isinstance(v, torch.Tensor):
                if v.requires_grad:
                    batch_data[k] = v.detach()

        return batch_data

    def training_step(self, batch, batch_idx):
        batch_data = self._shared_step_trainphase(batch, batch_idx)

        return batch_data

    def validation_step(self, batch, batch_idx):
        batch_data = self._shared_step_trainphase(batch, batch_idx)

        return batch_data

    def predict_step(self, batch, batch_idx):
        return self._predict_and_return_dict(batch)

    def _compute_metrics(self, step_output_list):
        raise NotImplementedError
