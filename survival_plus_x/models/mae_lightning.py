import pytorch_lightning as pl
import torch

from survival_plus_x.models.mae import MAE


class MAELightning(pl.LightningModule):
    def __init__(self,
                 vit,
                 learning_rate,
                 masking_ratio,
                 decoder_dim,
                 decoder_depth,
                 decoder_heads,
                 decoder_dim_head,
                 loss
                 ):
        super().__init__()
        if not (masking_ratio > 0 and masking_ratio < 1):
            raise ValueError('masking ratio must be kept between 0 and 1')

        self.save_hyperparameters()

        self.vit = vit
        # this can be discarded after
        # training most likely
        self.mae = MAE(
            encoder=self.vit,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            decoder_heads=decoder_heads,
            decoder_dim_head=decoder_dim_head
        )

        if loss == "mae":
            self.loss_fn = torch.nn.L1Loss()
        elif loss == "mse":
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss {loss}")

    def _shared_step(self, batch, batch_idx):
        imgs = batch["img"]

        # convert images to patches
        patches = self.vit.to_patch_embedding.to_patch(imgs)
        batch_size, num_patches, *_ = patches.shape

        # randomly choose indices for masking the patches
        num_masked = int(self.hparams.masking_ratio * num_patches)
        rand_indices = torch.rand(
            batch_size, num_patches).argsort(dim=-1).to(self.device)
        masked_indices = rand_indices[:, :num_masked]
        unmasked_indices = rand_indices[:, num_masked:]

        # compute reconstructions via the MAE
        pred_masked, gt_masked, reco = self.mae(
            patches, masked_indices, unmasked_indices)

        loss = self.loss_fn(pred_masked, gt_masked)

        return {
            "loss": loss,
            "patient": batch["patient"],
            "prediction_masked": pred_masked.detach(),
            "ground_truth_masked": gt_masked.detach(),
            "reco": reco.detach()
        }

    def training_step(self, batch, batch_idx):
        """
        During training, the incoming image patches
        should be masked randomly.
        """

        ret_dict = self._shared_step(batch, batch_idx)

        self.log("train_loss", ret_dict["loss"],
                 on_step=False,
                 logger=True,
                 prog_bar=True,
                 on_epoch=True,
                 sync_dist=True)

        return ret_dict

    def validation_step(self, batch, batch_idx):
        ret_dict = self._shared_step(batch, batch_idx)

        self.log("val_loss", ret_dict["loss"],
                 on_step=False,
                 logger=True,
                 prog_bar=True,
                 on_epoch=True,
                 sync_dist=True,
                 )

        self.log("hp_metric", ret_dict["loss"])

        return ret_dict

    def forward(self, batch):
        # this is inference mode
        """
        This is the inference logic where we would
        expect the MAE to accept images and masked patch
        indices and reconstruct the full image
        """
        imgs = batch["img"]
        # we expect to be passed the unmasked indices
        # during inference explicitely
        masked_indices = batch["masked_indices"]
        unmasked_indices = batch["unmasked_indices"]

        # convert images to patches
        patches = self.vit.to_patch_embedding.to_patch(imgs)

        # compute reconstructions via the MAE
        pred_masked, gt_masked, reco = self.mae(
            patches, masked_indices, unmasked_indices)

        return {
            "patient": batch["patient"],
            "prediction_masked": pred_masked.detach(),
            "ground_truth_masked": gt_masked.detach(),
            "reco": reco.detach()
        }

    def predict_step(self, batch, batch_idx):
        return self.forward(batch)

    def configure_optimizers(self):
        print("configure optimizer with learning rate",
              self.hparams.learning_rate)

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1.2e-5)

        print(optimizer.defaults["lr"])

        # TODO: also learning rate scheduler
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):

        mae_group = parent_parser.add_argument_group("MAE model")
        mae_group.add_argument(
            "--masking_ratio",
            type=float,
            default=0.75
        )
        mae_group.add_argument(
            "--decoder_dim",
            type=int,
            default=512
        )
        mae_group.add_argument(
            "--decoder_depth",
            type=int,
            default=6
        )
        mae_group.add_argument(
            "--decoder_heads",
            type=int,
            default=6
        )
        mae_group.add_argument(
            "--decoder_dim_head",
            type=int,
            default=64
        )
        mae_group.add_argument(
            "--autoencoder_loss",
            type=str,
            choices=["mae", "mse"],
            default="mse"
        )

        return parent_parser
