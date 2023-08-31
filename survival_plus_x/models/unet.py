import pytorch_lightning as pl
import torch

from monai.losses import DiceLoss

from survival_plus_x.models.survival_plus_unet import VariableDepthUNet


class UNET(pl.LightningModule):
    """
    Similar implementation as for Deep-MTS (https://arxiv.org/pdf/2109.07711v1.pdf),
    using a Unet the segmentation model and a DenseNet as the survival net.
    """

    def __init__(self,
                 learning_rate,
                 # UNET related
                 unet_in_channels,
                 unet_features_start=8,
                 ):
        """
        Parameters
        ----------
        """
        super().__init__()
        self.save_hyperparameters()

        unet_features = [unet_features_start] + [unet_features_start * 2 **
                                                 i for i in range(4)] + [unet_features_start]

        self.unet = VariableDepthUNet(
            spatial_dims=3,
            in_channels=unet_in_channels,
            out_channels=1,
            features=unet_features)

        self.dice_loss = DiceLoss(
            sigmoid=True,  # we have to manually do the sigmoid beforehand in forward pass then!
            reduction="mean",
        )
        # expects raw unnormalized scores and combines sigmoid + BCELoss for better
        # numerical stability.
        # expects B x C x H x W x D
        self.ce_loss = torch.nn.BCEWithLogitsLoss(
            reduction="mean")

        self.unet_output_activation = torch.nn.Sigmoid()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.1)

        return optimizer

    def forward(self, x_in):
        seg_logits, seg_mask, unet_feats = self._forward_without_mt_head(x_in)
        # feed the features from DenseNet and UNETR to the multitask head

        return {
            "seg_mask": seg_mask
        }

    def _forward_without_mt_head(self, x_in):
        seg_logits, unet_feats = self.unet(x_in)
        seg_mask = self.unet_output_activation(seg_logits)

        return seg_logits, seg_mask, unet_feats

    def _predict_and_return_dict(self, batch, step_name):
        assert step_name in ["training_step",
                             "validation_step", "predict_step"]

        img = batch["img"]
        seg_logits, seg_mask, unet_feats = self._forward_without_mt_head(img)

        # feed the features from DenseNet and UNETR to the multitask head
        # and compute survival losses and aggregate them
        return {
            "img": img.detach().cpu().numpy(),
            "mask": batch["mask"].detach().cpu().numpy(),
            "prediction": seg_mask,
            "prediction_logits": seg_logits
        }

    def _shared_step_trainphase(self, batch, batch_idx, train_or_val):
        step_name = {"train": "training_step",
                     "val": "validation_step"}[train_or_val]

        logging_args = dict(
            on_step=False, logger=True, prog_bar=True,
            on_epoch=True, sync_dist=False,
            batch_size=len(batch["patient"]))

        out_dict = self._predict_and_return_dict(batch, step_name)

        # compute segmentation loss
        mask = torch.as_tensor(batch["mask"], dtype=torch.float32)
        # Dont do the sigmoid since it will be done by the losses!
        # print("img:", img.shape, img.dtype)
        # print("mask:", mask.shape, mask.dtype, mask.min(), mask.max())
        # print("pred_mask:", pred_mask.shape, pred_mask.dtype, pred_mask.min(), pred_mask.max())
        seg_logits = out_dict["prediction_logits"]
        ce_loss = self.ce_loss(seg_logits, mask)
        dice_loss = self.dice_loss(seg_logits, mask)
        seg_loss = 0.5 * (ce_loss + dice_loss)

        loss = seg_loss

        out_dict["loss"] = loss
        out_dict["ce_loss"] = ce_loss.detach()
        out_dict["dice_loss"] = dice_loss.detach()
        out_dict["prediction"] = (
            out_dict["prediction"].detach())
        out_dict["prediction_logits"] = (
            out_dict["prediction_logits"].detach())

        # logging
        # a) total loss
        self.log(f"{train_or_val}_loss", loss, **logging_args)
        # e) single segmentation losses
        self.log(f"segmentation/ce_{train_or_val}_loss",
                 ce_loss, **logging_args)
        self.log(
            f"segmentation/dice_{train_or_val}_loss", dice_loss,
            **logging_args)

        return out_dict

    def training_step(self, batch, batch_idx):
        return self._shared_step_trainphase(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step_trainphase(batch, batch_idx, "val")

    def predict_step(self, batch, batch_idx):
        return self._predict_and_return_dict(batch, "predict_step")

    def _shared_epoch_end(self, step_output_list, train_or_val):
        # log segmentation metrics is not necessary since the dice loss
        # is the negative of the dice score and hence we already have a metric
        pass

    def training_epoch_end(self, step_output_list):
        self._shared_epoch_end(step_output_list, "train")

    def validation_epoch_end(self, step_output_list):
        self._shared_epoch_end(step_output_list, "val")

    @ staticmethod
    def add_model_specific_args(parent_parser):
        unet_group = parent_parser.add_argument_group("UNET")
        unet_group.add_argument(
            "--unet_features_start",
            type=int,
            default=8,
            help="Number of feature maps after first downsampling block of Unet. "
                 "Doubled in each further downsampling block."
        )
        return parent_parser
