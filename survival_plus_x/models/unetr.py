import numpy as np
import pytorch_lightning as pl
import torch

from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock,\
    UnetrUpBlock, UnetrPrUpBlock
from monai.losses import DiceLoss

from survival_plus_x.models.vit import ViT3D


def proj_feat(x, hidden_size, feat_size):
    """
    Reconvert the Attention layer output to an 'image' format
    that can be processed by conv layers
    """

    new_view = (x.size(0), *feat_size, hidden_size)
    x = x.view(new_view)
    new_axes = (0, len(x.shape) - 1) + \
        tuple(d + 1 for d in range(len(feat_size)))
    x = x.permute(new_axes).contiguous()
    return x


class UNETR(pl.LightningModule):
    """
    A UNETR only for segmentation
    """

    def __init__(self,
                 learning_rate,
                 # ViT related
                 vit_image_size,
                 vit_patch_size,
                 vit_dim,
                 vit_depth,
                 vit_heads,
                 vit_dim_head,
                 vit_mlp_dim,
                 vit_channels,
                 vit_dropout=0.,
                 vit_emb_dropout=0.,
                 vit_output_token="cls",
                 # UNETR related
                 unetr_res_block=True,
                 unetr_conv_block=True,
                 unetr_norm_name="instance",
                 unetr_feature_size=16,  # n filters in the conv part of UNETR
                 unetr_attention_layer_output_idx=[3, 6, 9],
                 ):
        """
        Parameters
        ----------
        """

        super().__init__()
        self.save_hyperparameters()

        assert vit_output_token in ["cls", "mean"]
        assert vit_depth >= 4, f"UNETR requires at least 4 layers for the VIT, got {vit_depth}"
        assert len(
            unetr_attention_layer_output_idx) == 3, "need to provide 3 indices of which ViT features to use as intermediate output for UNETR"

        # make sure that all indices are INTERMEDIATE outputs, not including
        # the last layers output
        assert all(
            idx < vit_depth - 1 for idx in unetr_attention_layer_output_idx), f"Intermediate transformer layer indices have to be smaller than {vit_depth-1}, but got {unetr_attention_layer_output_idx}."

        # when doing the upsamling we also want to be able to support patch sizes
        # that differ by dimension (e.g. not only 16x16x16), however, that makes
        # behaviour of the models more complicated and we need different upsamling
        # mechanisms
        upsample_kernel_size = [None] * 3
        for i in range(3):
            # in order to obtain back the original shape after upsampling 4 times
            # we need to have upsample_kernel_size[i]^4 == patch_size[i]
            val = int(vit_patch_size[i]**(0.25))
            if val**4 != vit_patch_size[i]:
                raise ValueError(
                    f"Upsampling for UNETR is only possible if the fourth root of patch_size[{i}] is"
                    " an integer. Try patch size 1, 16, 81, 256, ...")
            upsample_kernel_size[i] = val

        self.vit = ViT3D(
            image_size=vit_image_size,
            patch_size=vit_patch_size,
            dim=vit_dim,
            depth=vit_depth,
            heads=vit_heads,
            mlp_dim=vit_mlp_dim,
            pool=None,  # output will always be (batch, n_patches + 1, dim)
            channels=vit_channels,
            dim_head=vit_dim_head,
            dropout=vit_dropout,
            emb_dropout=vit_emb_dropout)

        # UNETR helpers
        hidden_size = vit_dim
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=vit_channels,
            out_channels=unetr_feature_size,
            kernel_size=3,
            stride=1,
            norm_name=unetr_norm_name,
            res_block=unetr_res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=unetr_feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=unetr_norm_name,
            conv_block=unetr_conv_block,
            res_block=unetr_res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=unetr_feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=unetr_norm_name,
            conv_block=unetr_conv_block,
            res_block=unetr_res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=unetr_feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=unetr_norm_name,
            conv_block=unetr_conv_block,
            res_block=unetr_res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=unetr_feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=unetr_norm_name,
            res_block=unetr_res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=unetr_feature_size * 8,
            out_channels=unetr_feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=unetr_norm_name,
            res_block=unetr_res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=unetr_feature_size * 4,
            out_channels=unetr_feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=unetr_norm_name,
            res_block=unetr_res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=unetr_feature_size * 2,
            out_channels=unetr_feature_size,
            kernel_size=3,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=unetr_norm_name,
            res_block=unetr_res_block,
        )
        self.unetr_out = UnetOutBlock(
            spatial_dims=3,
            in_channels=unetr_feature_size,
            out_channels=1  # NOTE: we only want to segment tumor, no different parts of it
        )

        self.dice_loss = DiceLoss(
            sigmoid=True,  # we have to manually do the sigmoid beforehand in forward pass then!
            reduction="mean",
        )
        # expects raw unnormalized scores and combines sigmoid + BCELoss for better
        # numerical stability.
        # expects B x C x H x W x D
        self.ce_loss = torch.nn.BCEWithLogitsLoss(
            reduction="mean")

        self.unetr_output_activation = torch.nn.Sigmoid()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.1)

        return optimizer

    def forward(self, x_in):
        seg_logits, seg_mask = self._forward_without_mt_head(x_in)
        # feed the features from DenseNet and UNETR to the multitask head
        return {
            "seg_mask": seg_mask
        }

    def _forward_without_mt_head(self, x_in):
        vit_out, hidden_states_out = self.vit(x_in)
        # our vit always return the class tokens as well which monais vit
        # does not do, so we have to take care of removing that in order to
        # apply the UNETR code from them
        vit_out = vit_out[:, 1:]
        for k in range(len(hidden_states_out)):
            hidden_states_out[k] = hidden_states_out[k][:, 1:]

        # apply the UNETR codes to get the segmentation mask
        feat_size = self.vit.to_patch_embedding.num_patches_per_dim
        enc1 = self.encoder1(x_in)

        x2 = hidden_states_out[
            self.hparams.unetr_attention_layer_output_idx[0]]
        x2_proj = proj_feat(x2, self.hparams.vit_dim, feat_size)
        enc2 = self.encoder2(x2_proj)

        x3 = hidden_states_out[
            self.hparams.unetr_attention_layer_output_idx[1]]
        x3_proj = proj_feat(x3, self.hparams.vit_dim, feat_size)
        enc3 = self.encoder3(x3_proj)

        x4 = hidden_states_out[
            self.hparams.unetr_attention_layer_output_idx[2]]
        x4_proj = proj_feat(x4, self.hparams.vit_dim, feat_size)
        enc4 = self.encoder4(x4_proj)

        dec4 = proj_feat(vit_out, self.hparams.vit_dim, feat_size)

        dec3 = self.decoder5(dec4, enc4)

        dec2 = self.decoder4(dec3, enc3)

        dec1 = self.decoder3(dec2, enc2)

        out = self.decoder2(dec1, enc1)

        seg_logits = self.unetr_out(out)

        seg_mask = self.unetr_output_activation(seg_logits)

        return seg_logits, seg_mask

    def _predict_and_return_dict(self, batch, step_name):
        assert step_name in ["training_step",
                             "validation_step", "predict_step"]

        img = batch["img"]
        seg_logits, seg_mask = self._forward_without_mt_head(img)

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
        unetr_group = parent_parser.add_argument_group("UNETR")
        unetr_group.add_argument(
            "--unetr_norm_name",
            type=str,
            default="instance"
        )
        unetr_group.add_argument(
            "--unetr_feature_size",
            type=int,
            default=16
        )
        unetr_group.add_argument(
            "--unetr_attention_layer_output_idx",
            type=int,
            nargs=3,
            default=[3, 6, 9],
        )
        return parent_parser
