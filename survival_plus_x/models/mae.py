import torch
from torch import nn
from einops import repeat, rearrange

from survival_plus_x.models.vit import Transformer


def patch_to_img_3d(batch_of_patches, patch_shape, n_patches_per_dim):
    """
    Reconstruct a batch of 3D images from the patch representation.

    Params
    ------
    batch_of_patches: torch.Tensor of shape B N_patches Pixels_per_patch
    patch_shape: tuple of length 3
        contains the shape of each patch along all three dimensions (Z Y X)
    n_patches_per_dim: tuple of length 3
        contains the number of patches to restore for each dimension (Z Y X)
    """
    return rearrange(
        batch_of_patches,
        'b (z y x) (pz py px c) -> b c (z pz) (y py) (x px)',
        pz=patch_shape[0],
        py=patch_shape[1],
        px=patch_shape[2],
        z=n_patches_per_dim[0],
        y=n_patches_per_dim[1],
        x=n_patches_per_dim[2])


class MAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder_dim,
        decoder_depth=1,
        decoder_heads=8,
        decoder_dim_head=64
    ):
        super().__init__()

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.patch_to_emb = self.encoder.to_patch_embedding.patch_to_emb
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]

        # decoder parameters

        self.enc_to_dec = nn.Linear(
            encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_dim=decoder_dim * 4)

        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, patches, masked_indices, unmasked_indices):
        """
        patches: torch.Tensor of shape (B Num_Patches Pixels_per_patch)
        masked_indices: torch.Tensor
            shape (B num_masked)
        unmasked_indices: torch.Tensor
            shape (B Num_Patches - num_masked)
        """
        # patch to encoder tokens and add positions
        batch, num_patches = patches.shape[:2]
        num_masked = masked_indices.shape[1]
        # needed for proper slicing the patches, otherwise
        # we end up with an additional dimension after indexing
        batch_range = torch.arange(batch)[:, None]

        # get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]

        # get the unmasked tokens to be encoded
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        unmasked_tokens = tokens[batch_range, unmasked_indices]

        # attend the unmasked tokens with vision transformer
        encoded_unmasked_tokens = self.encoder.transformer(unmasked_tokens)[-1]

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_unmasked_tokens)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d',
                             b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder

        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim=1)
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[:, :num_masked]
        pred_masked_pixel_values = self.to_pixels(mask_tokens)

        # reconstruction of full image by ordering the patches
        # for the unmasked patches we can take their values right away

        reco_patches = torch.zeros_like(patches)
        reco_patches[batch_range, masked_indices] = pred_masked_pixel_values
        reco_patches[batch_range,
                     unmasked_indices] = patches[batch_range, unmasked_indices]
        reco_images = patch_to_img_3d(
            reco_patches,
            patch_shape=self.encoder.to_patch_embedding.patch_shape,
            n_patches_per_dim=self.encoder.to_patch_embedding.num_patches_per_dim)

        # predictions for masked patches, gt for masked patches, full reco image
        # where the first two elements are only required for computing the loss
        return pred_masked_pixel_values, masked_patches, reco_images
