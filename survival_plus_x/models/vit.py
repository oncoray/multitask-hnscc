"""
Taken and adapted from https://github.com/lucidrains/vit-pytorch.
Also some inspiration for the embeddings taken from https://github.com/rwightman/pytorch-image-models

"""

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import numpy as np
# helpers


def pair(t):
    if isinstance(t, (tuple, list)):
        assert len(t) == 2
        return t
    else:
        return (t,) * 2


def triple(t):
    if isinstance(t, (tuple, list)):
        assert len(t) == 3
        return t
    else:
        return (t,) * 3

# classes


class PreNorm(nn.Module):
    """
    Applies LayerNormalisation before a given function

    Parameters
    ----------
    dim: int
        Dimensionality of the input features
    fn: callable (nn.Module)
        The function to apply to the input after
        doing layer normalisation
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    A simple MLP with two hidden layers and GELU nonlinearity
    that preserves input dimensionality.
    Dropout can be applied between layers and at output.
    Output activation is linear.

    Parameters
    ----------
    dim: int
        Dimensionality of the input features to this MLP
        and the output feature dimension
    hidden_dim: int
        Dimensionality of the intermediate feature layer
    dropout: float (between 0 and 1)
        Dropout rate between layers and at the output

    Returns
    -------
    A tensor of the same shape as the input.
    """

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    Self-attention layer (preserves dimensionality)

    Parameters
    ----------
    dim: int
        Dimensionality of the input features (and also the output features).
    heads: int
        Number of attention heads
    dim_head: int
        Dimensionality of the features that each head
        processes.
        The product dim_head * heads determines the dimensionality
        of the query, key and value obtained from the input feature.
    dropout: float (between 0 and 1)
        Dropout rate applied at the output.
        Not active when heads=1 and dim_head = dim

    Returns
    -------
    A tensor of the same shape as the input (batch_size, num_patches, patch_dim)
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # if the intermediate feature representation is not
        # equal to the input feature dimension, we need another
        # linear layer to map it back to the input dimensionality.
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """
        x: torch.Tensor
            Shape (batch_size, num_patches, feature_dimension)
        """
        # map features to internally used dimensionality
        # and make 3 chunks out of it by splitting the
        # feature dimension, each with dimension (batch_size, num_patches, inner_dim)
        # where inner_dim = num_heads * dim_head
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # now reshape so the dimension for the heads comes right after the batch
        # i.e. (batch, num_heads, num_patches, dim_head)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # attention matrix computation
        # q has shape (batch_size, num_heads, num_patches, dim_head)
        # and the transposed k has shape
        # (batch_size, num_heads, dim_head, num_patches)
        # and the matrix multiplication is broadcast to obtain output
        # of shape
        # (batch_size, num_heads, num_patches, num_patches).
        # So for each head, we get a square matrix full of attention values
        # for all pairs of patches
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # Apply softmax: now the sum over the last dimension is one
        # for all entries of the other three dimensions
        attn = self.attend(dots)

        # apply attention to the values
        # attn has shape (batch_size, n_heads, num_patches, num_patches)
        # and v has shape
        # (batch_size, n_heads, num_patches, dim_head)
        # and matrix multiplication is broadcast to obtain
        # shape
        # (batch_size, n_heads, num_patches, dim_head).
        # Since for attn the sum across each row (considering only last two dims)
        # is 1, this effectively
        # puts a weight on every element of the patch dimension.
        # This basically averages (not really, due to different attention weights)
        # each feature over all the patches
        out = torch.matmul(attn, v)
        # now fuse separate head results back together
        # to obtain shape (batch_size, num_patches, n_heads*head_dim=inner_dim)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # map back to original dimension
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Parameters
    ----------
    dim: int
        Dimensionality of input features
    depth: int
        Number of blocks to use.
        Each block consists of Prenorm + Attention followed by
        Prenorm + MLP
    heads: int
        Number of heads in each attention layer
    dim_head: int
        Feature dimensionality of each attention head
    mlp_dim: int
        Feature dimensionality of the MLP that follows attention layer
    dropout: float (between 0 and 1)
        Dropout rate in attention layers and MLP layers

    Returns
    -------
    A torch tensor of shape (batch_size, num_patches_in, dim_in), i.e.
    of same shape as the input.
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.,
                 return_intermediate_outputs=False):
        super().__init__()
        self.return_intermediate_outputs = return_intermediate_outputs

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads,
                        dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        """
        This applies a resnet style computation by adding the identity as well
        """
        intermediate_outputs = []
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            intermediate_outputs.append(x)

        if self.return_intermediate_outputs:
            return intermediate_outputs
        else:
            return x


def vit_output_head(dim_input, n_outputs, bias):
    return nn.Sequential(
        nn.LayerNorm(dim_input),
        nn.Linear(dim_input, n_outputs, bias=bias))


class ViTBase(nn.Module):
    """
    The Vision transformer

    Parameters
    ----------
    image_size: int or tuple of ints (length two)
        Specify height and width of images
    patch_size: int or tuple of ints (length two)
        Specify the height and width of each image patch
        that will be created and for which features are computed.
    dim: int
        Dimensionality of the feature representation for each patch.
    depth: int
        Number of transformer blocks to use
    heads: int
        Number of attention heads within the Attention layers of each
        transformer block
    mlp_dim: int
        Dimensionality of the MLP within each Transformer block.
    pool: str, either 'cls' or 'mean'
        Determines the way the output of the transformer is used.
        If 'mean', the second dimension (after batch) is averaged,
        i.e. over all patches and the class token.
        Otherwise, the first entry of the second dimension is used
        (which is for the class token)

    channels: int
        Number of input channels of the images
    dim_head: int
        Dimensionality of features that are processed by each attention
        head.
    dropout: float (between 0 and 1)
        Dropout rate in the Transformer blocks
    emb_dropout: float (between 0 and 1)
        Dropout rate after the patch embedding + positional encoding

    Returns
    -------
    a Tensor with shape (batch_size, dim), and a list of all other attenion layer outputs of size (batch_size, n_patches, dim)
    """

    def __init__(self, *,
                 patch_embedding_cls,
                 image_size,
                 patch_size,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 pool='cls',
                 channels=3,
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.):

        super().__init__()
        assert pool in {
            'cls', 'mean', None}, 'pool type must be either cls (cls token), mean (mean pooling) or None'

        # store hyperparameter as member variables
        # to save them as hyperparameters in a lightning
        # module
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.pool = pool
        self.channels = channels
        self.dim_head = dim_head
        self.dr = dropout
        self.emb_dr = emb_dropout

        self.to_patch_embedding = patch_embedding_cls(
            image_size, patch_size, channels, dim)

        num_patches = self.to_patch_embedding.num_patches

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # the cls_token is the feature representation of just one
        # more artificial patch that gets added
        # to have a reference on which attention scores we want to look at
        # (that can view all other patches)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout,
            return_intermediate_outputs=True)

    def forward(self, img):
        # (now batch_size, n_patches, patch_dim)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # repeat the class token for all samples in the batch
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # append the feature of the class token to all patch features
        x = torch.cat((cls_tokens, x), dim=1)
        # add the positional embedding information to the feature for each patch
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # input to the transformer: (batch_size, num_patches+1, feature_dim)
        # and output is of same shape
        transformer_outputs = self.transformer(x)

        # output of only the last layer
        x = transformer_outputs[-1]
        # collapse the patch dimension by either taking the feature
        # representation of the class token (self.pool == 'cls')
        # or the average feature representation of all tokens.
        # Result is of shape (batch_size, feature_dim)
        if self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "cls":
            x = x[:, 0]
        elif self.pool is None:
            # NOTE: we keep the x as is
            pass

        # NOTE: the transformer_outputs still always carry the class token
        return x, transformer_outputs


class PatchEmbeddingBase(nn.Module):
    def __init__(self, image_size, patch_size, channels, dim):
        super().__init__()

        image_shape = np.array(self._process_size(image_size))
        patch_shape = np.array(self._process_size(patch_size))

        if not np.all(image_shape % patch_shape == 0):
            raise ValueError(
                'Image dimensions must be divisible by the patch size.')
        # number of patches along each dimension
        self.num_patches_per_dim = tuple(image_shape // patch_shape)
        # total number of patches
        self.num_patches = np.prod(self.num_patches_per_dim)

        self.image_shape = tuple(image_shape)
        self.patch_shape = tuple(patch_shape)

        # dimensionality of each patch
        # i.e. total number of pixels including channel
        patch_dim = channels * np.prod(patch_shape)
        self.patch_dim = patch_dim

        self.to_patch = self._create_patch_layer(patch_shape.tolist())
        self.patch_to_emb = self._create_embedding_layer(patch_dim, dim)

    def _process_size(self, img_size):
        raise NotImplementedError

    def _create_patch_layer(self, patch_shape):
        raise NotImplementedError

    def _create_embedding_layer(self, patch_dim, dim):
        return nn.Linear(patch_dim, dim)

    def forward(self, img):

        x = self.to_patch(img)
        return self.patch_to_emb(x)


class PatchEmbedding2D(PatchEmbeddingBase):
    def _process_size(self, img_size):
        return pair(img_size)

    def _create_patch_layer(self, patch_shape):
        return Rearrange(
            'b c (y py) (x px) -> b (y x) (py px c)',
            py=patch_shape[0], px=patch_shape[1])


class PatchEmbedding3D(PatchEmbeddingBase):
    def _process_size(self, img_size):
        return triple(img_size)

    def _create_patch_layer(self, patch_shape):
        print("patch_shape", patch_shape)
        return Rearrange(
            'b c (z pz) (y py) (x px) -> b (z y x) (pz py px c)',
            pz=patch_shape[0],
            py=patch_shape[1],
            px=patch_shape[2])

# TODO: we can also do the patch embedding with a convolution as is done
# in the timm library


class PatchEmbedding3DWithConv(PatchEmbedding3D):
    def _create_patch_layer(self, patch_shape):
        # not needed since the convolution does both
        # in one go
        return torch.nn.Identity()

    def _create_embedding_layer(self, patch_dim, dim):
        in_channels = int(self.patch_dim / np.prod(self.patch_shape))

        # print("patch embedd with conv layer")
        # print("in_channels=", in_channels)
        # print("out_channels=", dim)
        # print("kernel_size/stride=", self.patch_shape)

        return torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=self.patch_shape,
            stride=self.patch_shape
        )

    def forward(self, img):
        # is of shape batch_size, patch_dim, (n_z, n_y, n_x)
        x_with_chan = super().forward(img)
        # print("Patch embed output shape is", x_with_chan.shape)
        # should be (batch_size, n_patches, patch_dim)

        op = Rearrange('b c z y x -> b (z y x) c')
        x = op(x_with_chan)
        return x


class ViT(ViTBase):
    def __init__(self, *,
                 image_size,
                 patch_size,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 pool='cls',
                 channels=3,
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.):

        super().__init__(
            patch_embedding_cls=PatchEmbedding2D,
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool=pool,
            channels=channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout)


class ViT3D(ViTBase):
    def __init__(self, *,
                 image_size,
                 patch_size,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 pool='cls',
                 channels=3,
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.,
                 patch_embedding_cls=PatchEmbedding3D):

        assert patch_embedding_cls in [
            PatchEmbedding3D, PatchEmbedding3DWithConv]

        super().__init__(
            patch_embedding_cls=patch_embedding_cls,
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool=pool,
            channels=channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout)
