import tensorflow as tf
from tensorflow.keras.layers import (Conv3D, Conv3DTranspose, MaxPooling3D, UpSampling3D,
                                     BatchNormalization, LeakyReLU, Add, Dropout,
                                     LayerNormalization, MultiHeadAttention, Dense, Input, concatenate)
from tensorflow.keras.models import Model

# ------------------------------------------------------------------------- #
#                        1) Basic 3D Residual Block                         #
# ------------------------------------------------------------------------- #

def residual_block(x, filters, kernel_size=(3, 3, 3)):
    """
    A 3D residual block with BN + LeakyReLU.
    """
    shortcut = x

    # Convolution 1
    x = Conv3D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Convolution 2
    x = Conv3D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    # Match channels in shortcut if needed
    if shortcut.shape[-1] != filters:
        shortcut = Conv3D(filters, (1, 1, 1), padding='same')(shortcut)

    # Residual Add
    x = Add()([shortcut, x])
    x = LeakyReLU()(x)
    return x

# ------------------------------------------------------------------------- #
#            2) Patch Embedding / Unembedding for 3D volumes               #
# ------------------------------------------------------------------------- #

class PatchEmbed3D(tf.keras.layers.Layer):
    """
    3D Patch Embedding:
      - Splits the (D,H,W) volume into 3D patches of size patch_size.
      - Maps each patch to an embed_dim feature vector via a Conv3D.
    """
    def __init__(self, patch_size=(2,2,2), embed_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = Conv3D(filters=embed_dim,
                           kernel_size=patch_size,
                           strides=patch_size,
                           padding='valid')

    def call(self, x):
        x = self.proj(x)
        b, d, h, w, c = tf.unstack(tf.shape(x))
        x = tf.reshape(x, (b, d*h*w, c))
        return x, d, h, w

class PatchUnembed3D(tf.keras.layers.Layer):
    """
    3D Patch Unembedding:
    """
    def __init__(self, embed_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def call(self, x, d, h, w):
        b = tf.shape(x)[0]
        x = tf.reshape(x, (b, d, h, w, self.embed_dim))
        return x

# ------------------------------------------------------------------------- #
#       3) A Minimal Transformer Block for Patch Embeddings (3D)           #
# ------------------------------------------------------------------------- #

def transformer_block_3d(x,
                         patch_size=(2,2,2),
                         embed_dim=32,
                         num_heads=2,
                         ff_dim=64):
    """
    A minimal "ViT-like" block for 3D volumes.
    """
    pe = PatchEmbed3D(patch_size=patch_size, embed_dim=embed_dim)
    x_seq, d, h, w = pe(x)

    x_norm = LayerNormalization()(x_seq)
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x_norm, x_norm)
    x_seq = x_seq + attn_output

    x_norm = LayerNormalization()(x_seq)
    ffn = Dense(ff_dim, activation='relu')(x_norm)
    ffn = Dense(embed_dim)(ffn)
    x_seq = x_seq + ffn

    pu = PatchUnembed3D(embed_dim=embed_dim)
    x_out = pu(x_seq, d, h, w)

    return x_out

# ------------------------------------------------------------------------- #
#    4) Putting It All Together in a Res-UNet with Patch Transformer        #
# ------------------------------------------------------------------------- #

def my_res_unet_transformer_patch_model(
    dim_in,
    Ncl,
    filters=[32, 48, 64],
    dropout_rate=0.1,
    patch_size=(1,1,1),      # patch size for the patch embedding
    embed_dim=32,            # dimension to embed each patch
    num_heads=2,
    ff_dim=64,
    num_transformer_blocks=1
):
    """
    A Res-UNet that includes patch-based Transformer blocks in the bottleneck.
    """
    inputs = Input(shape=(dim_in, dim_in, dim_in, 1))
    x = inputs
    down_layers = []

    # ----------------- Encoder ----------------- #
    for f in filters[:-1]:
        x = residual_block(x, f)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        down_layers.append(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)

    # ----------------- Bottleneck (Res Blocks) ----------------- #
    for _ in range(2):
        x = residual_block(x, filters[-1])

    # ----------------- Bottleneck (Transformer Blocks) ----------------- #
    for _ in range(num_transformer_blocks):
        x = transformer_block_3d(
            x,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim
        )
        if embed_dim != filters[-1]:
            x = Conv3D(filters[-1], kernel_size=1, padding='same')(x)

    # ----------------- Decoder ----------------- #
    for f, skip in zip(reversed(filters[:-1]), reversed(down_layers)):
        x = UpSampling3D(size=(2,2,2))(x)
        x = concatenate([x, skip])
        x = residual_block(x, f)
        x = residual_block(x, f)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

    # ----------------- Output ----------------- #
    outputs = Conv3D(Ncl, (1, 1, 1), padding='same', activation='softmax')(x)

    # Build Model
    model = Model(inputs, outputs)
    return model

