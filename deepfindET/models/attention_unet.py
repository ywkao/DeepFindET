from tensorflow.keras.layers import (Conv3D, MaxPooling3D, UpSampling3D,
                                     BatchNormalization, concatenate, Input,
                                     Activation, Multiply, Add, Dropout)
from tensorflow.keras.models import Model
import tensorflow as tf

def conv_block(x, filters):
    x = Conv3D(filters, (3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(filters, (3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def attention_block(skip, gating, inter_channels):
    """
    skip:       the higher-resolution (encoder) feature map
    gating:     the coarser, lower-resolution (decoder) feature (already upsampled or not)
    inter_channels: number of intermediate channels to project both skip & gating to

    Classic Attention U-Net approach:
    1) Downsample skip (spatial) to match gating, project both to inter_channels
    2) Add them -> ReLU
    3) 1x1 conv -> sigmoid => single-channel attention map
    4) Upsample that map back to skip's original spatial size
    5) Multiply with the original skip
    """

    # 1) Match gating channels via 1x1
    phi_g = Conv3D(inter_channels, kernel_size=1, padding='same')(gating)

    # 2) Downsample skip if needed to match gating's spatial shape
    #    e.g., kernel_size=2, strides=2 reduces the skip's height/width/depth
    theta_x = Conv3D(inter_channels, kernel_size=1, strides=1, padding='same')(skip)
    print("Shape of phi_g: " , phi_g.shape)
    print("Shape of theta_x: ", theta_x.shape)

    # Add + ReLU
    concat_xg = Add()([theta_x, phi_g])
    concat_xg = Activation('relu')(concat_xg)
    print("shape of concat_xg: ", concat_xg.shape)

    # 3) Single-channel attention coefficients
    psi = Conv3D(1, kernel_size=1, padding='same')(concat_xg)
    psi = Activation('sigmoid')(psi)
    print("Shape of psi: ", psi.shape)

    # 4) Upsample the attention coefficients back to skip's size
    #    so it has the same spatial shape as the original skip
    #psi = UpSampling3D(size=(2, 2, 2))(psi)
    print("Shape of psi after upsampling: " , psi.shape)

    # 5) Multiply to gate the skip connection
    out = Multiply()([skip, psi])

    return out

def attention_unet(dim_in, Ncl, filters=[32, 48, 64], dropout_rate=0.2):
    input_tensor = Input(shape=(dim_in, dim_in, dim_in, 1))

    # -------------------------
    # Encoder
    # -------------------------
    x = input_tensor
    down_layers = []
    for f in filters:
        x = conv_block(x, f)
        down_layers.append(x)
        x = MaxPooling3D((2, 2, 2))(x)
        x = Dropout(dropout_rate)(x)

    # -------------------------
    # Bottleneck
    # -------------------------
    x = conv_block(conv_block(x, filters[-1]), filters[-1])

    # -------------------------
    # Decoder
    # -------------------------
    # We'll pair the reversed filters with reversed skip connections.
    # For each skip 's', we have a gating signal 'x' that is upsampled
    # to match s's spatial size. Then we call attention_block(s, x, f).
    #
    # Classic approach: the gating has smaller feature resolution, so
    # we skip is downsampled in the attention block to match gating
    # (or gating is not fully upsampled yet). Then we upsample the
    # final mask back to skip's resolution.

    for f, skip in zip(reversed(filters), reversed(down_layers)):
        # Up-sample the gating signal from the lower resolution
        x = UpSampling3D((2, 2, 2))(x)
        x = Dropout(dropout_rate)(x)  # optional extra dropout
        print("Shape of x: " , x.shape)
        # Attention gating
        # skip has shape e.g. (B, H, W, D, f) from the encoder
        # x has shape   e.g. (B, H, W, D, ???) from the decoder
        print("Shape of skip: ", skip.shape)
        att_skip = attention_block(skip, x, inter_channels=f)
        print("Shape of att_skip:" , att_skip.shape)
        # Combine the gated skip + the decoder feature
        x = concatenate([att_skip, x])
        x = conv_block(x, f)  # reduce to the desired filter dimension again
        x = Dropout(dropout_rate)(x)

    # Final segmentation head
    output = Conv3D(Ncl, (1, 1, 1), padding='same', activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output)
    return model

