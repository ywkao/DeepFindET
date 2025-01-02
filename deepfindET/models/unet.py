from tensorflow.keras.layers import (Conv3D, MaxPooling3D, UpSampling3D, 
                                     BatchNormalization, LeakyReLU, Dropout, 
                                     Input, concatenate)
from tensorflow.keras.models import Model

def conv_block(x, filters, kernel_size=(3,3,3), dropout_rate=0.0):
    """
    A convolutional block that applies:
       Conv3D -> BatchNorm -> LeakyReLU -> (optional Dropout)
       Conv3D -> BatchNorm -> LeakyReLU
    """
    x = Conv3D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    
    x = Conv3D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    return x

def my_unet_model(dim_in, Ncl, filters=[48, 64, 80], dropout_rate=0.0):
    """
    A "plain" U-Net architecture that more closely mimics the structure of your res_unet.py:
      - Similar filter arrangement
      - 4 repeated blocks in the bottleneck
      - Use of BatchNorm, LeakyReLU, and optional Dropout
    """
    
    inputs = Input(shape=(dim_in, dim_in, dim_in, 1))
    
    # ---------- Encoder ----------
    down_layers = []
    x = inputs
    for f in filters[:-1]:
        x = conv_block(x, f, dropout_rate=dropout_rate)
        down_layers.append(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    # ---------- Bottleneck (repeat N times, here 4 for illustration) ----------
    for _ in range(4):
        x = conv_block(x, filters[-1], dropout_rate=dropout_rate)

    # ---------- Decoder ----------
    # We reverse filters[:-1] because we "come back" up in reverse order
    for f, skip_connection in zip(reversed(filters[:-1]), reversed(down_layers)):
        x = UpSampling3D(size=(2, 2, 2))(x)
        x = concatenate([x, skip_connection])
        
        # In res_unet.py, each decoder level has multiple blocks
        # (e.g. 2 residual_block calls). We mimic that pattern:
        x = conv_block(x, f, dropout_rate=dropout_rate)
        x = conv_block(x, f, dropout_rate=dropout_rate)

    # ---------- Output Layer ----------
    outputs = Conv3D(Ncl, (1, 1, 1), padding='same', activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

