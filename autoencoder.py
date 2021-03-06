from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

from pixel_shuffler import PixelShuffler

import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.utils.training_utils import multi_gpu_model

optimizer = Adam( lr=5e-5, beta_1=0.5, beta_2=0.999 )

IMAGE_SHAPE = (64,64,3)
ENCODER_DIM = 1024

def conv( filters ):
    def block(x):
        x = Conv2D( filters, kernel_size=5, strides=2, padding='same' )(x)
        x = LeakyReLU(0.1)(x)
        return x
    return block

def upscale( filters ):
    def block(x):
        x = Conv2D( filters*4, kernel_size=3, padding='same' )(x)
        x = LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
        return x
    return block

def Encoder():
    input_ = Input( shape=IMAGE_SHAPE )
    x = input_
    x = conv( 128)(x)
    x = conv( 256)(x)
    x = conv( 512)(x)
    x = conv(1024)(x)
    x = Dense( ENCODER_DIM )( Flatten()(x) )
    x = Dense(4*4*1024)(x)
    x = Reshape((4,4,1024))(x)
    x = upscale(512)(x)
    return Model( input_, x )

def Decoder():
    input_ = Input( shape=(8,8,512) )
    x = input_
    x = upscale(256)(x)
    x = upscale(128)(x)
    x = upscale( 64)(x)
    x = Conv2D( 3, kernel_size=5, padding='same', activation='sigmoid' )(x)
    return Model( input_, x )

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    cnt = 0
    for x in local_device_protos:
        if x.device_type == 'GPU':
            cnt += 1
    return cnt

with tf.device("/cpu:0"):
    encoder = Encoder()
    decoderA = Decoder()
    decoderB = Decoder()

    input = Input( shape=IMAGE_SHAPE )

    autoencoderA = Model( input, decoderA( encoder(input) ) )
    autoencoderB = Model( input, decoderB( encoder(input) ) )

# Multi GPU models 
if get_available_gpus() > 1:
    distributed_autoencoderA = multi_gpu_model(autoencoderA, gpus=4)
    distributed_autoencoderB = multi_gpu_model(autoencoderB, gpus=4)

# Models compilations
autoencoderA.compile( optimizer=optimizer, loss='mean_absolute_error' )
autoencoderB.compile( optimizer=optimizer, loss='mean_absolute_error' )

if get_available_gpus() > 1:
    distributed_autoencoderA.compile( optimizer=optimizer, loss='mean_absolute_error' )
    distributed_autoencoderB.compile( optimizer=optimizer, loss='mean_absolute_error' )

def getAutoencoders():
    if get_available_gpus() > 1:
        return encoder, decoderA, decoderB, distributed_autoencoderA, distributed_autoencoderB
    return encoder, decoderA, decoderB, autoencoderA, autoencoderB

    