import cv2
import numpy
import os

from image_augmentation import random_transform
from image_augmentation import random_warp

from model import autoencoder_A
from model import autoencoder_B
from model import encoder, decoderA, decoderB

NUM_OF_EPOCHS = 1000000
batchSize = 64
        
try:
    encoder .load_weights("models/encoder.h5"  )
    decoderA.load_weights("models/decoder_A.h5")
    decoderB.load_weights("models/decoder_B.h5")
except:
    print("No models loaded!")

def loadImages( directory, convert=None ):
    imgPaths = [ x.path for x in os.scandir( directory ) if x.name.endswith(".jpg") or x.name.endswith(".png") ]
    loadedImags = ( cv2.imread(fn) for fn in imgPaths )

    if convert:
        loadedImags = ( convert(img) for img in loadedImags )

    imags = numpy.empty( ( len(imgPaths), ) + loadedImags[0].shape, dtype=loadedImags[0].dtype )

    for i,image in enumerate( loadedImags ):
        imags[i] = image
    return imags

def updateWeights():
    encoder  .save_weights( "models/encoder.h5"   )
    decoderA.save_weights( "models/decoder_A.h5" )
    decoderB.save_weights( "models/decoder_B.h5" )
    print( "/n'Updating weights :)" )

random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.4,
    }

def getTrainingData( images, batchSize ):
    indices = numpy.random.randint( len(images), size=batchSize )
    for i,index in enumerate(indices):
        image = images[index]
        image = random_transform( image, **random_transform_args )
        warpedImg, targetImg = random_warp( image )

        if i == 0:
            warpedImages = numpy.empty( (batchSize,) + warpedImg.shape, warpedImg.dtype )
            targetImages = numpy.empty( (batchSize,) + targetImg.shape, warpedImg.dtype )

        warpedImages[i] = warpedImg
        targetImages[i] = targetImg

    return warpedImages, targetImages

def getTransposeAxes( n ):
    if n % 2 == 0:
        y = list( range( 1, n-1, 2 ) )
        x = list( range( 0, n-1, 2 ) )
    else:
        y = list( range( 0, n-1, 2 ) )
        x = list( range( 1, n-1, 2 ) )
    return y, x, [n-1]

def stackImages( images ):
    imagesShape = numpy.array( images.shape )
    newAxes = getTransposeAxes( len( imagesShape ) )
    newShape = [ numpy.prod( imagesShape[x] ) for x in newAxes ]
    return numpy.transpose( images, axes = numpy.concatenate( newAxes ) ).reshape( newShape )


def main():
    imagesA = loadImages( "data/A" ) / 255.0
    imagesB = loadImages( "data/B" ) / 255.0

    imagesA += imagesB.mean( axis=(0,1,2) ) - imagesA.mean( axis=(0,1,2) )

    print("Loaded", len(imagesA), "images for model A")
    print("Loaded", len(imagesB), "images for model B")

    for epoch in range(NUM_OF_EPOCHS):
        warpedA, targetA = getTrainingData( imagesA, batchSize ) 
        warpedB, targetB = getTrainingData( imagesB, batchSize )

        lossA = autoencoder_A.train_on_batch( warpedA, targetA )  
        lossB = autoencoder_B.train_on_batch( warpedB, targetB )
        print(epoch, ': ', lossA, lossB )

        if epoch % 100 == 0:
            updateWeights()
            testA = targetA[0:14]
            testB = targetB[0:14]

        figureA = numpy.stack([ testA, autoencoder_A.predict( testA ), autoencoder_B.predict( testA ), ], axis=1 )
        figureB = numpy.stack([ testB, autoencoder_B.predict( testB ), autoencoder_A.predict( testB ), ], axis=1 )

        figure = numpy.concatenate( [ figureA, figureB ], axis=0 )
        figure = figure.reshape( (4,7) + figure.shape[1:] )
        figure = stackImages( figure ) 

        figure = numpy.clip( figure * 255, 0, 255 ).astype('uint8')

if __name__ == "__main__":
    main()