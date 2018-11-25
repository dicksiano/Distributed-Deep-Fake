import cv2
import numpy
import os

from model import autoencoder_A
from model import autoencoder_B
from model import encoder, decoderA, decoderB

NUM_OF_EPOCHS = 1000000
BATCH_SIZE = 64
        
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


def main():
    imagesA = loadImages( "data/A" ) / 255.0
    imagesB = loadImages( "data/B" ) / 255.0

    imagesA += imagesB.mean( axis=(0,1,2) ) - imagesA.mean( axis=(0,1,2) )

    print("Loaded", len(imagesA), "images for model A")
    print("Loaded", len(imagesB), "images for model B")

    for epoch in range(NUM_OF_EPOCHS):
        warpedA, targetA = getTrainingData( imagesA, BATCH_SIZE ) # TODO
        warpedB, targetB = getTrainingData( imagesB, BATCH_SIZE )

        lossA = autoencoder_A.trainOnBatch( warpedA, targetA )  # TODO
        lossB = autoencoder_B.trainOnBatch( warpedB, targetB )
        print(epoch, ': ', lossA, lossB )

        if epoch % 100 == 0:
            updateWeights()
            testA = targetA[0:14]
            testB = targetB[0:14]

        figureA = numpy.stack([ testA, autoencoder_A.predict( testA ), autoencoder_B.predict( testA ), ], axis=1 )
        figureB = numpy.stack([ testB, autoencoder_B.predict( testB ), autoencoder_A.predict( testB ), ], axis=1 )

        figure = numpy.concatenate( [ figureA, figureB ], axis=0 )
        figure = figure.reshape( (4,7) + figure.shape[1:] )
        figure = stackImages( figure ) # TODO

        figure = numpy.clip( figure * 255, 0, 255 ).astype('uint8')

if __name__ == "__main__":
    main()