import cv2
import numpy
import os

from umeyama import umeyama

from autoencoder import distributed_autoencoderA
from autoencoder import distributed_autoencoderB
from autoencoder import encoder, decoderA, decoderB

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
    
    for i,image in enumerate( loadedImags ):
        if i == 0: 
            imags = numpy.empty( ( len(imgPaths), ) + image.shape, dtype=image.dtype )
        imags[i] = image
    return imags

def updateWeights():
    encoder  .save_weights( "models/encoder.h5"   )
    decoderA.save_weights( "models/decoder_A.h5" )
    decoderB.save_weights( "models/decoder_B.h5" )
    print( "/n'Updating weights :)" )

randomTransform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.4,
    }

def getTrainingData( images, batchSize ):
    indices = numpy.random.randint( len(images), size=batchSize )
    for i,index in enumerate(indices):
        image = images[index]
        image = randomTransform( image, **randomTransform_args )
        warpedImg, targetImg = randomWarp( image )

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

def randomTransform( image, rotation_range, zoom_range, shift_range, random_flip ):
    h,w = image.shape[0:2]
    rotation = numpy.random.uniform( -rotation_range, rotation_range )
    scale = numpy.random.uniform( 1 - zoom_range, 1 + zoom_range )
    tx = numpy.random.uniform( -shift_range, shift_range ) * w
    ty = numpy.random.uniform( -shift_range, shift_range ) * h
    mat = cv2.getRotationMatrix2D( (w//2,h//2), rotation, scale )
    mat[:,2] += (tx,ty)
    result = cv2.warpAffine( image, mat, (w,h), borderMode=cv2.BORDER_REPLICATE )
    if numpy.random.random() < random_flip:
        result = result[:,::-1]
    return result

def randomWarp( image ):
    assert image.shape == (256,256,3)
    range_ = numpy.linspace( 128-80, 128+80, 5 )
    mapx = numpy.broadcast_to( range_, (5,5) )
    mapy = mapx.T

    mapx = mapx + numpy.random.normal( size=(5,5), scale=5 )
    mapy = mapy + numpy.random.normal( size=(5,5), scale=5 )

    interpMapx = cv2.resize( mapx, (80,80) )[8:72,8:72].astype('float32')
    interpMapy = cv2.resize( mapy, (80,80) )[8:72,8:72].astype('float32')

    warped_image = cv2.remap( image, interpMapx, interpMapy, cv2.INTER_LINEAR )

    srcPoints = numpy.stack( [ mapx.ravel(), mapy.ravel() ], axis=-1 )
    dstPoints = numpy.mgrid[0:65:16,0:65:16].T.reshape(-1,2)
    mat = umeyama( srcPoints, dstPoints, True )[0:2]

    target_image = cv2.warpAffine( image, mat, (64,64) )

    return warped_image, target_image


def main():
    imagesA = loadImages( "data/A" ) / 255.0
    imagesB = loadImages( "data/B" ) / 255.0

    imagesA += imagesB.mean( axis=(0,1,2) ) - imagesA.mean( axis=(0,1,2) )

    print("Loaded", len(imagesA), "images for model A")
    print("Loaded", len(imagesB), "images for model B")

    for epoch in range(NUM_OF_EPOCHS):
        warpedA, targetA = getTrainingData( imagesA, batchSize ) 
        warpedB, targetB = getTrainingData( imagesB, batchSize )

        lossA = distributed_autoencoderA.train_on_batch( warpedA, targetA )  
        lossB = distributed_autoencoderB.train_on_batch( warpedB, targetB )
        print(epoch, ': ', lossA, lossB )

        if epoch % 100 == 0:
            updateWeights()
            testA = targetA[0:14]
            testB = targetB[0:14]

#        figureA = numpy.stack([ testA, autoencoderA.predict( testA ), autoencoderB.predict( testA ), ], axis=1 )
#        figureB = numpy.stack([ testB, autoencoderB.predict( testB ), autoencoderA.predict( testB ), ], axis=1 )

#        figure = numpy.concatenate( [ figureA, figureB ], axis=0 )
#        figure = figure.reshape( (4,7) + figure.shape[1:] )
#        figure = stackImages( figure ) 

#        figure = numpy.clip( figure * 255, 0, 255 ).astype('uint8')

if __name__ == "__main__":
    main()
