import cv2 as cv
import numpy as np
import os
from face_recognition import face_locations
from dlib import DLIB_USE_CUDA

from autoencoder import autoencoderA
from autoencoder import autoencoderB
from autoencoder import encoder, decoderA, decoderB
        
try:
    encoder .load_weights("models/encoder.h5"  )
    decoderA.load_weights("models/decoder_A.h5")
    decoderB.load_weights("models/decoder_B.h5")
except:
    print("No models loaded!")
    exit()

def read_video(filepath):
  vidcap = cv.VideoCapture(filepath)
  
  frm_len = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
  frm_cnt = 0

  while frm_cnt < frm_len:
    vidcap.set(cv.CAP_PROP_POS_FRAMES, frm_cnt)
    success, frm_img = vidcap
    if not success:
      print('Cannot read video!')
      break
    
    fr_model = 'hog'
    if DLIB_USE_CUDA:
      fr_model = 'cnn'
    face_boxes = face_locations(frm_img, model=fr_model)
    face_boxes = None
    cpy_img = np.copy(frm_img)
    for box in face_boxes:
      top, right, bottom, left = box
      cv.rectangle(cpy_img,(left, top), (right, bottom), (0,255,0),3)

    cv.imshow(filepath, cpy_img)
    key = cv.waitKey(0) & 0xFF
    if key == ord('q'):
        break
      
    frm_cnt += 1

def main():
  videopath = os.path.join('data', 'video', 'LGqtS7TpfGs.mp4')
  read_video(videopath)

if __name__ == '__main__':
  main()