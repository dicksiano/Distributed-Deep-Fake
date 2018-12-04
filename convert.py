import cv2 as cv
import numpy as np
import os
from tqdm import tqdm
from face_recognition import face_locations, face_encodings, compare_faces
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

Amean = np.array([0.274813, 0.30193042, 0.46728145])
Bmean = np.array([0.274813, 0.30193042, 0.46728145])

def adjust_avg_color(old_face, new_face):
  """ Perform average color adjustment """
  for i in range(new_face.shape[-1]):
    old_avg = old_face[:, :, i].mean()
    new_avg = new_face[:, :, i].mean()
    diff_int = (int)(old_avg - new_avg)
    for int_h in range(new_face.shape[0]):
      for int_w in range(new_face.shape[1]):
        temp = (new_face[int_h, int_w, i] + diff_int)
        if temp < 0:
          new_face[int_h, int_w, i] = 0
        elif temp > 255:
          new_face[int_h, int_w, i] = 255
        else:
          new_face[int_h, int_w, i] = temp

def smooth_mask(old_face, new_face):
  """ Smooth the mask """
  width, height, _ = new_face.shape
  crop = slice(0, width)
  mask = np.zeros_like(new_face)
  mask[height // 15:-height // 15, width // 15:-width // 15, :] = 255
  mask = cv.GaussianBlur(mask, (15, 15), 10) # pylint: disable=no-member
  new_face[crop, crop] = (mask / 255 * new_face + (1 - mask / 255) * old_face)

def cropImg(img):
  return cv.resize(img.copy(), (80,80))[8:72,8:72]

def read_video(videopath, samplepath, autoencoder):
  vidcap = cv.VideoCapture(videopath)
  cmp_img = cv.imread(samplepath, 3)
  cmp_enc = face_encodings(cmp_img)

  fourcc = cv.VideoWriter_fourcc(*'XVID')
  outvid = cv.VideoWriter('output.avi',fourcc, 24.0, (640,480))

  frm_len = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
  frm_cnt = (5 * frm_len) // 342
  vidcap.set(cv.CAP_PROP_POS_FRAMES, frm_cnt)
  play = 1
  for i in tqdm(range(frm_len)):
  # while frm_cnt < frm_len:
    # print(frm_cnt)
    # vidcap.set(cv.CAP_PROP_POS_FRAMES, frm_cnt)
    success, frm_img = vidcap.read()
    # frm_img = cv.resize(frm_img, None, fx=0.7, fy=0.7)
    if not success:
      print('Cannot read video!')
      break
    rgb_img = frm_img[:,:,::-1] # Convert cv2 BGR to RGB
    fr_model = 'hog'
    if DLIB_USE_CUDA:
      fr_model = 'cnn'
    face_boxes = face_locations(rgb_img, model='hog')
    face_encs = face_encodings(rgb_img, face_boxes)
    cpy_img = np.copy(frm_img)
    
    for box in face_boxes:
      top, right, bottom, left = box
      # cv.rectangle(cpy_img,(left, top), (right, bottom), (0,0,255),3)

    for encoding in face_encs:
      matches = compare_faces(cmp_enc, encoding)
      for idx, match in enumerate(matches):
        if match:
          print()
          box = face_boxes[idx]
          top, right, bottom, left = box
          # print(left, top, right, bottom)
          face_img = frm_img[top:bottom, left:right]
          # cpy_img[top:bottom, left:right] = np.zeros((bottom-top, right-left, 3))
          rsz_fac_img = cropImg(face_img)
          norm_fac_img = rsz_fac_img / 255.0
          # norm_fac_img += Bmean - Amean
          norm_fac_batch = np.expand_dims(norm_fac_img, 0)
          norm_new_face = autoencoder.predict(norm_fac_batch)[0]
          new_face = np.clip(norm_new_face * 255, 0, 255).astype('uint8')
          adjust_avg_color(rsz_fac_img, new_face)
          smooth_mask(rsz_fac_img, new_face)
          new_width = (64 * face_img.shape[0]) // 80
          new_height = (64 * face_img.shape[1]) // 80
          new_face = cv.resize(new_face, (new_width, new_height))
          centerx = top+(bottom-top)//2
          centery = left+(right-left)//2
          newtop = centerx-new_width//2
          # newbottom = centerx+new_width//2
          newleft = centery-new_height//2
          # newright = centery+new_height//2
          # print(centerx, centery, newtop, newbottom, newleft, newright)
          cpy_img[newtop:newtop+new_height, newleft:newleft+new_width] = new_face
          # cv.rectangle(cpy_img,(left, top), (right, bottom), (0,255,0),3)

    success= outvid.write(cv.resize(cpy_img,(640,480)))
    # cv.imshow('Video', cpy_img)#cv.resize(cpy_img,None))
    # key = cv.waitKey(play) & 0xFF
    # if key == ord('q'):
    #     break
    # if key == ord('w'):
    #     frm_cnt += 14
    # if key == ord('s'):
    #     frm_cnt -= 14
    # if key == ord('d'):
    #     frm_cnt += 0
    # if key == ord('a'):
    #     frm_cnt -= 2
    # if key == ord('p'):
    #   if play:
    #     play = 0
    #   else:
    #     play = 1
    frm_cnt += 1
  # outvid.release()

def main():
  videopath = os.path.join('data', 'video', 'lOqpHIwpNDA.mp4')
  samplepath = os.path.join('data', 'faces', 'samples', 'jimmy', 'lOqpHIwpNDA.png')
  read_video(videopath, samplepath, autoencoderA)

if __name__ == '__main__':
  main()