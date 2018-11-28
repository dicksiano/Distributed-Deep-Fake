import youtube_dl
import os
import time
import cv2
import numpy as np
import tqdm
from face_recognition import face_locations, face_encodings, compare_faces

class Extract:
  def __init__(self, arguments):
    self.arguments = arguments
    if(self.arguments['debug']):
      print('List parameters: ', self.arguments)
    path = os.path.join('data', 'video', '%(id)s.%(ext)s')
    self.ydl_opts = {
      'outtmpl': path,
      'download_archive': 'already_downloaded'
    }

  
  def download_videos(self, links: list, ydl_opts):
    video_info = []
    with youtube_dl.YoutubeDL(self.ydl_opts) as ydl:
      for link in links:
        video_info.append(ydl.extract_info(link, download=True))
    return video_info

  def save_face(self, info):
    video_file = info['id'] + '.mp4'
    name = self.arguments['name'][0]
    video_path = os.path.join('data', 'video', video_file)
    faces_path = os.path.join('data', 'faces', 'samples')
    if name:
      faces_path = os.path.join(faces_path, name)
      print(faces_path)
    if not os.path.exists(faces_path):
      os.makedirs(faces_path)
    
    vidcap = cv2.VideoCapture(video_path)

    frame_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    # start = time.time()
    frame_cnt = 0
    face_count = 0
    while frame_cnt < frame_length:
      vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_cnt)
      success, src_img = vidcap.read()
      img = np.copy(src_img)
      if not success:
        print('Failed to read video!')
        break
      face_boxes = face_locations(img, number_of_times_to_upsample=0, model='cnn')

      if face_boxes:
        box = face_boxes[face_count]
        cv2.rectangle(img,(box[3], box[0]), (box[1], box[2]), (0,255,0),3)

      height, width, channels = img.shape
      msg = 'Frame ' + str(frame_cnt) + ' | a/d or w/s : Switch Frames | Space: Switch Faces | Enter: Save'
      cv2.putText(img, msg, (0, height),
          cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
      
      cv2.imshow(video_file, img)
      key = cv2.waitKey(0) & 0xFF
      if key == ord('q'):
        break
      elif key == ord('a'):
        if frame_cnt > 0:
          face_count = 0
          frame_cnt -= 1
      elif key == ord('d'):
        if frame_cnt < frame_length-1:
          face_count = 0
          frame_cnt += 1
      elif key == ord('s'):
        if frame_cnt > 14:
          face_count = 0
          frame_cnt -= 15
      elif key == ord('w'):
        if frame_cnt < frame_length-16:
          face_count = 0
          frame_cnt += 15
      elif key == 32: # space
        face_count = (face_count + 1) % len(face_boxes)
      elif key == 13: # enter
        top, right, bottom, left = box
        face_img = src_img[top:bottom, left:right]
        save_path = os.path.join(faces_path, info['id']+'.png')
        cv2.imwrite(save_path, face_img)
        break
    cv2.destroyAllWindows()

  def debug_face_recognition(self, info):
    video_file = info['id'] + '.mp4'
    video_path = os.path.join('data', 'video', video_file)
    faces_path = os.path.join('data', 'faces')
    name = self.arguments['name']
    if name:
      faces_path = os.path.join(faces_path, name)
    faces_path = os.path.join(faces_path, info['id'])
    if not os.path.exists(faces_path):
      os.makedirs(faces_path)
      
    cmp_faces = False
    cmp_enconding = None
    if self.arguments['save_face']:
      cmp_faces = True
      compare_path = os.path.join('data', 'faces', 'samples')
      if name:
        compare_path = os.path.join(compare_path, name)
      compare_path = os.path.join(compare_path, info['id']+'.png')
      cmp_img = cv2.imread(compare_path, 3)
      cmp_enconding = face_encodings(cmp_img)

    if not os.path.exists(faces_path):
      os.makedirs(faces_path)
    vidcap = cv2.VideoCapture(video_path)

    frame_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    # start = time.time()
    frame_cnt = 0
    for i in tqdm.tqdm(range(frame_length)):
      # break
      success, img = vidcap.read()
      img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
      rgb_img = img[:, :, ::-1] # Conver cv2 BGR to RGB
      face_boxes = face_locations(rgb_img, number_of_times_to_upsample = 0, model='cnn')
      face_encs = face_encodings(rgb_img, face_boxes)

      for enconding in face_encs:
        matches = compare_faces(cmp_enconding, enconding)
        if True in matches:
          idx = matches.index(True)
          box = face_boxes[idx]
          top, right, bottom, left = box
          face_img = img[top:bottom, left:right]
          face_path = os.path.join(faces_path, str(i)+'_'+str(idx)+'.png')
          cv2.imwrite(face_path, face_img)
      # for idx, box in enumerate(face_boxes):
      #   top, right, bottom, left = box
      #   face_img = img[top:bottom, left:right]
      #   face_path = os.path.join(faces_path, str(i)+'_'+str(idx)+'.png')
      #   cv2.imwrite(face_path, face_img)

      # for box in face_box:
      #   cv2.rectangle(img,(box[3], box[0]), (box[1], box[2]), (0,255,0),3)

      # height, width, channels = img.shape
      # cv2.putText(img, 'Frame ' + str(frame_cnt), (0, height),
      #     cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
      
      # cv2.imshow(video_file, img)
      # if cv2.waitKey(25) & 0xFF == ord('q'):
      #   break

  def process(self):
    videos_info = self.download_videos(self.arguments['links'], self.ydl_opts)

    for info in videos_info:
      if self.arguments['save_face']:
        self.save_face(info)
      self.debug_face_recognition(info)