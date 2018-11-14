import youtube_dl
import os
import time
import cv2
from face_recognition import face_locations

class Extract:
  def __init__(self, arguments):
    self.arguments = arguments

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

  def debug_face_recognition(self, info):
    video_file = info['id'] + '.mp4' 
    path = os.path.join('data', 'video', video_file)
    vidcap = cv2.VideoCapture(path)

    start = time.time()
    success, img = vidcap.read()
    img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
    frame_cnt = 0
    while success:
      face_box = face_locations(img, model='cnn')
      for box in face_box:
        cv2.rectangle(img,(box[3], box[0]), (box[1], box[2]), (0,255,0),3)

      height, width, channels = img.shape
      cv2.putText(img, 'Frame ' + str(frame_cnt), (0, height),
          cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
      
      cv2.imshow(video_file, img)
      
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break
      if frame_cnt % 100 == 0:
        time_elapsed = time.time() - start
        print('Frame', frame_cnt)
        print('Shape', img.shape)
        print('Time Elapsed:', time_elapsed, 's')
        print('Medium Frame Rate:', frame_cnt / time_elapsed)
        print()
      success, img = vidcap.read()
      img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
      frame_cnt += 1

  def process(self):
    videos_info = self.download_videos(self.arguments['links'], self.ydl_opts)

    for info in videos_info:
      self.debug_face_recognition(info)