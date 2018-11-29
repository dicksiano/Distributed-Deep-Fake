import cv2 as cv
import os

print('')
trump_path = os.path.join('data', 'trump')
cage_path = os.path.join('data', 'cage')

print("Opening TRUMP files...")
trumpfiles = [f for f in os.listdir(trump_path) if os.path.isfile(os.path.join(trump_path, f))]

print("Resizing TRUMP images to (256,256)")
for f in trumpfiles:
  img_path = os.path.join(trump_path, f)
  print(f, img_path)
  org_img = cv.imread(img_path,3)
  if org_img.shape != (256,256,3):
    rsz_img = cv.resize(org_img, (256,256))
    cv.imwrite(img_path, rsz_img)

print("Opening CAGE files...")
cagefiles = [f for f in os.listdir(cage_path) if os.path.isfile(os.path.join(cage_path, f))]

print("Resizing CAGE images to (256,256)")
for f in cagefiles:
  img_path = os.path.join(cage_path, f)
  print(f, img_path)
  org_img = cv.imread(img_path,3)
  if org_img.shape != (256,256,3):
    rsz_img = cv.resize(org_img, (256,256))
    cv.imwrite(img_path, rsz_img)