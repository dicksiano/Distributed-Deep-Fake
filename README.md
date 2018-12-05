# Distributed-Deep-Fake

## Install dependencies
https://gist.github.com/zhanwenchen/e520767a409325d9961072f666815bb8

## Required pip packages
 - Python 3.6
 - dlib
 - face_recognition
 - numpy
 - keras 2.2.0
 - tensorflow (tensorflow-gpu)

## Training images
 - Create data/ folder
 - Inside data folder, create folders A and B to store training images for people A and B.

## Run Distributed Version
 - dist_tf_trainer.py
 - Setup network configurations on setClusterConfig function, where ps are parameters servers.

## Run Multi-GPU version/Normal version
 - trainer.py
 - Have CUDA setup to Multi-GPU
 - autoencoder.py is setup to detect multi-gpu systems
