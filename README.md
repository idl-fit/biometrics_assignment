# Assignment 3 - Face Recognition

This repository contains scripts developed for a biometrics course assignment. 

## Scripts

### 1. pre-process.py
This script automates the preprocessing of face images using the Dlib library. It performs the following tasks:
- Downloads a specified dataset and the Dlib shape predictor.
- Processes frontal face images by detecting, aligning, cropping, and resizing them to 112x112 pixels.
- Saves the processed images in a designated folder.
- Logs any failed detections for review.

### 2. facenet_feature_extraction.py
This script extracts face embeddings from pre-processed images using the DeepFace library with the Facenet model. It performs the following tasks:
- Reads all JPEG images from a specified source directory.
- Computes embeddings for each image.
- Saves each embedding as a NumPy file (`.npy`) in a designated output directory.
