#Assignment3 - FaceRecognition
This repository contains scripts developed for a biometrics course assignment. 

##pre-process.py
This script automates the preprocessing of face images using the Dlib library. It downloads a specified dataset and the Dlib shape predictor, processes frontal face images by detecting, aligning, cropping, and resizing them to 112x112 pixels. The processed images are saved in a designated folder, while any failed detections are logged for review.

##facenet_feature_extraction.py
This script extracts face embeddings from pre-processed images using the DeepFace library with the Facenet model. It reads all JPEG images from a specified source directory and computes their embeddings, saving each as a NumPy file (.npy) in a designated output directory.
