import os
import numpy as np
from deepface import DeepFace
from tqdm import tqdm  # Import tqdm for progress bar

# Define the paths
source_folder = "./assignment3/Pre_processed_dataset"
dest_folder = "./assignment3/Face_Embeddings"

# Create the destination folder if it doesn't exist
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

# Get a list of all jpg images in the source folder
image_files = [img_file for img_file in os.listdir(source_folder) if img_file.endswith('.jpg')]

# Use tqdm to create a progress bar
with tqdm(total=len(image_files), desc="Processing Images", unit="image") as pbar:
    # Process each JPG image in the source folder
    for img_file in image_files:
        img_path = os.path.join(source_folder, img_file)

        try:
            # Using ArcFace model to analyze the image and get embeddings
            result = DeepFace.represent(img_path, model_name='Facenet', enforce_detection=False)
            embeddings = result[0]['embedding']

            # Print the dimension of the embeddings
            #print(f"\rProcessed {img_file}: Embedding Dimension = {len(embeddings)}", end='')

            # Save the embeddings as a .npy file
            embedding_filename = img_file.replace('.jpg', '.npy')  # Change the file extension to .npy
            embedding_path = os.path.join(dest_folder, embedding_filename)
            np.save(embedding_path, embeddings)

        except Exception as e:
            print(f"Failed to process {img_file}: {e}")

        # Update the progress bar
        pbar.update(1)