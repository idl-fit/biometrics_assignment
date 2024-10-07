import os
import cv2
import dlib
import shutil
import numpy as np
import requests
import zipfile
from tqdm import tqdm  # Import tqdm for progress bar

# Load the dlib face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Load the predictor file


def download_file(url, destination):
    """Download a file from a given URL to a specified destination."""
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad responses
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):  # Download in chunks
            f.write(chunk)


def download_and_extract_zip(url, extract_to):
    """Download a zip file from the given URL and extract it."""
    local_filename = os.path.join(extract_to, url.split('/')[-1])

    # Send a GET request
    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # Check for request errors
        # Open a local file for writing the downloaded content
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):  # Download in chunks
                f.write(chunk)

    # Extract the ZIP file
    with zipfile.ZipFile(local_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Remove the zip file after extraction
    os.remove(local_filename)


def align_face(image, landmarks):
    """Align the face based on eye coordinates."""
    left_eye_pts = [36, 37, 38, 39, 40, 41]  # Left eye landmarks
    right_eye_pts = [42, 43, 44, 45, 46, 47]  # Right eye landmarks

    left_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in left_eye_pts], axis=0)
    right_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in right_eye_pts], axis=0)

    # Calculate the angle to align the eyes horizontally
    eye_delta_x = right_eye[0] - left_eye[0]
    eye_delta_y = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(eye_delta_y, eye_delta_x))

    # Center between the eyes and face size
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    scale = 1.0

    # Rotate the image to align eyes horizontally
    rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale)
    aligned_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    return aligned_image


def preprocess_and_save(image_path, output_path):
    """Detect face, align it, crop and resize to 112x112 pixels."""
    image = cv2.imread(image_path)

    # Check if the image is loaded properly
    if image is None:
        return False  # No image loaded

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray, 1)

    if len(faces) == 0:
        return False  # No face detected

    # Use the first face detected for alignment
    face = faces[0]
    landmarks = predictor(gray, face)

    # Align the face using the landmarks
    aligned_image = image

    # Crop the face using the face bounding box
    x, y, w, h = (face.left(), face.top(), face.width(), face.height())
    cropped_face = aligned_image[y:y + h, x:x + w]

    # Resize the face to 112x112 pixels
    if cropped_face.size == 0:  # Ensure cropped_face is not empty
        return False

    resized_face = cv2.resize(cropped_face, (112, 112))

    # Save the processed image
    cv2.imwrite(output_path, resized_face)
    return True  # Success


def process_images(src_folder, dest_folder, fail_folder):
    """Process all frontal images in the source folder, and handle failed cases."""
    failed_count = 0

    images_path = os.path.join(src_folder, 'cfp-dataset/Data', 'Images')
    if not os.path.exists(images_path):
        print(f"Images directory does not exist: {images_path}")
        return failed_count

    # Calculate total images in frontal folders only
    total_images = sum(
        len(files) for indv_id in os.listdir(images_path)
        for _, _, files in os.walk(os.path.join(images_path, indv_id, 'frontal'))
        if os.path.isdir(os.path.join(images_path, indv_id, 'frontal'))  # Check if frontal folder exists
    )

    with tqdm(total=total_images, desc="Processing Images", unit="image") as pbar:
        for indv_id in os.listdir(images_path):
            frontal_folder = os.path.join(images_path, indv_id, 'frontal')
            if os.path.isdir(frontal_folder):  # Ensure it's a directory
                for img_file in os.listdir(frontal_folder):
                    if img_file.endswith('.jpg'):  # Process only JPEG images
                        img_path = os.path.join(frontal_folder, img_file)
                        img_no = img_file.split('.')[0]  # Extract the image number
                        new_filename = f"{indv_id}_{img_no}.jpg"
                        output_path = os.path.join(dest_folder, new_filename)

                        # Open and display the image
                        #image = cv2.imread(img_path)
                        #if image is not None:
                         #   cv2.imshow("Processing Image", image)

                        # Process the image: face detection, alignment, cropping, and resizing
                        if not preprocess_and_save(img_path, output_path):
                            failed_count += 1
                            # Copy the failed image to fail_folder
                            failed_output_path = os.path.join(fail_folder, new_filename)
                            shutil.copy(img_path, failed_output_path)

                        # Update the progress bar
                        pbar.update(1)

                        # Wait for a short period to display the image
                        #cv2.waitKey(100)  # Display each image for 100 milliseconds

                        # Close the image window
                        cv2.destroyAllWindows()

    return failed_count


# Main function to download, extract, and process the dataset
def main(base_path):
    """Main function to handle downloading, extraction, and processing."""

    # Create assignment3 directory inside the base path
    assignment_folder = os.path.join(base_path, 'assignment3')
    if not os.path.exists(assignment_folder):
        os.makedirs(assignment_folder)

    # URLs for the dataset and the shape predictor
    zip_url = "http://cfpw.io/cfp-dataset.zip"
    shape_predictor_url = "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"

    # Download the shape predictor file
    predictor_file = os.path.join(assignment_folder, "shape_predictor_68_face_landmarks.dat")
    download_file(shape_predictor_url, predictor_file)

    # Download and extract the dataset
    download_and_extract_zip(zip_url, assignment_folder)

    # Define destination and fail folders inside assignment3
    dest_folder = os.path.join(assignment_folder, 'Pre_processed_dataset')
    fail_folder = os.path.join(assignment_folder, 'Dlib_fail_to_detect_cases')

    # Create destination and fail-to-detect folders if they don't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    if not os.path.exists(fail_folder):
        os.makedirs(fail_folder)

    # Process the images and track failures
    failed_images = process_images(assignment_folder, dest_folder, fail_folder)
    print(f"Number of images where face detection failed: {failed_images}")


# Example usage: Change this to your desired base path
base_path = "/home/user/Desktop/Project"  # Replace with the actual path
main(base_path)
