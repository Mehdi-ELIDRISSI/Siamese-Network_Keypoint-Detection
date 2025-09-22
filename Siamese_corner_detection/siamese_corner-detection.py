# Author : EL IDRISSI Mehdi

import numpy as np
import cv2 as cv
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

import generate_dataset  # Import the python script that generates datasets

# Folders with datasets
FOLDER_PICTURES_CLEAN = 'dataset/generated/clean'
FOLDER_PICTURES_NOISY = 'dataset/generated/noisy'

# Patch size
PATCH_SIZE = 16

trained_model = None  # Initialize the model variable to None at the beginning

# Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Compatible with 16x16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x8 after maxpool

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 4x4 after maxpool

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 2x2 after maxpool
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 2 * 2, 512),  # Adapt to the new final size
            nn.ReLU(),
            nn.Linear(512, 128)  # 128-dimensional vector
        )

    def forward_one(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Flatten for Fully Connected
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# Custom Dataset
class CornerDataset(Dataset):
    def __init__(self, image_paths_clean, corner_coords, transform=None):
        self.image_paths_clean = image_paths_clean  # Clean folder
        self.corner_coords = corner_coords
        self.transform = transform

    def __len__(self):
        return len(self.image_paths_clean)

    def __getitem__(self, idx):
        # Load the clean image
        clean_img = cv.imread(self.image_paths_clean[idx], cv.IMREAD_GRAYSCALE)

        # Create the path for the noisy image using the 'noisy' folder
        image_name = os.path.basename(self.image_paths_clean[idx])
        noisy_img_name = image_name.replace('image_gen_', 'image_gen_n_')
        noisy_img_path = os.path.join(FOLDER_PICTURES_NOISY, noisy_img_name).replace("\\", "/")
        noisy_img = cv.imread(noisy_img_path, cv.IMREAD_GRAYSCALE)

        if noisy_img is None:
            print(f"⚠️ Missing noisy image ({noisy_img_path}), using the clean image instead.")
            noisy_img = clean_img.copy()
        # else :
        #     print("Path of the generated noisy image:", noisy_img_path)

        corners = self.corner_coords[idx]
        if self.transform:
            clean_img = self.transform(clean_img)
            noisy_img = self.transform(noisy_img)

        # Generate positive (real corners) and negative (noisy corners) patch pairs
        patch1, patch2, label = self.generate_patch_pairs(noisy_img, clean_img, corners, idx)
        return patch1, patch2, label

    def generate_patch_pairs(self, noisy_img, clean_img, corners, idx_image):
        h, w = clean_img.shape

        if not corners:  # Check if corners is empty
            print(f"⚠️ No corner detected, generating a random patch for image image_gen_{idx_image}.")
            return self.generate_random_patch(clean_img)

        # Ensure each corner is a tuple with exactly 2 elements
        valid_corners = [corner for corner in corners if len(corner) == 2]

        # If no valid corners, raise an error
        if not valid_corners:
            print(f"⚠️ No valid corner found for image image_gen_{idx_image}.")
            return self.generate_random_patch(clean_img)

        # If valid corners are available, select one randomly
        cx, cy = random.choice(valid_corners)
        patch1 = clean_img[max(0, cy - PATCH_SIZE // 2):cy + PATCH_SIZE // 2, max(0, cx - PATCH_SIZE // 2):cx + PATCH_SIZE // 2]

        # Negative corner (noisy - generate a random corner not close to a real corner)
        while True:
            nx, ny = random.randint(0, w - PATCH_SIZE), random.randint(0, h - PATCH_SIZE)
            if not any(np.sqrt((nx - x) ** 2 + (ny - y) ** 2) < 10 for x, y in corners):  # Ensure it is far from the real corners
                break
        patch2 = noisy_img[ny:ny + PATCH_SIZE, nx:nx + PATCH_SIZE]

        # Resize patches
        patch1 = cv.resize(patch1, (PATCH_SIZE, PATCH_SIZE))
        patch2 = cv.resize(patch2, (PATCH_SIZE, PATCH_SIZE))

        # Convert to tensors
        patch1 = torch.tensor(patch1, dtype=torch.float32).unsqueeze(0) / 255.0
        patch2 = torch.tensor(patch2, dtype=torch.float32).unsqueeze(0) / 255.0

        # If the corner is real (positive), the label is 1, otherwise it is noisy (negative) and the label is 0
        label = torch.tensor([1], dtype=torch.float32) if np.sqrt((nx - cx) ** 2 + (ny - cy) ** 2) > 10 else torch.tensor([0], dtype=torch.float32)

        return patch1, patch2, label

    def generate_random_patch(self, img, patch_size=32):
        h, w = img.shape[:2]

        # Choose a random point in the image
        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)

        # Extract a random patch
        patch = img[y:y+patch_size, x:x+patch_size]
        patch = cv.resize(patch, (PATCH_SIZE, PATCH_SIZE))

        # Convert to tensor
        patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0) / 255.0

        return patch_tensor, patch_tensor, torch.tensor([0], dtype=torch.float32)  # Neutral label

# Function to detect corners with Harris, ORB and FAST
def detect_corners(image_path, use_orb=True, use_fast=True, use_harris=True):
    img = cv.imread(image_path)
    if img is None:
        print(f"⚠️ Error loading image: {image_path}")
        return [], [], []  # Return empty lists if the image is invalid

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    corners_harris = []
    if use_harris:
        gray_float = np.float32(gray)
        blur = cv.GaussianBlur(gray_float, (7, 7), 1.5)
        dst = cv.cornerHarris(blur, 3, 5, 0.04)
        dst = cv.dilate(dst, None)
        threshold = 0.01 * dst.max()
        corners_harris = [(x, y) for y in range(dst.shape[0]) for x in range(dst.shape[1]) if dst[y, x] > threshold]

    corners_orb = []
    if use_orb:
        orb = cv.ORB_create()
        kp = orb.detect(gray, None)
        corners_orb = [(int(kp[i].pt[0]), int(kp[i].pt[1])) for i in range(len(kp))]

    corners_fast = []
    if use_fast:
        fast = cv.FastFeatureDetector_create()
        kp = fast.detect(gray, None)
        corners_fast = [(int(kp[i].pt[0]), int(kp[i].pt[1])) for i in range(len(kp))]

    print(f"\n__________________________________\nWorking on image {image_path} :\n")
    print(f"Harris corners detected: {len(corners_harris)}")
    print(f"ORB corners detected: {len(corners_orb)}")
    print(f"FAST corners detected: {len(corners_fast)}\n__________________________________")

    return corners_harris, corners_orb, corners_fast

# Grouping nearby points with DBSCAN
def group_nearby_points(corners, eps=3.0, min_samples=3):
    """
    Group nearby corners together using DBSCAN.

    Parameters:
        corners (list): List of detected corners as tuples (x, y).
        eps (float): The neighborhood radius within which points are considered neighbors.
        min_samples (int): The minimum number of points to form a cluster.

    Returns:
        clusters (list): List of clusters, each cluster is a list of corners.
    """

    if len(corners) == 0:
        print("⚠️ No corner to group.")
        return []

    # Convert corners to a NumPy array
    corners_array = np.array(corners)

    # Apply DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(corners_array)

    # Group points with the same label
    clusters = []
    for label in set(labels):
        if label != -1:  # Ignore "noise" (label -1)
            cluster = corners_array[labels == label]
            clusters.append(cluster)

    return clusters

# Train the model
def train_siamese_network(nb_trains, nb_epoch):
    model = load_siamese_model()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load data
    image_paths = [os.path.join(FOLDER_PICTURES_CLEAN, f'image_gen_{i}.png').replace("\\", "/") for i in range(0, nb_trains)]
    corner_coords = [sum(detect_corners(path), []) for path in image_paths]
    dataset = CornerDataset(image_paths, corner_coords)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=1, shuffle=True)

    # Training
    for epoch in range(nb_epoch + 1):
        for patch1, patch2, label in dataloader:
            optimizer.zero_grad()
            output1, output2 = model(patch1, patch2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Save the model after training
    torch.save(model.state_dict(), "siamese_model-harris-orb-fast.pth")
    print("Model saved as 'siamese_model-harris-orb-fast.pth'")

    return model

def evaluate_with_siamese(model, image_path, threshold_min=0.01, threshold_max=0.2, eps_dbscan=3.0, min_samples_dbscan=2, num_random_patches=25):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Error loading image: {image_path}")
        return [], [], []

    h, w = img.shape

    patch_scores = []

    for y in range(0, h - PATCH_SIZE, PATCH_SIZE // 4):

        print(f"Image processed at {y / (h - PATCH_SIZE) * 100} %")  # Calculation based on the y-position of the current processed patch, on the total height

        for x in range(0, w - PATCH_SIZE, PATCH_SIZE // 4):
            patch = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

            distances = []
            for _ in range(num_random_patches):
                ry = np.random.randint(0, h - PATCH_SIZE)
                rx = np.random.randint(0, w - PATCH_SIZE)
                ref_patch = img[ry:ry + PATCH_SIZE, rx:rx + PATCH_SIZE]
                ref_patch_tensor = torch.tensor(ref_patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

                with torch.no_grad():  # Add this line to disable gradient calculation
                    output1, output2 = model(patch_tensor, ref_patch_tensor)
                    distance = F.pairwise_distance(output1, output2).item()
                distances.append(distance)

            score = min(distances)
            patch_scores.append((x + PATCH_SIZE // 4, y + PATCH_SIZE // 4, score))

    # *** CALCULATE THRESHOLDS WITH PERCENTILES ***
    scores = [score for _, _, score in patch_scores]

    # Choose percentiles (e.g., 20% and 90%)
    low_percentile = 20
    high_percentile = 90

    # Define thresholds based on scores
    threshold_min = np.percentile(scores, low_percentile)
    threshold_max = np.percentile(scores, high_percentile)

    print(f"Adjusted thresholds: min={threshold_min:.4f}, max={threshold_max:.4f}")

    # *** Apply filtering with the new thresholds ***
    detected_corners = [(x, y) for x, y, score in patch_scores if threshold_min < score < threshold_max]

    # *** VISUALIZATION OF DISTRIBUTION ***
    plt.hist(scores, bins=50)  # Histogram of scores
    plt.axvline(threshold_min, color='red', linestyle='dashed', linewidth=2, label=f"Min ({low_percentile}th pctl)")
    plt.axvline(threshold_max, color='green', linestyle='dashed', linewidth=2, label=f"Max ({high_percentile}th pctl)")
    plt.title("Distribution of Patch Scores")
    plt.xlabel("Similarity Score (distance)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # Grouping with DBSCAN
    if detected_corners:
        corners_array = np.array(detected_corners)

        # *** ADJUST DBSCAN PARAMETERS HERE ***
        db = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan)  # Start with these
        labels = db.fit_predict(corners_array)

        clusters_siamese = []
        for label in set(labels):
            if label != -1:
                cluster_points = corners_array[labels == label]
                clusters_siamese.append(cluster_points.tolist())
    else:
        clusters_siamese = []

    # Group and display corners detected by Harris, ORB and FAST for comparison
    corners_harris, corners_orb, corners_fast = detect_corners(image_path, use_orb=True, use_fast=True, use_harris=True)

    clusters_harris = group_nearby_points(corners_harris, eps=3.0, min_samples=3)
    clusters_orb = group_nearby_points(corners_orb, eps=3.0, min_samples=3)
    clusters_fast = group_nearby_points(corners_fast, eps=3.0, min_samples=3)

    print(f"\nGrouping corners by Harris: {len(clusters_harris)} clusters")
    print(f"Grouping corners by ORB: {len(clusters_orb)} clusters")
    print(f"Grouping corners by FAST: {len(clusters_fast)} clusters")

    print(f"\nNumber of corners detected by the Siamese model (before grouping): {len(detected_corners)}")

    print(f"Number of clusters detected by the Siamese model: {len(clusters_siamese)}\n\n")

    # Create color images to display corners
    img_siamese = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    img_harris = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    img_orb = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    img_fast = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    # Draw clusters for the Siamese model (in blue)
    for cluster in clusters_siamese:
        cluster_center = np.mean(cluster, axis=0).astype(int)
        cv.circle(img_siamese, (cluster_center[0], cluster_center[1]), 3, (255, 0, 0), -1)  # Blue

    # Draw directly the corners detected by the Siamese model (in red)
    for x, y in detected_corners:
        cv.circle(img_siamese, (x, y), 1, (0, 0, 255), -1)  # Red

    # Draw the corners detected by Harris (in blue)
    for cluster in clusters_harris:
        for x, y in cluster:
            cv.circle(img_harris, (x, y), 2, (255, 0, 0), -1)  # Blue

    # Draw the corners detected by ORB (in green)
    for cluster in clusters_orb:
        for x, y in cluster:
            cv.circle(img_orb, (x, y), 3, (0, 255, 0), -1)  # Green

    # Draw the corners detected by FAST (in yellow)
    for cluster in clusters_fast:
        for x, y in cluster:
            cv.circle(img_fast, (x, y), 3, (0, 255, 255), -1)  # Yellow

    # Display images with corners
    cv.imshow('Corners Detected by the Siamese Model', img_siamese)
    cv.imshow('Corners Detected by Harris', img_harris)
    cv.imshow('Corners Detected by ORB', img_orb)
    cv.imshow('Corners Detected by FAST', img_fast)

    # Wait for window closure
    cv.waitKey(0)
    cv.destroyAllWindows()

    return detected_corners, clusters_siamese, clusters_harris, clusters_orb, clusters_fast

def load_siamese_model():
    model = SiameseNetwork()
    global trained_model

    if os.path.exists("siamese_model-harris-orb-fast.pth"):
        model.load_state_dict(torch.load("siamese_model-harris-orb-fast.pth"))
        model.eval()
        print("Model loaded from 'siamese_model-harris-orb-fast.pth'")
        trained_model = model
    else:
        print("No model found. Creating a new model.")
        # Initialize weights if no model is found. This is crucial!
        for layer in model.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(layer.weight)  # Or other initialization method
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    return model

def train_model():
    fichiers = [f for f in os.listdir(FOLDER_PICTURES_CLEAN) if os.path.isfile(os.path.join(FOLDER_PICTURES_CLEAN, f))]
    nombre_fichiers = len(fichiers)

    nb_trains = int(input(f"On how many images do you want to train? (Max = {nombre_fichiers})\n"))
    nb_epoch = int(input("How many epochs?\n"))

    trained_model = train_siamese_network(nb_trains, nb_epoch)
    print("Training finished!")
    return trained_model

def test_model(trained_model):
    if trained_model is None:
        print("⚠️ No trained model. Please train a model first.")
        return

    path_test = input("On which image do you want to test the model?\n")
    evaluate_with_siamese(trained_model, path_test)

def generate_dataset_call():
    generate_dataset.main()
    return

def main():
    trained_model = load_siamese_model()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    while True:
        print("\n=== Main Menu ===")
        print("0. Generate a dataset")
        print("1. Train the model")
        print("2. Test an image")
        print("3. Quit\n")

        choix = input("Choose an option: ")

        if choix == "0":
            generate_dataset_call()
        elif choix == "1":
            trained_model = train_model()
        elif choix == "2":
            test_model(trained_model)
        elif choix == "3":
            print("Program finished.")
            break
        else:
            print("\nInvalid option. Please try again.\n")

if __name__ == "__main__":
    main()