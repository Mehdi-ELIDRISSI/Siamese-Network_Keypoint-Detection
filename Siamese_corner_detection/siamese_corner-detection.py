# Author : EL IDRISSI Mehdi

import os
import cv2 as cv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
from sklearn.cluster import DBSCAN

import generate_dataset  # Script de generation du dataset

# ========================== CONFIGURATION ==========================
FOLDER_PICTURES_CLEAN = 'dataset/generated/clean'
FOLDER_PICTURES_NOISY = 'dataset/generated/noisy'
PATCH_SIZE = 32
REFERENCE_IMAGES = 5
trained_model = None
# ===================================================================

# ========================== SIAMESE NETWORK ========================
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256*4*4,512),
            nn.ReLU(),
            nn.Linear(512,128)
        )

    def forward_one(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward(self, input1, input2):
        return self.forward_one(input1), self.forward_one(input2)
# ===================================================================

# ========================== CONTRASTIVE LOSS =======================
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss,self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((1-label)*torch.pow(euclidean_distance,2) +
                          label*torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),2))
        return loss
# ===================================================================

# ========================== DATASET =================================
class CornerDataset(Dataset):
    def __init__(self, image_paths_clean, corner_coords):
        self.image_paths_clean = image_paths_clean
        self.corner_coords = corner_coords

    def __len__(self):
        return len(self.image_paths_clean)

    def __getitem__(self, idx):
        clean_path = self.image_paths_clean[idx]
        clean_img = cv.imread(clean_path, cv.IMREAD_GRAYSCALE)

        noisy_img_name = os.path.basename(clean_path).replace('image_gen_','image_gen_n_')
        noisy_img_path = os.path.join(FOLDER_PICTURES_NOISY,noisy_img_name)
        noisy_img = cv.imread(noisy_img_path, cv.IMREAD_GRAYSCALE)
        if noisy_img is None:
            noisy_img = clean_img.copy()

        corners = self.corner_coords[idx]
        patch1, patch2, label = self.generate_patch_pairs(clean_img, noisy_img, corners)
        return patch1, patch2, label

    def to_tensor(self, patch):
        patch = cv.resize(patch,(PATCH_SIZE,PATCH_SIZE))
        return torch.tensor(patch,dtype=torch.float32).unsqueeze(0)/255.0

    def generate_patch_pairs(self, clean_img, noisy_img, corners):
        h,w = clean_img.shape
        if not corners:
            x=random.randint(0,w-PATCH_SIZE)
            y=random.randint(0,h-PATCH_SIZE)
            patch = clean_img[y:y+PATCH_SIZE,x:x+PATCH_SIZE]
            return self.to_tensor(patch), self.to_tensor(patch), torch.tensor([1.0])

        cx,cy = random.choice(corners)
        patch_clean = clean_img[max(0,cy-PATCH_SIZE//2):cy+PATCH_SIZE//2,
                                max(0,cx-PATCH_SIZE//2):cx+PATCH_SIZE//2]
        patch_noisy = noisy_img[max(0,cy-PATCH_SIZE//2):cy+PATCH_SIZE//2,
                                max(0,cx-PATCH_SIZE//2):cx+PATCH_SIZE//2]
        pos_pair = (self.to_tensor(patch_clean), self.to_tensor(patch_noisy), torch.tensor([0.0]))

        while True:
            nx,ny = random.randint(0,w-PATCH_SIZE), random.randint(0,h-PATCH_SIZE)
            if all(np.sqrt((nx-x)**2 + (ny-y)**2) > 10 for x,y in corners):
                break
        patch_neg = noisy_img[ny:ny+PATCH_SIZE,nx:nx+PATCH_SIZE]
        neg_pair = (self.to_tensor(patch_clean), self.to_tensor(patch_neg), torch.tensor([1.0]))
        return pos_pair if random.random()<0.5 else neg_pair
# ===================================================================

# ========================== DETECTION CLASSIQUE ====================
def detect_corners(image_path, use_orb=True, use_fast=True, use_harris=True):
    img = cv.imread(image_path)
    if img is None: return [],[],[]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners_harris, corners_orb, corners_fast = [],[],[]

    if use_harris:
        gray_float = np.float32(gray)
        dst = cv.cornerHarris(gray_float,3,5,0.04)
        dst = cv.dilate(dst,None)
        threshold = 0.01*dst.max()
        corners_harris = [(x,y) for y in range(dst.shape[0]) for x in range(dst.shape[1]) if dst[y,x]>threshold]

    if use_orb:
        orb = cv.ORB_create()
        kp = orb.detect(gray,None)
        corners_orb = [(int(k.pt[0]),int(k.pt[1])) for k in kp]

    if use_fast:
        fast = cv.FastFeatureDetector_create()
        kp = fast.detect(gray,None)
        corners_fast = [(int(k.pt[0]),int(k.pt[1])) for k in kp]

    print(f"\nImage {os.path.basename(image_path)} → Harris:{len(corners_harris)} | ORB:{len(corners_orb)} | FAST:{len(corners_fast)}")
    return corners_harris, corners_orb, corners_fast
# ===================================================================

# ========================== CLUSTERING ============================
def group_nearby_points(corners, eps=6.0, min_samples=2):
    if not corners: return []
    corners_array = np.array(corners)
    db = DBSCAN(eps=eps,min_samples=min_samples)
    labels = db.fit_predict(corners_array)
    clusters = [corners_array[labels==k] for k in set(labels) if k!=-1]
    return clusters
# ===================================================================

# ========================== TRAINING ==============================
def train_siamese_network(nb_trains, nb_epoch):
    model = load_siamese_model()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    image_paths = [os.path.join(FOLDER_PICTURES_CLEAN,f'image_gen_{i}.png') for i in range(nb_trains)]
    corner_coords = [sum(detect_corners(path),[]) for path in image_paths]
    dataset = CornerDataset(image_paths,corner_coords)
    dataloader = DataLoader(dataset,batch_size=16,shuffle=True)

    for epoch in range(nb_epoch):
        for patch1,patch2,label in dataloader:
            optimizer.zero_grad()
            output1, output2 = model(patch1,patch2)
            loss = criterion(output1,output2,label)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{nb_epoch} - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(),"siamese_model-harris-orb-fast.pth")
    print("✅ Modele sauvegarde : siamese_model-harris-orb-fast.pth")
    return model
# ===================================================================

# ========================== REFERENCE EMBEDDINGS ===================
def precalc_reference_embeddings(model, nb_reference=REFERENCE_IMAGES):
    """
    Pre-calcule les embeddings des patches de reference pour les premieres images
    et retourne un tensor de dimension [N_patches, embedding_dim].
    """
    image_files = sorted([f for f in os.listdir(FOLDER_PICTURES_CLEAN) if f.endswith('.png')])[:nb_reference]
    reference_embeddings = []
    stride = PATCH_SIZE // 2  # pour correspondre a evaluate_with_siamese

    for idx, fname in enumerate(image_files):
        path = os.path.join(FOLDER_PICTURES_CLEAN, fname)
        img_gray = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue
        h, w = img_gray.shape

        for y in range(0, h - PATCH_SIZE, stride):
            for x in range(0, w - PATCH_SIZE, stride):
                patch = img_gray[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
                with torch.no_grad():
                    embedding, _ = model(patch_tensor, patch_tensor)
                    reference_embeddings.append(embedding.squeeze(0))

        print(f"Pre-calcul embeddings : {idx + 1}/{len(image_files)} images", end="\r")
    print()

    if reference_embeddings:
        reference_embeddings = torch.stack(reference_embeddings)
        # Sauvegarde pour reutilisation
        torch.save(reference_embeddings, "reference_embeddings.pt")
        print("✅ Embeddings de reference sauvegardes dans 'reference_embeddings.pt'")
    else:
        reference_embeddings = torch.empty((0, 128))  # fallback si pas d'image
        print("⚠️ Aucun embedding calcule.")

    return reference_embeddings
# ===================================================================

# ========================== EVALUATION =============================
def evaluate_with_siamese(model, image_path, embedding_file="reference_embeddings.pt"):
    import os

    # Lire l'image en couleur (pour affichage) et en gris (pour patches)
    img_color = cv.imread(image_path)
    if img_color is None:
        print("⚠️ Image invalide :", image_path)
        return

    img_gray = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img_gray is None:
        print("⚠️ Impossible de lire l'image en grayscale :", image_path)
        return

    h, w = img_gray.shape
    stride = PATCH_SIZE // 2
    detected_points_model = []

    # --- Gestion des embeddings de reference ---
    if os.path.exists(embedding_file):
        reference_embeddings = torch.load(embedding_file)
        print(f"✅ Embeddings charges depuis '{embedding_file}'")
    else:
        reference_patches = []
        for y in range(0, h - PATCH_SIZE, stride):
            for x in range(0, w - PATCH_SIZE, stride):
                patch = img_gray[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
                reference_patches.append(patch_tensor)
        with torch.no_grad():
            # Calcul des embeddings Siamese
            reference_embeddings = torch.stack([model(p, p)[0].squeeze(0) for p in reference_patches])
        torch.save(reference_embeddings, embedding_file)
        print(f"✅ Embeddings calcules et sauvegardes dans '{embedding_file}'")

    # --- Detection Siamese ---
    total = ((h - PATCH_SIZE) // stride) * ((w - PATCH_SIZE) // stride)
    count = 0
    for y in range(0, h - PATCH_SIZE, stride):
        for x in range(0, w - PATCH_SIZE, stride):
            patch = img_gray[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
            with torch.no_grad():
                patch_embedding, _ = model(patch_tensor, patch_tensor)
                patch_embedding = patch_embedding.squeeze(0)
                distances = torch.norm(reference_embeddings - patch_embedding, dim=1)
                score = distances.min().item()
            detected_points_model.append((x, y, score))
            count += 1
            if count % 100 == 0 or count == total:
                print(f"Progression : {count}/{total} ({count / total * 100:.1f}%)", end="\r")

    # --- Points avant clustering ---
    scores = [s for _, _, s in detected_points_model]
    threshold = np.percentile(scores, 50)
    points_before_clustering = [(x, y) for x, y, s in detected_points_model if s < threshold]

    # Detection classiques
    corners_harris, corners_orb, corners_fast = detect_corners(image_path)

    print(f"\n\n\n\n__________________________\n\nHarris : {len(corners_harris)} points | ORB : {len(corners_orb)} | FAST : {len(corners_fast)}\n\n")

    # Affichage nombre de points Siamese avant clustering
    print(f"Siamese avant clustering : {len(points_before_clustering)} points")


    # --- Clustering DBSCAN ---
    cluster_centers_model = []
    if points_before_clustering:
        corners_array = np.array(points_before_clustering)
        db = DBSCAN(eps=PATCH_SIZE//2, min_samples=2)  # eps=16 pixels, min_samples=2
        labels = db.fit_predict(corners_array)
        clusters = [corners_array[labels == k] for k in set(labels) if k != -1]
        cluster_centers_model = [np.mean(c, axis=0).astype(int).tolist() for c in clusters]


    # Affichage nombre de points Siamese apres clustering
    print(f"Siamese apres clustering : {len(cluster_centers_model)} points")


    # --- Creation des 5 images pour affichage ---
    images = []

    # 1. Harris
    img_harris = img_color.copy()
    for x, y in corners_harris:
        cv.circle(img_harris, (x, y), 2, (255, 0, 0), -1)
    images.append(("Harris", img_harris))

    # 2. ORB
    img_orb = img_color.copy()
    for x, y in corners_orb:
        cv.circle(img_orb, (x, y), 2, (0, 255, 0), -1)
    images.append(("ORB", img_orb))

    # 3. FAST
    img_fast = img_color.copy()
    for x, y in corners_fast:
        cv.circle(img_fast, (x, y), 2, (0, 255, 255), -1)
    images.append(("FAST", img_fast))

    # 4. Siamese avant clustering
    img_siamese_before = img_color.copy()
    for x, y in points_before_clustering:
        cv.circle(img_siamese_before, (x, y), 2, (0, 0, 255), -1)
    images.append(("Siamese Avant Clustering", img_siamese_before))

    # 5. Siamese apres clustering
    img_siamese_after = img_color.copy()
    for x, y in cluster_centers_model:
        cv.circle(img_siamese_after, (x, y), 3, (0, 0, 255), -1)
    images.append(("Siamese Apres Clustering", img_siamese_after))

    # --- Affichage ---
    for title, img_disp in images:
        cv.imshow(title, img_disp)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return points_before_clustering, cluster_centers_model, corners_harris, corners_orb, corners_fast
# ===================================================================

# ========================== UTILITAIRES ==========================
def load_siamese_model():
    model = SiameseNetwork()
    global trained_model
    if os.path.exists("siamese_model-harris-orb-fast.pth"):
        model.load_state_dict(torch.load("siamese_model-harris-orb-fast.pth"))
        model.eval()
        print("✅ Modele charge depuis siamese_model-harris-orb-fast.pth")
    else:
        print("⚠️ Aucun modele trouve, creation d’un nouveau modele.")
        for layer in model.modules():
            if isinstance(layer,(nn.Conv2d,nn.Linear)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None: nn.init.zeros_(layer.bias)
    trained_model = model
    return model

def train_model():
    files = [f for f in os.listdir(FOLDER_PICTURES_CLEAN) if f.endswith('.png')]
    nb_trains = int(input(f"Nombre d'images pour l'entraînement (max={len(files)}) : "))
    nb_epoch = int(input("Nombre d'epoques : "))
    return train_siamese_network(nb_trains, nb_epoch)

def test_model(trained_model):
    if trained_model is None:
        print("⚠️ Aucun modele entraîne.")
        return
    path_test = input("Chemin de l'image a tester : ")
    print('\n')
    evaluate_with_siamese(trained_model, path_test, embedding_file="reference_embeddings.pt")

def generate_dataset_call():
    generate_dataset.main()
# ===================================================================

# ========================== MAIN ===================================
def main():
    trained_model = load_siamese_model()
    while True:
        print("\n=== MENU PRINCIPAL ===")
        print("0. Generer un dataset")
        print("1. Entraîner le modele")
        print("2. Tester une image")
        print("3. Quitter")
        choix = input("Choix : ")
        if choix=="0":
            generate_dataset_call()
        elif choix=="1":
            trained_model = train_model()
        elif choix=="2":
            test_model(trained_model)
        elif choix=="3":
            print("Fin du programme.")
            break
        else:
            print("Option invalide.\n")

if __name__=="__main__":
    main()
