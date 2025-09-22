import numpy as np
import cv2 as cv
import os
import sys
import random

foldern = 'dataset/generated/noisy'
folderc = 'dataset/generated/clean'

# Check and create folders if necessary
folders = ["dataset", "dataset/generated", foldern, folderc]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

def harris_corner(picture):
    img = cv.imread(picture)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    gray = np.float32(gray)

    blur = cv.GaussianBlur(gray, (7,7), 1.5)

    dst = cv.cornerHarris(blur,3,5,0.04)

    dst = cv.dilate(dst,None)

    

    dynamic_threshold = 0.01 * dst.max()

    nms_window_size = 5

    # Locate corners by removing non-maxima in a larger window
    keypoints_nms = []
    for y in range(nms_window_size // 2, dst.shape[0] - nms_window_size // 2):
        for x in range(nms_window_size // 2, dst.shape[1] - nms_window_size // 2):
            local_patch = dst[y - nms_window_size // 2 : y + nms_window_size // 2 + 1,
                              x - nms_window_size // 2 : x + nms_window_size // 2 + 1]
            if dst[y, x] == local_patch.max() and dst[y, x] > dynamic_threshold:
                keypoints_nms.append((x, y))

    keypoints_final = []
    for (x, y) in keypoints_nms:
        if all(np.sqrt((x - kx) ** 2 + (y - ky) ** 2) > 10 for kx, ky in keypoints_final):
            keypoints_final.append((x, y))

    # Mark detected keypoints
    for x, y in keypoints_final:
        cv.circle(img, (x, y), 6, (0, 0, 255), -1)

    nb_corner_detec = len(keypoints_final)

    print(f"The file {picture} has {nb_corner_detec} corners detected.\n")

    return nb_corner_detec

def generate_image(i):

    size_x = random.randint(200, 1000)
    size_y = random.randint(200, 1000)

    size = [size_x,size_y,3]

    img = np.zeros(size, dtype=np.uint8)

    # Nb shapes to generate
    num_shapes = random.randint(4, 16)

    for _ in range(num_shapes):
        shape_type = random.choice(['line', 'rectangle', 'circle', 'ellipse', 'polygon'])
        color = tuple(random.randint(0, 255) for _ in range(3))

        if shape_type == 'line':
            # Random line
            pt1 = (random.randint(0, size_x), random.randint(0, size_y))
            pt2 = (random.randint(0, size_x), random.randint(0, size_y))
            thickness = random.randint(1, 5)
            cv.line(img, pt1, pt2, color, thickness)

        elif shape_type == 'rectangle':
            # Random rectangle
            x1, y1 = random.randint(0, size_x), random.randint(0, size_y)
            x2, y2 = random.randint(0, size_x), random.randint(0, size_y)
            thickness = random.randint(1, 5) if random.random() > 0.5 else -1
            cv.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        elif shape_type == 'circle':
            # Random circle
            center = (random.randint(0, size_x), random.randint(0, size_y))
            radius = random.randint(10, min(size_x, size_y) // 4)
            thickness = random.randint(1, 5) if random.random() > 0.5 else -1
            cv.circle(img, center, radius, color, thickness)

        elif shape_type == 'ellipse':
            # Random ellipse
            center = (random.randint(0, size_x), random.randint(0, size_y))
            axes = (random.randint(10, size_x // 4), random.randint(10, size_y // 4))
            angle = random.randint(0, 360)
            start_angle = 0
            end_angle = 360
            thickness = random.randint(1, 5) if random.random() > 0.5 else -1
            cv.ellipse(img, center, axes, angle, start_angle, end_angle, color, thickness)

        elif shape_type == 'polygon':
            # Random polygon
            num_points = random.randint(3, 8)
            points = np.array([
                (random.randint(0, size_x), random.randint(0, size_y)) for _ in range(num_points)
            ], np.int32)
            points = points.reshape((-1, 1, 2))
            is_closed = random.choice([True, False])
            thickness = random.randint(1, 5) if random.random() > 0.5 else -1
            if thickness == -1:
                cv.fillPoly(img, [points], color)
            else:
                cv.polylines(img, [points], is_closed, color, thickness)

    filename = os.path.join(folderc, f"image_gen_{i}.png")
    cv.imwrite(filename, img)

    noise = np.random.randint(random.randint(0,50), random.randint(51,100), size, dtype=np.uint8)
    img = cv.add(img, noise)

    filename = os.path.join(foldern, f"image_gen_n_{i}.png")
    cv.imwrite(filename, img)
    return img

def main():

    number_picture_to_generate = int(input("How many images do you want to generate ?\n"))

    for i in range (number_picture_to_generate):
        generate_image(i)

if __name__ == "__main__":
    main()