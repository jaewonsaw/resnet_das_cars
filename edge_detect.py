import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.cluster import DBSCAN


def normalize(a):
    a = (a - np.min(a))/(np.max(a) - np.min(a))
    return a


def detect_lines(filename, plot=False):
    orig_img = np.flipud(np.load(filename).T)
    img = normalize(orig_img)
    img[img<0.5] = 0
    img = (255*np.stack([img]*3, axis=-1)).astype(np.uint8)
    # Load and process image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Get (x, y) coordinates of all edge pixels
    points = np.column_stack(np.where(edges > 0))  # (y, x)
    points = points[:, ::-1]  # convert to (x, y)

    # Use DBSCAN to cluster edge points
    db = DBSCAN(eps=10, min_samples=20).fit(points)
    labels = db.labels_

    # For visualization
    result = img.copy()
    unique_labels = set(labels)
    detected_edges = []

    for label in unique_labels:
        if label == -1:
            continue  # noise

        cluster_points = points[labels == label]
        x = cluster_points[:, 0]
        y = cluster_points[:, 1]

        if len(x) < 40:
            continue

        # Fit least squares line: y = mx + b
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]

        # Filter only diagonal-ish lines
        if abs(m) < 0.2:
            continue

        # Start/end points for drawing
        x_start, x_end = np.min(x), np.max(x)
        y_start = int(m * x_start + b)
        y_end = int(m * x_end + b)

        cv2.line(result, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
        detected_edges.append((label, m, (x_start, y_start), (x_end, y_end)))
        #print(f"[Group {label}] Slope: {m:.2f}, Start: ({x_start}, {y_start}), End: ({x_end}, {y_end})")
        
    if plot:
        # Top: Original image
        plt.subplot(2, 1, 1)
        plt.imshow(orig_img)
        plt.title("Original DAS plot")

        # Bottom: Result image
        plt.subplot(2, 1, 2)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title("Detected Diagonal Lines")
        plt.show()
        
    return detected_edges