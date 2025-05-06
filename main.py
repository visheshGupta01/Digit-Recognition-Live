import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os
import ssl
import time

# Setting HTTPS connection (for OpenML dataset download)
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

# Load MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
print(pd.Series(y).value_counts())

# Preprocessing
classes = [str(i) for i in range(10)]
nclasses = len(classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=7500, test_size=2500, random_state=9)
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

# Train model
clf = LogisticRegression(solver='saga', multi_class='multinomial', max_iter=1000).fit(X_train_scaled.values, y_train)

# Evaluate
y_pred = clf.predict(X_test_scaled) 
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = grey.shape

        # Draw rectangle
        upper_left = (int(width / 2 - 56), int(height / 2 - 56))
        bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
        cv2.rectangle(grey, upper_left, bottom_right, (0, 255, 0), 2)

        # Extract ROI
        roi = grey[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

        # Preprocess ROI for prediction
        im_pil = Image.fromarray(roi)
        image_bw = im_pil.convert('L')
        image_bw_resize = image_bw.resize((28, 28), Image.Resampling.LANCZOS)
        image_bw_resize_inverted = PIL.ImageOps.invert(image_bw_resize)

        pixel_filter = 20
        min_pixel = np.percentile(image_bw_resize_inverted, pixel_filter)
        image_bw_resize_inverted_scaled = np.clip(image_bw_resize_inverted - min_pixel, 0, 255)
        max_pixel = np.max(image_bw_resize_inverted)
        if max_pixel != 0:
            image_bw_resize_inverted_scaled = np.asarray(image_bw_resize_inverted_scaled) / max_pixel
        else:
            image_bw_resize_inverted_scaled = np.asarray(image_bw_resize_inverted_scaled)

        test_sample = image_bw_resize_inverted_scaled.reshape(1, 784)
        test_pred = clf.predict(test_sample)

        # Display prediction
        cv2.putText(grey, f'Predicted: {test_pred[0]}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Digit Recognition', grey)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print("Error:", e)

cap.release()
cv2.destroyAllWindows()
