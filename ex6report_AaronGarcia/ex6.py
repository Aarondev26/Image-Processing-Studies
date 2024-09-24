import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def loadImagesFromFolder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(label)
    return images, np.array(labels)

#assign images and labels to array values
train_images, train_labels = loadImagesFromFolder("dataset/train")
test_images, test_labels = loadImagesFromFolder("dataset/test")
    
#convert normalize pixel values, and convert images to 1d array (of vectors) 
train_data = np.array([cv2.resize(img, (64, 64)).flatten() / 255.0 for img in train_images])
test_data = np.array([cv2.resize(img, (64, 64)).flatten() / 255.0 for img in test_images])

#print shapes of data
print(f"train data: {train_data.shape}")
print(f"train labels: {train_labels.shape}")
print(f"test data: {test_data.shape}")
print(f"test labels: {test_labels.shape}")

print("\n")
print("")

#---
#--- begin training/testing ---
#---
for i in range(1, 17, 1):
    #create and train the KNN model
    knn = KNeighborsClassifier(n_neighbors=i, metric="euclidean")
    knn.fit(train_data, train_labels)

    #predict on the test data
    predictions = knn.predict(test_data)

    #evaluate the model
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)
    print(f"n neighbor: {i}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(report)