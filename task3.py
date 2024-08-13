import os
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load dataset paths
def load_images_from_folder(folder, label, image_size=(64, 64)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            img = img.flatten()  # Flatten the image
            images.append(img)
            labels.append(label)
    return images, labels

# Save predictions to CSV
def save_predictions_to_csv(predictions, output_file='submission.csv'):
    df = pd.DataFrame(predictions, columns=['ImageId', 'Label'])
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# Paths to the dataset folders
cat_folder = 'add cat folder here'
dog_folder = 'add dog folder here'

# Load cat images
cat_images, cat_labels = load_images_from_folder(cat_folder, label=0)  # label 0 for cats
# Load dog images
dog_images, dog_labels = load_images_from_folder(dog_folder, label=1)  # label 1 for dogs

# Combine cat and dog images
X = np.array(cat_images + dog_images)
y = np.array(cat_labels + dog_labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the SVM model
svm = SVC(kernel='rbf')  # You can try other kernels like 'rbf'

# Train the model
svm.fit(X_train, y_train)

# Predict on the test set
y_pred = svm.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save predictions to CSV
# Assuming the test images are named in a way that corresponds to their index
test_image_ids = range(len(y_pred))  # Create an image ID list
predictions = list(zip(test_image_ids, y_pred))  # Combine IDs with predictions
save_predictions_to_csv(predictions)
