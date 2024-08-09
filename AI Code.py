import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import random
from PIL import Image

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_auc_score
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

df = pd.read_csv(r'C:\Users\rohan\Downloads\YICCHALLENGE\clinicaldata.csv')
df.info()
print(df.isnull().sum())
df.drop(['DoctorInCharge'], axis=1, inplace=True)

selected_features = [
    'PatientID',
    'Ethnicity',
    'Diabetes',
    'CholesterolHDL',
    'MMSE',
    'FunctionalAssessment',
    'MemoryComplaints',
    'BehavioralProblems',
    'ADL'
]

X_clinical = df[selected_features]
y_clinical = df['Diagnosis']

X_train_clinical, X_test_clinical, y_train_clinical, y_test_clinical = train_test_split(X_clinical, y_clinical, test_size = 0.2, random_state = 42)

'''
plt.figure(figsize=(25,12))
sns.heatmap(df,annot=True,fmt='.2f',cmap='coolwarm')
plt.title('Correlation Matrix',fontsize=26)
plt.show
'''

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_clinical, y_train_clinical)

y_pred_clinical = rf_model.predict(X_test_clinical)

accuracy_clinical = accuracy_score(y_test_clinical, y_pred_clinical)
report = classification_report(y_test_clinical, y_pred_clinical)

print(accuracy_clinical)

conf_matrix = confusion_matrix(y_test_clinical, y_pred_clinical)

def preprocess_input(user_input):
    input_df = pd.DataFrame([user_input])
    input_df = input_df.reindex(columns=X_train_clinical.columns, fill_value=0)
    return input_df

def predict_alzheimers(user_input):
    #Preprocess the input
    input_df = preprocess_input(user_input)
    #Make prediction
    prediction = rf_model.predict(input_df)
    return "Alzheimer's" if prediction[0] == 1 else "No Alzheimer's"

user_input = {
    'Ethnicity': 3,
    'Diabetes': 0,
    'CholesterolHDL': 81,
    'MMSE': 4,
    'FunctionalAssessment': 5,
    'MemoryComplaints': 0,
    'BehavioralProblems': 0,
    'ADL':  4
}

result = predict_alzheimers(user_input)
print(f"Prediction: {result}")

image_mapping = pd.read_csv(r'C:\Users\rohan\Downloads\YICCHALLENGE\image_mapping.csv')

demented_images_dir = r'C:\Users\rohan\Downloads\YICCHALLENGE\Dataset\Demented'
non_demented_images_dir = r'C:\Users\rohan\Downloads\YICCHALLENGE\Dataset\Non_Demented'

image_dict = pd.Series(image_mapping.imageFileName.values, index=image_mapping.PatientID).to_dict()

# Iterate through clinical data
# for index, row in df.iterrows():
#     try:
#         patient_id = row['PatientID']
#         diagnosis = row['Diagnosis']

#         patient_id = int(patient_id)
        
#         if diagnosis == 1:  # Alzheimer's diagnosis
#             images_dir = demented_images_dir
#         else:  # Non-Alzheimer's diagnosis
#             images_dir = non_demented_images_dir

#         # Look up the image filename for this patient ID
#         if patient_id in image_dict:
#             old_filename = image_dict[patient_id]
#             old_file_path = os.path.join(images_dir, old_filename)
#             new_file_path = os.path.join(images_dir, f"{patient_id}")  # New filename

#             if os.path.exists(f"{old_file_path}.jpg"):
#                 os.rename(f"{old_file_path}.jpg", f"{new_file_path}.jpg")
#                 print(f"Renamed {old_filename} to {patient_id}.jpg")
#             else:
#                 print(f"File {old_filename} not found in {images_dir}")
#         else:
#             print(f"No image found for patientID {patient_id}")

#     except KeyError as e:
#         print(f"KeyError: {e} - Check column names in your CSV file.")

dataset = r'C:\Users\rohan\Downloads\YICCHALLENGE\Dataset'

images = []
labels = []
identifiers_list = []

def image_processing(file_directory, images_list, labels_list, identifiers_list):
    for classification in os.listdir(file_directory):
        classification_lbl = os.path.join(file_directory, classification)
        for image_file in os.listdir(classification_lbl):
            image_path = os.path.join(classification_lbl, image_file)
            MRI_img = cv2.imread(image_path)
            MRI_img = cv2.cvtColor(MRI_img, cv2.COLOR_BGR2GRAY)
            MRI_img = MRI_img / 255.0
            images_list.append(MRI_img)
            labels_list.append(classification)
            identifiers_list.append(image_file.split('.')[0])

image_processing(dataset, images, labels, identifiers_list)


MRI_images = np.array(images)
MRI_classifications = np.array(labels)


def map_to_alzheimers(label):
    if label in ['Mild_Demented', 'Moderate_Demented', 'Very_Mild_Demented']:
        return 1 
    else:
        return 0  


one_hot_encoded = np.array([map_to_alzheimers(label) for label in MRI_classifications])

for i in range(10):
    print(f"Image {i+1}:")
    print(f"Original Label: {MRI_classifications[i]}")
    print(f"One-Hot Encoded Label: {one_hot_encoded[i]}")
    print("-" * 40)

print(MRI_images.shape)

print(np.unique(labels))

print(X_train_clinical.shape)

# Plot the count of each unique value in labels
sns.countplot(x=labels)
plt.show()
# Display the count of each unique value in labels using pandas
print(pd.Series(labels).value_counts())

label_encoder = LabelEncoder()
one_hot_encoded = label_encoder.fit_transform(MRI_classifications)
print(one_hot_encoded)

# Flatten images for RandomForest
n_samples = MRI_images.shape[0]
height = MRI_images.shape[1]
width = MRI_images.shape[2]

# Flatten each image to a 1D vector
MRI_images = MRI_images.reshape(n_samples, height * width)

X_train, X_test, y_train, y_test = train_test_split(MRI_images, one_hot_encoded, test_size=0.2, random_state=42)
      
model = Sequential()

model.add(Conv2D(32,(3,3), activation = 'relu', input_shape=(128,128,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3),  activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(128,(3,3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(256,(3,3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
 
model.add(Flatten())
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.3))

model.add(Dense(4, activation = 'softmax'))

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train.reshape(-1, 128, 128, 1),
    y_train,   
    validation_data=(X_test.reshape(-1, 128, 128, 1), y_test),
    epochs=1,
)

# Create DataFrames
df_images = pd.DataFrame({'Image': MRI_images.tolist(), 'Identifier': identifiers_list})
df_labels = pd.DataFrame({'Label': labels, 'Identifier': identifiers_list})

# Assuming X_clinical DataFrame exists and has a PatientID column
df_clinical = pd.DataFrame(X_clinical)
df_clinical['Identifier'] = df_clinical['PatientID'].astype(str)  # Ensure 'Identifier' is of type str

# Convert Identifier columns to strings for consistency
df_images['Identifier'] = df_images['Identifier'].astype(str)
df_labels['Identifier'] = df_labels['Identifier'].astype(str)
df_clinical['Identifier'] = df_clinical['Identifier'].astype(str)

# Merge labels and clinical data
df_labels_clinical = pd.merge(df_labels, df_clinical, on='Identifier')
print(df_labels_clinical.shape)

df_combined_filtered = pd.merge(df_images, df_labels_clinical, on='Identifier')

print(df_labels['Identifier'].head())
print(df_clinical['Identifier'].head())

print(df_labels.head())
print(df_clinical.head())

print(df_combined_filtered.shape)

# Assuming you want to split based on the combined DataFrame
X_train_df, X_test_df = train_test_split(df_combined_filtered, test_size=0.2, random_state=42)

# Extract features and labels
X_train_images = np.array(X_train_df['Image'].tolist())
y_train = X_train_df['Label']
X_train_clinical = X_train_df.drop(columns=['Image', 'Label', 'Identifier'])

print(X_train_clinical.shape)
print(X_train_images.shape)

print(f"Unique Identifiers in df_images: {df_images['Identifier'].nunique()}")
print(f"Unique Identifiers in df_labels_clinical: {df_labels_clinical['Identifier'].nunique()}")

print(df_combined_filtered['Identifier'].nunique())
print(len(X_train_clinical))
print(len(X_train_images))


X_test_images = np.array(X_test_df['Image'].tolist())
y_test = X_test_df['Label']
X_test_clinical = X_test_df.drop(columns=['Image', 'Label', 'Identifier'])

# Assuming Random Forest model is already trained and CNN model is already trained
# Generate predictions for the training data
rf_train_pred_proba = rf_model.predict_proba(X_train_clinical)
rf_test_pred_proba = rf_model.predict_proba(X_test_clinical)

# Assuming CNN model is trained and images are preprocessed
cnn_train_pred_proba = model.predict(X_train_images.reshape(-1, 128, 128, 1))
cnn_test_pred_proba = model.predict(X_test_images.reshape(-1, 128, 128, 1))

print(rf_train_pred_proba.shape)
print(cnn_train_pred_proba.shape)
print(X_train_clinical.shape)

train_meta_features = np.hstack((rf_train_pred_proba, cnn_train_pred_proba))
test_meta_features = np.hstack((rf_test_pred_proba, cnn_test_pred_proba))

# Ensure y_train and y_test are formatted correctly for the meta-model
y_train_meta = y_train
y_test_meta = y_test

# Train a meta-model (Logistic Regression)
meta_model = LogisticRegression(random_state=42)
meta_model.fit(train_meta_features, y_train_meta)

# Evaluate the meta-model on the test data
meta_test_pred = meta_model.predict(test_meta_features)
meta_accuracy = accuracy_score(y_test_meta, meta_test_pred)
meta_report = classification_report(y_test_meta, meta_test_pred)

print(f"Meta-Model Accuracy: {meta_accuracy}")
print(meta_report)