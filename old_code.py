import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, LeakyReLU

# Load the training dataset
train_data = pd.read_csv('c:/Users/Aniketh/PycharmProjects/programfolder2/MI_Codes/Hackathon/train.csv')

# Extract features and labels
X = train_data.drop(columns=['filename', 'label'])  # Exclude filename and label from features
y = train_data['label']

# Perform one-hot encoding for labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size = 0.05)#, random_state = 42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Build the ANN model
ann_model = Sequential()
ann_model.add(Dense(units=512, activation = 'tanh', input_dim=X_train_scaled.shape[1]))
ann_model.add(Dense(units=200, activation = LeakyReLU(alpha = 0.02)))
#ann_model.add(Dense(units=100, activation = 'softmax'))
ann_model.add(Dense(units=len(label_encoder.classes_), activation='softplus'))
#ann_model.add(Dense(units=len(label_encoder.classes_), activation='softplus'))


# Compile the ANN model
ann_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the ANN model
ann_model.fit(X_train_scaled, y_train, epochs = 100, batch_size = 27, validation_data=(X_val_scaled, y_val))

test_data = pd.read_csv('c:/Users/Aniketh/PycharmProjects/programfolder2/MI_Codes/Hackathon/test.csv')

test_data_features = scaler.transform(test_data.drop(columns=['id']))

test_data_features_reshaped = test_data_features.reshape(test_data_features.shape[0], test_data_features.shape[1], 1)

ann_predictions = ann_model.predict(test_data_features)

ann_predicted_labels = ann_predictions.argmax(axis=1)

ann_submission_df = pd.DataFrame({'id': test_data['id'], 'label': ann_predicted_labels})

ann_submission_df.to_csv('c:/Users/Aniketh/PycharmProjects/programfolder2/MI_Codes/Hackathon/music_genre_ann_predictions.csv', index=False)

# Load the true labels for the test set (without headers)
y_true = pd.read_csv('c:/Users/Aniketh/PycharmProjects/programfolder2/MI_Codes/Hackathon/ground_truth.csv')

# Drop the 'Usage' column and the first row
y_true = y_true.drop(columns=['Usage', 'id'])
y_true = y_true.drop(y_true.index[0])
#print(y_true)

# Create DataFrame y2 without headers
y2 = pd.DataFrame(list(zip(test_data['id'], ann_predicted_labels)))

# Drop the first row from y2
y2 = y2.drop(columns = [0])
y2 = y2.drop(y2.index[0])
#print(y2)

# Calculate accuracy
accuracy = accuracy_score(y_true, y2)
print(f'Accuracy: {accuracy * 100:.2f}%')

import matplotlib.pyplot as plt
# Plotting the distribution of genres
plt.figure(figsize=(10, 6))
train_data['label'].value_counts().plot(kind='bar')
plt.title('Genre Distribution in Training Data')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.show()