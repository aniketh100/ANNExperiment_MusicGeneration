import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.layers import ELU, PReLU, LeakyReLU
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load the training dataset
train_data = pd.read_csv('c:/Users/Aniketh/PycharmProjects/programfolder2/MI_Codes/Hackathon/train.csv')

# Extract features and labels
X = train_data.drop(columns=['filename', 'label'])  # Exclude filename and label from features
y = train_data['label']

# Perform one-hot encoding for labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.08, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Reshape data for LSTM input (assuming sequential nature of data)
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_val_reshaped = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1], 1)

# Build the LSTM RNN model
rnn_model = Sequential()
rnn_model.add(LSTM(units=512, activation='tanh', input_shape=(X_train_reshaped.shape[1], 1), recurrent_activation='tanh'))
rnn_model.add(Dense(units=256, activation=LeakyReLU(alpha=0.11)))
rnn_model.add(Dense(units=len(label_encoder.classes_), activation='softmax'))

# Compile the RNN model
rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the RNN model
rnn_model.fit(X_train_reshaped, y_train, epochs=80, batch_size=28, validation_data=(X_val_reshaped, y_val))

# Load and preprocess the test data
test_data = pd.read_csv('c:/Users/Aniketh/PycharmProjects/programfolder2/MI_Codes/Hackathon/test.csv')
test_data_features = scaler.transform(test_data.drop(columns=['id']))
test_data_features_reshaped = test_data_features.reshape(test_data_features.shape[0], test_data_features.shape[1], 1)

# Make predictions using the RNN model
rnn_predictions = rnn_model.predict(test_data_features_reshaped)
rnn_predicted_labels = rnn_predictions.argmax(axis=1)

# Create submission DataFrame
rnn_submission_df = pd.DataFrame({'id': test_data['id'], 'label': rnn_predicted_labels})

# Save predictions to a CSV file
rnn_submission_df.to_csv('c:/Users/Aniketh/PycharmProjects/programfolder2/MI_Codes/Hackathon/music_genre_rnn_predictions.csv', index=False)

# Load true labels for the test set (without headers)
y_true = pd.read_csv('c:/Users/Aniketh/PycharmProjects/programfolder2/MI_Codes/Hackathon/ground_truth.csv', header=None, names=['id', 'label'])

# Calculate accuracy
accuracy = accuracy_score(y_true['label'], rnn_predicted_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')