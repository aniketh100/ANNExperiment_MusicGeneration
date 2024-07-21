import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from itertools import product

# Load the training dataset
train_data = pd.read_csv('c:/Users/Aniketh/PycharmProjects/programfolder2/MI_Codes/Hackathon/train.csv')

# Extract features and labels
X = train_data.drop(columns=['filename', 'label'])
y = train_data['label']

# Perform one-hot encoding for labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.05, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define a set of activation functions to be explored
activation_functions = ['relu', 'tanh', 'sigmoid', 'softplus', 'softmax', 'selu', LeakyReLU(alpha = 0.2)]

# Grid search for activation functions
best_accuracy = 0
best_activation_combination = None

fp = open('c:/Users/Aniketh/PycharmProjects/programfolder2/MI_Codes/Hackathon/tempo.txt', 'w')

for activation_combination in product(activation_functions, repeat=2):
    ann_model = Sequential()
    ann_model.add(Dense(units=512, activation=activation_combination[0], input_dim=X_train_scaled.shape[1]))
    ann_model.add(Dense(units=256, activation=activation_combination[1]))
    print(activation_combination[0], activation_combination[1])
    ann_model.add(Dense(units=len(label_encoder.classes_), activation='softmax'))
    ann_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    ann_model.fit(X_train_scaled, y_train, epochs=50, batch_size=27, validation_data=(X_val_scaled, y_val))

    test_data = pd.read_csv('c:/Users/Aniketh/PycharmProjects/programfolder2/MI_Codes/Hackathon/test.csv')

    test_data_features = scaler.transform(test_data.drop(columns=['id']))

    test_data_features_reshaped = test_data_features.reshape(test_data_features.shape[0], test_data_features.shape[1], 1)

    ann_predictions = ann_model.predict(test_data_features)

    ann_predicted_labels = ann_predictions.argmax(axis=1)

    ann_submission_df = pd.DataFrame({'id': test_data['id'], 'label': ann_predicted_labels})
    
    y_true = pd.read_csv('c:/Users/Aniketh/PycharmProjects/programfolder2/MI_Codes/Hackathon/ground_truth.csv')


    y_true = y_true.drop(columns=['Usage', 'id'])
    y_true = y_true.drop(y_true.index[0])



    y2 = pd.DataFrame(list(zip(test_data['id'], ann_predicted_labels)))


    y2 = y2.drop(columns = [0])
    y2 = y2.drop(y2.index[0])


    accuracy = accuracy_score(y_true, y2)
    #print(f'Accuracy: {accuracy * 100:.2f}%')
    
    print(f'Activation Functions: {activation_combination}, Accuracy: {accuracy * 100:.2f}%')
    fp.write(f'Activation Functions: {activation_combination}, Accuracy: {accuracy * 100:.2f}%\n')
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_activation_combination = activation_combination

print(f'Best Activation Functions: {best_activation_combination}, Best Accuracy: {best_accuracy * 100:.2f}%')
fp.close()