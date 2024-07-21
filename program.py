import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, LeakyReLU, GRU


train_data = pd.read_csv('c:/Users/Aniketh/PycharmProjects/programfolder2/MI_Codes/Hackathon/train.csv')

fp = open('c:/Users/Aniketh/PycharmProjects/programfolder2/MI_Codes/Hackathon/check.txt', 'a')

X = train_data.drop(columns=['filename', 'label']) 
y = train_data['label']


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size = 0.08)#, random_state = 100)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


ann = Sequential()

f1 = 'relu'
f2 = 'relu'

ann.add(Dense(units=512, activation = f1, input_dim=X_train_scaled.shape[1]))
ann.add(Dense(units=256, activation = f2))
ann.add(Dense(units=len(label_encoder.classes_), activation='softmax'))

ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

ann.fit(X_train_scaled, y_train, epochs = 156, batch_size = 28, validation_data=(X_val_scaled, y_val))

test_data = pd.read_csv('c:/Users/Aniketh/PycharmProjects/programfolder2/MI_Codes/Hackathon/test.csv')

test_data_features = scaler.transform(test_data.drop(columns=['id']))

test_data_features_reshaped = test_data_features.reshape(test_data_features.shape[0], test_data_features.shape[1], 1)

ann_predictions = ann.predict(test_data_features)

ann_predicted_labels = ann_predictions.argmax(axis=1)

ann_submission_df = pd.DataFrame({'id': test_data['id'], 'label': ann_predicted_labels})

ann_submission_df.to_csv('c:/Users/Aniketh/PycharmProjects/programfolder2/MI_Codes/Hackathon/music_genre_ann_predictions.csv', index=False)


y_true = pd.read_csv('c:/Users/Aniketh/PycharmProjects/programfolder2/MI_Codes/Hackathon/ground_truth.csv')


y_true = y_true.drop(columns=['Usage', 'id'])
y_true = y_true.drop(y_true.index[0])



y2 = pd.DataFrame(list(zip(test_data['id'], ann_predicted_labels)))


y2 = y2.drop(columns = [0])
y2 = y2.drop(y2.index[0])


accuracy = accuracy_score(y_true, y2)
print(f'Accuracy: {accuracy * 100:.2f}%')

fp.write(f'Accuracy with {f1} and {f2}: {accuracy * 100:.2f}%\n')