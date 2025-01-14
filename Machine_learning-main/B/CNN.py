import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from medmnist import BloodMNIST

# pre-processing
def get_data_C(data):

    X = []
    y = []

    for i in range(len(data)):
        x, label = data[i]
        x = np.array(x)
        X.append(x)
        y.append(label[0])

    # Convert X and y to numpy arrays before returning
    X = np.array(X)
    y = np.array(y)

    # Normalize X to range [0, 1]
    X = X / 255.0

    return X, y

# results obtained
def get_result_C(pred, actual, name):

    accuracy = accuracy_score(actual, pred)
    confusion = confusion_matrix(actual, pred)
    precision = precision_score(actual, pred, average='weighted')  # Weighted for multi-class
    recall = recall_score(actual, pred, average='weighted')
    print(name, " Accuracy: ", accuracy)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(name, " Confusion Matrix: \n", confusion)


# CNN model
def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        # first layer
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),

        # second layer
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # fully connected layer
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # output layer
    ])
    return model

# struct the model

def TaskB_cnn(num_classes,X_train, y_train, X_val, y_val, X_test, y_test):
    input_shape = (28, 28, 3)
    model = build_cnn_model(input_shape, num_classes)


    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # train the model
    history=model.fit(X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=15,
            batch_size=16)
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    print(f"Final Training Accuracy: {train_accuracy[-1]:.4f}")
    print(f"Final Validation Accuracy: {val_accuracy[-1]:.4f}")

    # analysis the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # predict
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # evaluate the result
    get_result_C(y_pred_classes, y_test_classes, "CNN Model")
