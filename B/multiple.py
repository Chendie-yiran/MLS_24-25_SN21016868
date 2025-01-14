from medmnist import BloodMNIST
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay,precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


def get_data_B(data):

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

    # Flatten X to be 2D
    X = X.reshape(X.shape[0], -1)

    return X, y


def get_result_B(pred,actual, name):

    accuracy = accuracy_score(actual, pred)
    confusion = confusion_matrix(actual, pred)
    precision = precision_score(actual, pred, average='weighted')  # Weighted for multi-class
    recall = recall_score(actual, pred, average='weighted')
    print(name," Accuracy: ", accuracy)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(name," Confusion Matrix: \n", confusion)


def TaskB_RF(X_train, y_train, X_val, y_val, X_test, y_test, X_combined, y_combined):
    
    rf = RandomForestClassifier(n_estimators=100,  
                                random_state=42,
                                n_jobs=-1,
                                max_depth=15,
                                )  
        
    #cv_scores = cross_val_score(rf, X_combined, y_combined, cv=10)
    #print(f"Cross-validation accuracy: {cv_scores.mean():.6f} ± {cv_scores.std():.4f}")

    rf.fit(X_train, y_train)
    y_train_pred = rf.predict(X_train)
    y_val_pred = rf.predict(X_val)
    y_test_pred = rf.predict(X_test)

    get_result_B(y_train_pred, y_train, "Training")
    get_result_B(y_val_pred, y_val, "Validation")
    get_result_B(y_test_pred, y_test, "Testing")


def TaskB_nn(X_train, y_train, X_val, y_val, X_test, y_test, X_combined, y_combined):

    #neural networ
    MLP = MLPClassifier(hidden_layer_sizes=(128, 64),  
        activation='relu',             
        solver='adam',                 
        learning_rate_init=0.001,      
        alpha=0.01,      
        learning_rate='adaptive',            
        max_iter=5000,                 
        random_state=42,               
        )	
    
    #scores = cross_val_score(MLP, X_combined, y_combined, cv=3)
    #print("Neural network Average cross-validation accuracy: ", np.mean(scores))

    # Fit the MLP model on the training data
    MLP.fit(X_train, y_train)

    # Get predicted probabilities for training, validation, and testing sets
    y_train_pred = MLP.predict(X_train) # Probabilities for the positive class
    y_val_pred = MLP.predict(X_val)
    y_test_pred = MLP.predict(X_test)

    # Get and print the result for training, validation, and testing sets
    get_result_B(y_train_pred, y_train, "Training")
    get_result_B(y_val_pred, y_val, "Validation")
    get_result_B(y_test_pred, y_test, "Testing")

