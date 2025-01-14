from medmnist import BreastMNIST, BloodMNIST
import numpy as np
import A.binary as A
import B.multiple as B
import B.CNN as C
import tensorflow as tf


#Task A


# Load the dataset for task A
training_data = BreastMNIST(split='train', download=True,size=28)
validation_data = BreastMNIST(split='val', download=True,size=28)
test_data = BreastMNIST(split='test', download=True,size=28)
    

# Get the data from the dataset objects for task A
X_train, y_train = A.get_data_A(training_data)
X_val, y_val = A.get_data_A(validation_data)
X_combined = np.vstack((X_train, X_val))
y_combined = np.concatenate((y_train, y_val))
X_test, y_test = A.get_data_A(test_data)

# Task A: Logistic Regression
print("\nTask A: Logistic Regression\n")
A.TaskA_LR(X_train, y_train, X_val, y_val, X_test, y_test, X_combined, y_combined)

# Task A: SVM
print("\nTask A: SVM\n")
A.TaskA_SVM(X_train, y_train, X_val, y_val, X_test, y_test, X_combined, y_combined)

# Task A: Neural Network
print("\nTask A: Neural Network\n")
A.TaskA_nn(X_train, y_train, X_val, y_val, X_test, y_test, X_combined, y_combined)





#Task B


# Load the dataset for task B
# Load the dataset 
training_data = BloodMNIST(split='train', download=True,size=28)
validation_data = BloodMNIST(split='val', download=True,size=28)
test_data = BloodMNIST(split='test', download=True,size=28)

# Get the data from the dataset objects
X_train, y_train = B.get_data_B(training_data)
X_val, y_val = B.get_data_B(validation_data)
X_combined = np.vstack((X_train, X_val))
y_combined = np.concatenate((y_train, y_val))
X_test, y_test = B.get_data_B(test_data)

# Task B: Random Forest
print("\nTask B: Random Forest\n")
B.TaskB_RF(X_train, y_train, X_val, y_val, X_test, y_test, X_combined, y_combined)

# Task B: Neural Network
print("\nTask B: Neural Network\n")
B.TaskB_nn(X_train, y_train, X_val, y_val, X_test, y_test, X_combined, y_combined)



#Task B：CNN
# obtain the dataset
training_data = BloodMNIST(split='train', download=True, as_rgb=False)
validation_data = BloodMNIST(split='val', download=True, as_rgb=False)
test_data = BloodMNIST(split='test', download=True, as_rgb=False)

# obtain the data
X_train, y_train = C.get_data_C(training_data)
X_val, y_val = C.get_data_C(validation_data)
X_test, y_test = C.get_data_C(test_data)

# label convert into one-hot encoded vector
num_classes = len(np.unique(y_train))
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

print("\nTask B: CNN\n")
C.TaskB_cnn(num_classes,X_train, y_train, X_val, y_val, X_test, y_test)






