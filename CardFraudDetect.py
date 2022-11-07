
from google.colab import drive
drive.mount('/content/drive',force_remount=True)

"""## Read Data """

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm 

# get dataset path
dataset_path = '.../creditcard.csv'
## Read .csv file
# Convert label value from string to float
convert_label = lambda x: float(x.strip('"') or -1)
# Read data
dataset = np.genfromtxt(dataset_path,
                        delimiter=",",
                        skip_header=1,
                        converters={-1: convert_label},
                        dtype=None,
                        encoding=None)

n_samples = dataset.shape[0] #number of samples
n_classes = len(np.unique(dataset[:, -1])) #number of classes to be classified
n_features = dataset.shape[1] - 1 #number of features
print('Number of samples: ', n_samples)
print('Number of features: ', n_features)
print('Number of classes: ', n_classes)

"""## Shuffle, Extract, add bias, one-hot encoding label, normalize data """

# Shuffle data
np.random.seed(1)
dataset = np.random.permutation(dataset)
# Extract X and y
X, y_idx = dataset[:, :-1].astype(np.float64), dataset[:, -1].astype(np.int32)
# One-hot encoding label
y = np.array([np.zeros(n_classes) for _ in range(y_idx.shape[0])])
y[np.arange(len(y_idx)), y_idx] = 1
# Min-max normalization
X_min = np.min(X, axis=0)
X_max = np.max(X, axis=0)
X = (X - X_min) / (X_max - X_min)
# Add bias
bias = np.ones((X.shape[0], 1))
X = np.hstack((bias, X))

"""## Split train and validate set"""

# train, validate, test set ratio
TRAIN_SIZE = 0.7
VAL_SIZE = 0.2

TRAIN_IDX_END = int(TRAIN_SIZE * dataset.shape[0])
VAL_IDX_END = int(TRAIN_IDX_END + (VAL_SIZE * dataset.shape[0]))

#split train,validate and test set
X_train, y_train = X[:TRAIN_IDX_END], y[:TRAIN_IDX_END]
X_val, y_val = X[TRAIN_IDX_END:VAL_IDX_END], y[TRAIN_IDX_END:VAL_IDX_END]
X_test, y_test = X[VAL_IDX_END:], y[VAL_IDX_END:]

"""# Building model"""

#softmax function
def softmax(z):
  expZ=np.exp(z)
  return expZ/expZ.sum(axis=0)

## predict function
def predict(X, theta):
  z=theta.T.dot(X)
  y_hat=softmax(z)
  return y_hat

#compute loss function
def loss(y, y_hat):
  l=-y.T.dot(np.log(y_hat))
  return l

#gradient descent function
def gradient(X, y, y_hat):
  grad=X.dot((y_hat-y).T)
  return grad

#evaluate function
def evaluate(X_test, y_test, theta):
  accs = []
  losses = []
  n_samples = X_test.shape[0]
  n_features = X_test.shape[1]
  n_classes = np.unique(y_test, axis=0).shape[0]

  for i in range(n_samples):
    Xi=X_test[i].reshape((n_features,1))
    Yi=y_test[i].reshape((n_classes,1))
    y_pred=predict(Xi,theta)
    l=loss(Yi,y_pred)[0][0]
    losses.append(l)
    acc=(np.argmax(Yi)==np.argmax(y_pred))
    accs.append(acc)
  
  return sum(losses) / len(losses), sum(accs) / len(accs)

#training function
def fit(X_train, y_train, theta, EPOCHS=10, LR=1e-4, is_visualize=False):
  train_losses = []
  n_samples = X_train.shape[0]
  n_features = X_train.shape[1]
  n_classes = np.unique(y_train, axis=0).shape[0]
  for epoch in range(EPOCHS):
    progress_bar = tqdm(range(n_samples), desc=f"EPOCH {epoch}", position=0) # Tạo thanh progress thể hiện tiến độ training
    for i in progress_bar:
      #take from train set independent and dependent value
      X_i = X_train[i]
      y_i = y_train[i]

      X_i = X_i.reshape((n_features, 1))
      y_i = y_i.reshape((n_classes, 1))

      #predict output
      y_hat = predict(X_i, theta)

      #calculate loss and training accuracy
      train_loss = loss(y_i, y_hat)[0][0]
      train_losses.append(train_loss)

      #compute gradient
      grad = gradient(X_i, y_i, y_hat)

      #update theta
      theta = theta - LR * grad

      progress_bar.set_postfix({"Train Loss": train_loss}) # Thêm thông tin train loss vừa tính được vào thanh progress

  print("\nTRAINING COMPLETE")

  if is_visualize:
    plt.plot(train_losses, color='green')
    plt.title("Train loss over batch")
    plt.show()

  return theta

"""# Train model"""

#Start with random theta
np.random.seed(1)
theta = np.random.uniform(size=(X_train.shape[1], np.unique(y_train, axis=0).shape[0]))

#define hyper parameters
EPOCHS = 1
LR = 1e-3
theta = fit(X_train, y_train, theta, EPOCHS, LR, is_visualize=True)

"""# Evaluate model"""

#Evaluate model on validate set
val_loss, val_acc = evaluate(X_val, y_val, theta)
print("Validation loss: ", np.round(val_loss, 3))
print("Validation accuracy: ", np.round(val_acc, 3))

#Evaluate model on test set
test_loss, test_acc = evaluate(X_test, y_test, theta)
print("Test loss: ", np.round(test_loss, 3))
print("Test accuracy: ", np.round(test_acc, 3))
