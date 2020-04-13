import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

### Displaying the image###
index = 150
# plt.imshow(train_set_x_orig[index])
# plt.show()

# Printing it's label
print("y =", train_set_y[:, index], "it's a ",
      classes[np.squeeze(train_set_y[:, index])].decode("utf-8"), "picture")

m_train = len(train_set_x_orig)
m_test = len(test_set_x_orig)
num_px = len(train_set_x_orig[0])

print("Number of training examples: m_train = " + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Height/Width of each image: num_px = " + str(num_px))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_set_x shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y.shape))


# Reshape the training and test examples

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


def sigmoid(z):
    s = 1 / (np.exp(-z)+1)
    return s


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[0]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1/m * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
    dw = 1/m * np.dot(X, (A - Y).T)  # X * dz (a - Y)
    db = 1/m * np.sum(A - Y)  # dz (a - Y)
    grads = {"dw": dw, "db": db}
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    # will be used to plot the learning curve
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w -= (learning_rate * dw)
        b -= (learning_rate * db)
        costs.append(cost)

        # print costs if the user wants
        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i} = {cost}")

    grads = {"dw": dw, "db": db}
    params = {"w": w, "b": b}
    return params, grads, costs
