"""
Implementation of a multilayer neural net, 
to classify handwritten digits from the MNIST dataset
"""


#%% fetch our dataset via sklearn's fetch_openml,
from sklearn.datasets import fetch_openml


X,y = fetch_openml('mnist_784', version=1, return_X_y=True)

#data is returned as pandas DataFrame and series object, so .values returns the underlying numpy array
X= X.values
y= y.astype(int).values

#X will be of shape (70000,784) -> i.e 70000 images, with 784 pixels each
print(f'The shape of X is {X.shape}')

#y will be of shape 70000 -> each of the value 0-9, representing the class label for each image
print(f'The shape of y is {y.shape}')


# %% The pixels are originally in a range of (0 to 255) i.e. a 8-bit unsigned integer,
#pixel value directly indicates brightness i.e.  0 is a completely black pixel whereas 255 is completely white.
#We now normalize as gradient based optimization is more stable under this case. 

#normalize from [0,255] -> [-1,1], 
X= ((X/255.) -.5)*2


#%% Visualise the images 
import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)

ax = ax.flatten()
for i in range(10):
    img = X[y == i][0].reshape(28,28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


# %% Plot examples of the same digit 
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax= ax.flatten()

for i in range(25): 
    img = X[y==7][i].reshape(28,28)
    ax[i].imshow(img, cmap='Grays')
    
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout
plt.show()

# %% Divide dataset into training, validation and test subsets
from sklearn.model_selection import train_test_split

#'stratify=y' -> ensures proporition of each class in y is preserved in the train/test split
X_temp, X_test, y_temp, y_test = train_test_split(X,y, test_size=10000, random_state=123, stratify=y)

X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=5000, random_state=123, stratify=y_temp)

#%% Define helper functins 
import numpy as np

#define our activation function , sigmoid activation-> maps numbers in R to [-1,1]
def sigmoid(z): 
    return 1./ (1. + np.exp(-z))


#
def int_to_onehot(y, num_labels): 
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i,val]=1
    
    return ary

# %%
