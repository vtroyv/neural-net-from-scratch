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
print(f'the y.shape[0] is {y.shape[0]}')

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

# %% NeuralNetMLP class

class NeuralNetMLP:
    #instantiate the weight matrices and bias vectors for hte hidden and output layer.
    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        super().__init__()
    
        self.num_classes = num_classes
    
        #hidden
        rng = np.random.RandomState(random_seed)
    
        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)
    
        #output
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)
        
    
    def forward(self, x):
        #Hidden layer
        
        #input dim: [n_examples, n_features]
        #           dot [n_hidden, n_features].T
        # output dim: [n_examlpes, n_hidden]
        
        z_h = np.dot(x, self.weight_h.T)+ self.bias_h
        a_h = sigmoid(z_h)
        
        #Output layer
        #input dim: [n_examples, n_hidden]
        #           dot [n_classes_n_iddent].T
        #output dim:    [n_examples, n_classes]
        
        z_out= np.dot(a_h, self.weight_out.T) + self.bias_out
        
        #a_out represents the class memership probabilities that we can convert to class labels, we also need
        #a_h (activation values from hidden layer) to optimize the model paramters
        a_out =  sigmoid(z_out)
        return a_h, a_out
    
    
    def backward(self, x, a_h, a_out, y):
        #######################
        ###Output layer weights
        #######################
        
        #one-hot encoding
        y_onehot = int_to_onehot(y, self.num_classes)
        
        #part 1: dLoss/dOutWeights
        ## =dLoss/dOutAct * dOutAct/dOutNet * dOutNet/dOutWeight
        ## where deltaOut = dLoss/dOutAct * dOutAct/dOutNet
        ## for convenient re-use
        
        #input/output dim: [n_examples, n_classes] -> i.e our loss is the MSE, so this is MSE derivative 
        d_loss__d_a_out =2.*(a_out - y_onehot)/y.shape[0]
        
        #input/output dim: [n_examples, n_classes]
        d_a_out__d_z_out =  a_out * (1. -a_out) # sigmoid derivative 
        
        #output dim: [n_examples, n_classes]
        delta_out =  d_loss__d_a_out * d_a_out__d_z_out
        
        # gradient for output weights
        
        #[n_examples, n_hidden]
        d_z_out__dw_out = a_h
        
        #input dim: [n_classes, n_examples] 
        #              dot [n_examples, n_hidden]
        #output dim: [n_classes, n_hidden]
        
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)
        
        ##################################
        # Part 2: dLoss/dHiddenWeights
        ## = DeltaOut * dOutNet/dHiddenAct * dHiddenAct/dHiddenNet
        #       * dHiddenNet/dWeight
        
        #[n_classes, n_hidden]
        d_z_out__a_h = self.weight_out 
        
        #output dim: [n_examples, n_hidden]
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)
        
        # [n_examples, n_hidden]
        d_a_h__d_z_h = a_h * (1. -a_h) #sigmoid derivative
        
        # [n_examples, n_hidden]
        d_z_h__d_w_h = x
        
        #output dim: [n_hidden, n_features]
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)
        
        return (d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h)
    
    
        
        
        #the backward method implements the backpropagation algorithm, which calculates gradients of the loss with respect
        #to the weight and bias parameters. Similar to adaline, these gradients are then used to updates these paramters via gradient descent.
        
#%%
model = NeuralNetMLP(num_features=28*28, num_hidden=50, num_classes=10)        
# %%
#This cell defines a mini-batch generator, it takes in our dataset and divides it into mini-batches of a desired size
#for stochastic gradient descent training.

num_epochs = 50
minibatch_size=100

def minibatch_generator(X,y, minibatch_size):
    indices= np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, indices.shape[0] - minibatch_size +1, minibatch_size):
        batch_idx = indices[start_idx: start_idx + minibatch_size]
        yield X[batch_idx], y[batch_idx]
        
#Check the minibatch generator works as intended
for i in range(num_epochs):
    #iterate over minibatches
    minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)
    
    for X_train_mini, y_train_mini in minibatch_gen:
        break
    break

print(X_train_mini.shape)
print(y_train_mini.shape)
        
#%% Next we define our loss function and performance metric
def mse_loss(targets, probas, num_labels=10):
    onehot_targets = int_to_onehot(targets, num_labels=num_labels)
    return np.mean((onehot_targets-probas)**2)

def accuracy(targets, predicted_labels): 
    return np.mean(predicted_labels == targets)

#Now we test these functions by computing the initial validation set MSE and accuracy of the model 
#we instantiated in the previous cell

_, probas = model.forward(X_valid)
mse = mse_loss(y_valid, probas)

print(f'Initial validation MSE: {mse: .1f}')

predicted_labels = np.argmax(probas, axis=1)
acc = accuracy(y_valid, predicted_labels)

print(f'Initial validation accuracy: {acc*100:.1f}%')

# %% Compute MSE accuracy incrementally by iterating over the dataset one mini-batch at a time for memory efficieny

def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    mse, correct_pred, num_examples = 0., 0,0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
    for i, (features, targets) in enumerate(minibatch_gen):
        _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)
        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        loss = np.mean((onehot_targets - probas) **2)
        correct_pred += (predicted_labels == targets).sum()
        num_examples += targets.shape[0]
        mse += loss
        
    mse = mse/i
    acc = correct_pred/num_examples
    return mse, acc


        
        

# %% Once more before implmenting the training loop, let's test the function and compute initial training set MSE
# and accuracy of the mdoel we instantiated in the previous section 

mse, acc = compute_mse_and_acc(model, X_valid, y_valid)
print(f'Initial valid MSE: {mse:.1f}')
print(f'Initial valid accuracy: {acc*100:.1f}%')



# %% code to train model 

def train(model, X_train, y_train, X_valid, num_epochs, learning_rate=0.1):
    epoch_loss=[]
    epoch_train_acc=[]
    epoch_valid_acc=[]
    

    for e in range(num_epochs):
        #iterate over minibatches
        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)
        for X_train_mini, y_train_mini in minibatch_gen:
            ### COmute outputs ###
            a_h, a_out =  model.forward(X_train_mini)
        
            ### Compute gradients###
            d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h = model.backward(X_train_mini, a_h, a_out, y_train_mini)
        
            ###Update weights ###
            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out -= learning_rate * d_loss__d_b_out
        
        
            #### Epoch logging #####
            train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train)
        
            valid_mse, valid_acc= compute_mse_and_acc(model, X_valid, y_valid)
        
            train_acc, valid_acc = train_acc*100, valid_acc*100
            epoch_train_acc.append(train_acc)
            epoch_valid_acc.append(valid_acc)
            epoch_loss.append(train_mse)
        
        print(f'Epoch: {e+1:03d}/{num_epochs:03d} '
                f'| Train MSE: {train_mse:.2f} '
                f'| Train Acc: {train_acc:.2f}% '
                f'| Valid Acc: {valid_acc:.2f}%')
            
    return epoch_loss, epoch_train_acc, epoch_valid_acc
        
        
# %% Execute and train model for 50 epochs
np.random.seed(123) # for the training set shuffling
epoch_loss, epoch_train_acc, epoch_valid_acc = train(model, X_train, y_train, X_valid, num_epochs=50, learning_rate=0.1)

# %%
