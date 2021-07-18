#!/usr/bin/env python
# coding: utf-8

# # CAP 6619 - Deep Learning
# ## Summer 2021 - Dr Marques
# ## Assignment 1 Asam Mahmood
# ## Handwritten Digit Classifier Using the MNIST Dataset

# Useful references and sources:
# 
# - https://www.tensorflow.org/datasets/catalog/mnist
# 
# - https://en.wikipedia.org/wiki/MNIST_database 
# 
# - https://keras.io/examples/vision/mnist_convnet/
# 
# - https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/shallow_net_in_keras.ipynb 

# ### (OPTIONAL) TODO 1
# 
# Add your own sources and references here.

# Below you can see some libs i had to pull in to help build and run code aand help time performance on execution
# 
# absl-py==0.13.0
# appdirs==1.4.4
# argon2-cffi==20.1.0
# as==0.1
# astunparse==1.6.3
# async-generator==1.10
# attrs==21.2.0
# backcall==0.2.0
# bleach==3.3.0
# blis==0.7.4
# cachetools==4.2.2
# catalogue==2.0.4
# certifi==2021.5.30
# cffi==1.14.5
# chardet==4.0.0
# click==7.1.2
# colorama==0.4.4
# cycler==0.10.0
# cymem==2.0.5
# decorator==5.0.9
# defusedxml==0.7.1
# entrypoints==0.3
# et-xmlfile==1.1.0
# flatbuffers==1.12
# gast==0.4.0
# google-auth==1.32.0
# google-auth-oauthlib==0.4.4
# google-pasta==0.2.0
# grpcio==1.34.1
# h5py==3.1.0
# idna==2.10
# ipykernel==5.5.5
# ipython==7.23.1
# ipython-genutils==0.2.0
# ipywidgets==7.6.3
# jedi==0.18.0
# Jinja2==3.0.1
# joblib==1.0.1
# jsonschema==3.2.0
# jupyter==1.0.0
# jupyter-client==6.1.12
# jupyter-console==6.4.0
# jupyter-core==4.7.1
# jupyterlab-pygments==0.1.2
# jupyterlab-widgets==1.0.0
# keras-nightly==2.5.0.dev2021032900
# Keras-Preprocessing==1.1.2
# keras-visualizer==2.4
# kiwisolver==1.3.1
# Markdown==3.3.4
# MarkupSafe==2.0.1
# matplotlib==3.4.2
# matplotlib-inline==0.1.2
# mistune==0.8.4
# murmurhash==1.0.5
# nbclient==0.5.3
# nbconvert==6.0.7
# nbformat==5.1.3
# nest-asyncio==1.5.1
# nltk==3.6.2
# notebook==6.4.0
# notebook-as-pdf==0.5.0
# numpy==1.19.5
# oauthlib==3.1.1
# openpyxl==3.0.7
# opt-einsum==3.3.0
# packaging==20.9
# pandas==1.2.4
# pandoc==1.0.2
# pandocfilters==1.4.3
# parso==0.8.2
# pathy==0.5.2
# pickleshare==0.7.5
# Pillow==8.2.0
# ply==3.11
# preshed==3.0.5
# prometheus-client==0.10.1
# prompt-toolkit==3.0.18
# protobuf==3.17.3
# pyasn1==0.4.8
# pyasn1-modules==0.2.8
# pycparser==2.20
# pydantic==1.7.4
# pyee==8.1.0
# Pygments==2.9.0
# pyparsing==2.4.7
# PyPDF2==1.26.0
# pyppeteer==0.2.5
# pyrsistent==0.17.3
# python-dateutil==2.8.1
# pytz==2021.1
# pywin32==300
# pywinpty==1.1.1
# pyzmq==22.0.3
# qtconsole==5.1.0
# QtPy==1.9.0
# regex==2021.4.4
# requests==2.25.1
# requests-oauthlib==1.3.0
# rsa==4.7.2
# scikit-learn==0.24.2
# scipy==1.6.3
# seaborn==0.11.1
# Send2Trash==1.5.0
# six==1.15.0
# smart-open==3.0.0
# sns==0.1
# spacy==3.0.6
# spacy-legacy==3.0.6
# srsly==2.4.1
# tensorboard==2.5.0
# tensorboard-data-server==0.6.1
# tensorboard-plugin-wit==1.8.0
# tensorflow==2.5.0
# tensorflow-estimator==2.5.0
# termcolor==1.1.0
# terminado==0.10.0
# testpath==0.5.0
# thinc==8.0.6
# threadpoolctl==2.1.0
# tornado==6.1
# tqdm==4.61.1
# traitlets==5.0.5
# typer==0.3.2
# typing-extensions==3.7.4.3
# urllib3==1.26.5
# visualizer==0.0.10
# wasabi==0.8.2
# wcwidth==0.2.5
# webencodings==0.5.1
# websockets==8.1
# Werkzeug==2.0.1
# widgetsnbextension==3.5.1
# wordcloud==1.8.1
# wrapt==1.12.1
# xlrd==2.0.1
# 
# 

# ## Setup

# In[88]:


from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
import  keras.metrics
from tensorflow.keras.optimizers import RMSprop
import tensorflow
from keras_visualizer import visualizer
from keras.utils import np_utils
from tensorflow.keras import layers

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# ## Load and prepare the data

# In[75]:


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and validation sets
(X_train, y_train), (X_valid, y_valid) = mnist.load_data()


# In[76]:


X_train.shape


# In[77]:


y_train.shape


# In[78]:


y_train[0:12]


# In[79]:


plt.figure(figsize=(5,5))
for k in range(12):
    plt.subplot(3, 4, k+1)
    plt.imshow(X_train[k], cmap='Greys')
    plt.axis('off')
plt.tight_layout()
plt.show()


# In[80]:


X_valid.shape


# In[81]:


y_valid.shape


# In[82]:


y_valid[0]


# In[83]:


plt.imshow(X_valid[0], cmap='Greys')
plt.axis('off')
plt.show()


# In[84]:


# Reshape (flatten) images 
X_train_reshaped = X_train.reshape(60000, 784).astype('float32')
X_valid_reshaped = X_valid.reshape(10000, 784).astype('float32')

# Scale images to the [0, 1] range
X_train_scaled_reshaped = X_train_reshaped / 255
X_valid_scaled_reshaped = X_valid_reshaped / 255

# Renaming for conciseness
X_training = X_train_scaled_reshaped
X_validation = X_valid_scaled_reshaped

print("X_training shape (after reshaping + scaling):", X_training.shape)
print(X_training.shape[0], "train samples")
print("X_validation shape (after reshaping + scaling):", X_validation.shape)
print(X_validation.shape[0], "validation samples")


# In[90]:


# convert class vectors to binary class matrices
y_training = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_validation = keras.utils.np_utils.to_categorical(y_valid, num_classes)


# In[91]:


print(y_valid[0])
print(y_validation[0])


# ### Question 1

# ## X_training and Y_training  variables represents the data that will be used as the training data for our model. This is the data from which the algorithm will learn from for all weights in our  network.It will also help to predict the output you architect your model to determine. 
# 
# ## X_validation and Y_validation these variables represent/hold the unseen data. Data that will be given to the model to report performance metrics. It's also considered test data sets helps validate model. This will also help determine overfitting underfitting and validation. 

# ## PART 1 - Shallow neural network architecture

# In[92]:


model = Sequential()
model.add(Dense(64, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))


# In[93]:


model.summary()


# #### Question 2 

# 2A The Summary() Method shows the structure of the neural network. the nueral network we defined above. In the settings above you can see we feed the in the input layer with 784 nodes.The inputes represent a single pixel fomr the images. It also shows us param information.The parameter count is the amount of parameters that can be manipulated in the model. This also means the total  dimensions of the optimization problem.
# 
# 2B the NN has a single hidden layer which is why we call it the shallow nueral network.A shallow nueral network usually has 1 or 2 hidden layers.Also itt has 64 nodes. When we connect all nodes in the input layer to all the nodes in the hidden layer the nueral network will hgave 50240 parameters which you can validate with calulations below.
# 
# 2C The final layer displays the the output layer it has 10 nodes because of the 10 classes.

# In[94]:


(64*784)


# In[95]:


(64*784)+64


# In[96]:


(10*64)+10


# ### Configure model

# In[99]:


model.compile(
    loss='mean_squared_error', 
    optimizer="sgd", 
    metrics=['accuracy']
)


# In[ ]:





# ### (OPTIONAL) TODO 2 
# 
# Try different options for `loss` and `optimizer`, for example:
# ```
# model.compile(
#   optimizer='adam',
#   loss='categorical_crossentropy',
#   metrics=['accuracy']
# )
# ```

# 1-options

# In[47]:


model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=[
        keras.metrics.MeanSquaredError(),
        keras.metrics.AUC(),
    ]
)


# 2-options

# In[48]:


model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=[
        keras.metrics.MeanSquaredError(name='my_mse'),
        keras.metrics.AUC(name='my_auc'),
    ]
)


# 3-options

# In[49]:


model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=[
        'MeanSquaredError',
        'AUC',
    ]
)


# ### Train!

# ### Plot learning curves

# In[100]:


batch_size=128
epochs=500

history = model.fit(
  X_training, # training data
  y_training, # training targets
  epochs=epochs,
  batch_size=batch_size,
  verbose=1,
  validation_data=(X_validation, y_validation)
)


# In[101]:


# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# QUESTION 3: Does the model show any indication of overfitting? Why (not)?

# QUESTION 4: How do the accuracy and loss compare to the previous model?What can you infer from this comparison? 

# When we look at the graphs generated above the model converges because the loss lines form a horzontal line after the epoch is greater than 300. 
# There is no overfitting in the model because if you look at nth validation set it is not different from the behavior of the model with the training dataa  set. I can clearly infer the model does handle generalization very well and above graphs validates that.

# ### Evaluate the model

# In[102]:


model.evaluate(X_validation, y_validation)


# ### TODO 3
# 
# Write code to display the confusion matrix for your classifier and comment on the insights such confusion matrix provides.
# 
# See [this](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html) for an example.

# In[103]:


import warnings
warnings.filterwarnings("ignore")
def conMatrix():
    plt.figure(figsize=(20,10))
    sns.heatmap(confusion_matrix(y_valid, model.predict_classes(X_validation)) , 
                cmap=sns.color_palette("pastel", as_cmap=True),
               annot=True, fmt="d")
    plt.title("Conf Matrix")
    plt.show()
conMatrix()


# In[104]:


mod =  model.predict_classes(X_validation)
report = classification_report(y_valid, mod)
print(report)


# ### Question 3

# The confusion matrix can give us performance metrics of the model some metrics are like accuracy score and over all error.
# If we look at index 0 on first column you can see the model is doing well. You can see forom 980 samples that have 0 label 956 were done callsification right. If you look at index 5 you can see support for 892 and 738 classified correctly which is weaker. Index 5 has more issues on correct classfication
# 

# ### (OPTIONAL) TODO 4
# 
# Write code to display 10 cases where the classifier makes mistakes. Make sure to display both the true value as well as the predicted value.
# 
# See [this](https://conx.readthedocs.io/en/latest/MNIST.html) for an example.

# In[105]:



y_pred = model.predict_classes(X_validation)
print(y_pred)


# In[106]:


def plotHandler():
    plt.figure(figsize=(20,20))
    for k in range(10):
        plt.subplot(3, 4, k+1)
        plt.imshow(X_valid[y_valid!=y_pred][k], cmap='Greys')
        plt.title("act val:%d -- pred val:%d"%(y_valid[y_valid!=y_pred][k], y_pred[y_valid!=y_pred][k]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
plotHandler()


# ### Different optimizer Adam (OPTIONAL Questions)

# In[107]:


batch_size=128
epochs=1000

model.compile(
    loss='mean_squared_error', 
    optimizer="adam", 
    metrics=['accuracy']
)

history = model.fit(
  X_training, # training data
  y_training, # training targets
  epochs=epochs,# number of passes of the entire training datase
  batch_size=batch_size, # number of samples processed before the model is updated
  verbose=1,
  validation_data=(X_validation, y_validation)
)


# In[110]:


def plot():
    # show all all data in history
    print(history.history.keys())

     # display summary of history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # breakdown summary history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def confMatrix():
    plt.figure(figsize=(15,10))
    sns.heatmap(confusion_matrix(y_valid, model.predict_classes(X_validation)) , 
                cmap=sns.color_palette("pastel", as_cmap=True),
               annot=True, fmt="d")
    plt.title("Conf Matrix")
    plt.show()
plot()
confMatrix()


# In[111]:


print(classification_report(y_valid, model.predict_classes(X_validation)))


# In[112]:


model.evaluate(X_validation, y_validation)


# Optimizers are functions/algorithms used to manipulate the features of your neural network like learningr rates and weights to bring down the losses. Some goals for optimizers to assist with quicker results.Above you can see optimizer does help perform better.You can see that the loss lines for fit and test deteriorate heavy towward 480-500. As we increase iterations we will eventually get better scores. 

# ## PART 2 - Convolutional neural network (CNN) architecture

# In[113]:


model_cnn = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model_cnn.summary()


# ### Configure model

# In[114]:


model_cnn.compile(
    loss="categorical_crossentropy", 
    optimizer="adam", 
    metrics=["accuracy"]
)


# ### Prepare the data
# The CNN does not expect the images to be flattened.

# In[116]:


# upload
(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

# convert vectors to binary  matrices
y_training = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_validation = keras.utils.np_utils.to_categorical(y_valid, num_classes)

# update images to [0, 1] range 
X_train_cnn = X_train.astype("float32") / 255
X_valid_cnn = X_valid.astype("float32") / 255

# update  dimension of train/test inputs
X_train_cnn = np.expand_dims(X_train_cnn, -1)
X_valid_cnn = np.expand_dims(X_valid_cnn, -1)

# Make sure images have shape (28, 28, 1)
print("x_train shape:", X_train_cnn.shape)
print(X_train_cnn.shape[0], "train samples")
print(X_valid_cnn.shape[0], "test samples")


# ### Train!
# 

# In[117]:


batch_size=128
epochs=15

history = model_cnn.fit(
  X_train_cnn, # training data
  y_training, # training targets
  epochs=epochs,
  batch_size=batch_size,
  verbose=1,
  validation_data=(X_valid_cnn, y_validation)
)


# ### Plot learning curves

# In[118]:


def plot():
    # show all all data in history
    print(history.history.keys())

    # display summary of history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # breakdown summary history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
plot()


# ### Evaluate the model

# In[119]:


model_cnn.evaluate(X_valid_cnn, y_validation)


# In[124]:


def confMatrix():
    plt.figure(figsize=(15,10))
    sns.heatmap(confusion_matrix(y_valid, model_cnn.predict_classes(X_valid_cnn)) , 
                cmap=sns.color_palette("pastel", as_cmap=True),
               annot=True, fmt="d")
    plt.title("Conf Matrix")
    plt.show()
confMatrix()


# In[130]:


print(model_cnn.predict_classes(X_valid_cnn))
print(y_valid)

print(classification_report(y_valid, model_cnn.predict_classes(X_valid_cnn)))


# ### QUESTION 4: How do the accuracy and loss compare to the previous model?What can you infer from this comparison? 

# You look at the curves and the results from conf matrix above we notice better results from 90% to 99%.Loss value shows how poorly or well a model behaves after each iteration of optimization.The accuracy data is used to determine the models performance in an unobfuscated way. You can aslo validate with above we reduced iterations which means it helped us reduce the time by 50% which was alot helpful it was taxing on my cpu and took more time to run other models.You can validate with example above with deeper models we get more better performance resutls and also quicker results. This has some things as I was getting better results. I also continued to monitor any over fitting that might happening. So overfitting can be a possible issue with this.
# 
