#--------------Data preprocessing and Model training on image dataset 
#These are the libraries that we will use in this project 
 
from keras.models import Sequential 
from keras.layers.core import Dense, Dropout, Activation, Flatten 
from keras.layers.convolutional import Convolution2D, MaxPooling2D 
from keras.optimizers import SGD,RMSprop,adam #for uptemization 
from keras.utils import np_utils #to convert normal clas to binary  
 
import numpy as np import matplotlib.pyplot as plt import os 
 
from PIL import Image #use to read and modify from numpy import * 
 
# SKLEARN 
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split 
 
#This line of code defining the dimension of image in rows and columns  # input image dimensions 
img_rows, img_cols = 200, 200 
 
 
#This is defining number of channel in CNN  
 
# number of channels img_channels = 1 
Channel 1 representing gray scal images 
 

#Defining input and output paths of the images directory. In my case are the following. 
 
#path of folder of images  
path1 = r'C:\Users\Hameed Ali\.spyder-py3\Myprojects\Logos\mergerd_file' 
 
#path of folder to save images     
path2 = r'C:\Users\Hameed Ali\.spyder-py3\Myprojects\Logos\mergerd_file_resize' 
 
  
#This will listen all the images from input directory  
 
listing = os.listdir(path1) 
 
#This will show the size of samples (files) in the directory 
 
num_samples=size(listing) 
 
  
#This is opening and resizing images from path1 
 
for file in listing: 
    im = Image.open(path1 + '\\' + file)       
	img = im.resize((img_rows,img_cols)) 
#This will convert all the images to grayscale of density 255 
gray = img.convert('L') 
 
#This line will save the gray images to path2 
 
gray.save(path2 +'\\' +  file, "JPEG") 
 
 
#Creating images list from path2 
imlist = os.listdir(path2) 
 
#This opening an image from path2 to get size of the image 
 
# open one image to get size 
im1 = array(Image.open('mergerd_file_resize' + '\\'+ imlist[0])) 
 
#Get the size of the image m rows and n columns  
m,n = im1.shape[0:2] # get the size of the images 
 
#Getting the total length of files in imlist 
 
# get the number of images imnbr = len(imlist) 
 
#Creating matrix to store all flatten images 
immatrix = array([array(Image.open('mergerd_file_resize'+ '\\' + im2)).flatten()               
for im2 in imlist],'f') 
#Before labeling the data all the images are with label 1 
 
label=np.ones((num_samples,),dtype = int) 
 
#Assigning labels to image classes  
label[0:1847]=0 
label[1847:]=1 
 
#Shuffling labels with data 
 
data,Label = shuffle(immatrix,label, random_state=2) 
#Assigning data and labels to variable 
train_data = [data,Label] 
 
#Reshaping flattened images 
 
img=immatrix[167].reshape(img_rows,img_cols) 
#Showing reshaped image of index [67] 
plt.imshow(img) 
 
#Gray scaled image 
 
plt.imshow(img,cmap='gray') 
 
#Printing shape of 200 x 200 = 40000 image with total number of samples 
print (train_data[0].shape) print (train_data[1].shape) 
 
#Defining batch size 
 
#batch_size to train 
batch_size = 32 
#Defining number of classes 
# number of output classes nb_classes = 2 
 
#Number of epoch 
 
# number of epochs to train 
nb_epoch = 1 
 
#	Defining number of convolutional filters to use 
nb_filters = 32 
 
#Size of pooling after each layer 
 
# size of pooling area for max pooling nb_pool = 2 
 
#Convolutional kernel size 
# convolution kernel size 
nb_conv = 3 
 
#Assigning data to Xtrain and ytrain 
 
(X, y) = (train_data[0],train_data[1]) 
 
#Splitting data 
# STEP 1: split X and y into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4) 
 
#Shape after splitting data 
 
X_train.shape 
y_train.shape 
 
#Reshaping x_train and x_test 
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols) X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols) 
 
#Converting to float32 if not in advance 
 
X_train = X_train.astype('float32') 
X_test = X_test.astype('float32') 
 
#Normalizing data  
# To make it convert faster first normalize the data 
X_train /= 255 
X_test /= 255 
 
#Shape before categorical  
 
X_train.shape y_train.shape 
 
#Printing shapes  
print('X_train shape:', X_train.shape) print(X_train.shape[0], 'train samples') print(X_test.shape[0], 'test samples') 
#Categorical conversion 
 
# convert class vectors to binary class matrices 
Y_train = np_utils.to_categorical(y_train, nb_classes) 
Y_test = np_utils.to_categorical(y_test, nb_classes) 
 
#Shape of y after conversion  
Y_train.shape 
 
#Showing an image of x_train [100] 
 
i = 100 
plt.imshow(X_train[i, 0], interpolation='nearest') 
 
#Label of index i 
print("label : ", Y_train[i,:]) 
 
#	N # Preprocessing has been Done! 
 
# Training Model model = Sequential() 
 
#Ist layer with batch size and conv kernel  
model.add(Convolution2D(32, (3, 3), activation='relu',  
                        input_shape=(1,200,200), data_format='channels_first')) 
 
 
 
#	2nd layer with actiation function and channel 1 
 
model.add(Convolution2D(32, (3, 3), activation='relu',  border_mode='valid',                       
  input_shape=(1,200,200), data_format='channels_first')) 
 
#Activation fun relu 
convout1 = Activation('relu') 
model.add(convout1) 
 
#3rd layer 
 
model.add(Convolution2D(32, (3, 3))) 
convout2 = Activation('relu') 
model.add(convout2) 
 
#Pooling data after each layer  
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool))) 
 
#Dropout ratio to avoid overfitting  
 
model.add(Dropout(0.5)) 
 
#Flattening data before output layer 
 
model.add(Flatten()) 
#Defining dense 128 with activation function  
 
model.add(Dense(128)) model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
 
#Inserting softmax layer 
model.add(Dense(nb_classes)) 
model.add(Activation('softmax')) 
 
#	Compiling model  
 
model.compile(loss='categorical_crossentropy', optimizer='adadelta') 
 
 
#Model evaluation metric accuracy checking  hist = model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
#Final results with 99.8 % accuracy  
 
model.fit(X_train, Y_train, verbose=1) 
 
#	Classification report of Precision, Recall, F1Score 
 
