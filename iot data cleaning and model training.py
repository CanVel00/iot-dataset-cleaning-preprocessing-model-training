#1. Inserting libraries 
 
from keras.models import Sequential from keras.layers import Dense, Activation from sklearn.model_selection import train_test_split from keras.utils import to_categorical 
from sklearn.preprocessing import MinMaxScaler  import pandas as pd import numpy as np 
 
from sklearn.metrics import accuracy_score from sklearn.metrics import precision_score from sklearn.metrics import recall_score from sklearn.metrics import f1_score 
from sklearn.model_selection import cross_val_score from sklearn.model_selection import cross_val_predict 
2.	Loading unclean dataset 
 
#Load CSV 
df = pd.read_csv('unclean_data.csv') 
 
#2.	Checking data types, if datatypes is object type we should convert it to float before further processing as you can see there are many features with datatype object. 
 
df.dtypes 
#2.	SyntaxError show that this line of code do not understand space in column name we have to remove spaces in csv 
 
#Data type objects conversion 
sip = df.sourceip.str.replace('.','').astype(float) 
#2.	Rename features name in csv 
#2.	Now this will work  
 
#Data type objects conversion without mean! 
 
sip = df.sourceip.str.replace('.','').astype(float) df['sourceip'] = sip sip 
#2.	We have to do the same for the rest of columns also 
dip = df.destinationip.str.replace('.','').astype(float)   df['destinationip'] = dip dip 
     
tsmp = df.Timestamp.str.replace(':','').astype(float) df['Timestamp'] = tsmp tsmp 
# 
2.	For multiple objects in same features we can convert that one by one  
 
fid = df.flowid.str.replace('.','') df['flowid'] = fid 
fid  
     
fid = df.flowid.str.replace('-','').astype(float) df['flowid'] = fid 
fid 
#2.	Now again check data types 
dtypes 
#2.	Here are two more objects with string ‘infinity’, we will do the same for this also simply convert it to 0. 
#2.	Go to that features in csv apply filter and select infinity you will see it. 
#2.	Simply replace that with zero 
 
fbt = df.flowbytes.str.replace(‘Infinity’,’0’) 
#2.	The following line of code will show 5 rows by default from the edited dataframe 
 
df.head() 
 
# 2.	This will show you the the null values from the whole dataset while in our case there is no null. If there is any null value we can fill that with any other value accordingly! 
df.isnull().sum().sum() 
df_fill_with_0 = df.fillna(0, inplace=True) 
 
#2.	After fillna see the shape of data 
 
df.shape 
#2.	This line will show you the duplicate samples in data if there any 
 
df.duplicated() 
# 2.	Replacing labels through the following line of codes. We total 12 classes. 
 
df = df.replace('BENIGN','0') df = df.replace('DrDoS_DNS','1') df = df.replace('DrDoS_LDAP','2') df = df.replace('DrDoS_MSSQL','3') df = df.replace('DrDoS_NetBIOS','4') df = df.replace('DrDoS_NTP','5') df = df.replace('DrDoS_SSDP','6') df = df.replace('DrDoS_UDP','7') 
df = df.replace('Syn','8') df = df.replace('TFTP','9') df = df.replace('UDP-lag','10') 
df = df.replace('WebDDoS','11') 
 
# 20.	Before transforming data we have to shuffle it first 
#Shuffle data frame rows 
df = df.sample(frac=1).reset_index(drop=True) 
 
# 20.	After shuffling we are slicing data to separate labels from data in the following manner according to our number of features 
 
#Slicing Features and labels 
X	= df.iloc[:,0:86] 
Y	= df.iloc[:,86:87] 
# 22.	Next is to scale the data using minmax scaler there are other scaler also that we can use. Scalling B/W 0-1 scaler = MinMaxScaler() X = scaler.fit_transform(X) 
 
# 22.	After scaling check the dimensions of data 
 
X.ndim Y.ndim 
# 22.	Now split data into train and test for testing we are taking 20% od the whole data. 
#Spliting tran and test data 
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42) 
 
# 22.	As we have 12 classes so we have to check each unique class in why Using one hot encoding 
 
num_y =len( np.unique(Y)) 
#22.	Convert classes to categories  
y_train = to_categorical(y_train,num_y) 

# 22.	Now the data is ready we have train our model for this we are using CNN 
 
#Training model 
model = Sequential() 
# 22.	This the first input layer of our model with 86 neurons 
model.add(Dense(86, activation='relu',input_dim=86)) 
 
#22.	This is the second layer with activation function ‘relu’ 
 
model.add(Dense(86, activation='relu')) 

# 22.	3rd layer with 43 neurons  
model.add(Dense(43, activation='relu'))   

#22.	This is the last layer with softmax a fully connected layer and 12 out for each class  
 
model.add(Dense(12, activation='softmax')) 
 
# 22.	This will print the summary of our model 
model.summary() 
 
# 22.	Finally we are compiling our code using optimizer ‘adam’ we can use other also, and we are printing loss and accuracy of the model 
 
model.compile(optimizer='adam',           loss='categorical_crossentropy',           metrics=['accuracy']) 
# 22.	Results have generated from two epochs and batch size of 32 
 
model.fit(x_train, y_train, epochs=2, batch_size=32) 
# 22.	It the end we are saving our clean data frame in csv to output path 
 
export_csv = df.to_csv (r'C:\Users\Hameed Ali\.spyder-py3\Myprojects\multiclassissue.csv',    index = None, header=True) #Don't forget to add '.csv' at the end of the path 
 
# 22.	Classification report of Precision, Recall, F1Score  
 




