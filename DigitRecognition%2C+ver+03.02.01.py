
# coding: utf-8

# In[1]:

# Import necessary library and enable graph plot

import numpy as np
import pandas as pd
import datetime

import matplotlib.pyplot as plt
get_ipython().magic('pylab inline')

import os, sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm # SVC method
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA


# In[3]:

# Project directory: D:/Projects/5.0 Personal project/26. Digit Recognition
# Prepare the environment

def now():
    tmp = datetime.datetime.now().strftime("%m/%d/%y %H:%M:%S")
    return tmp

def change_dir(dir):
    os.chdir("D:/Projects/5.0 Personal project/26. Digit Recognition")
    print("{}: Current Working Directory: {}".format(now(), os.getcwd()))
    os.chdir(dir)
    print("{}: Now Working Directory: {}".format(now(), os.getcwd()))

data_dir = "./Data"
change_dir(str(data_dir))


# In[4]:

#Load training dataset
train = pd.read_csv('train.csv')
print("The sizing of training data: {}".format(train.shape))
print("First 5 records: \n")
print(train.head(5))

# This will list the entire column name
# print(list(train))

target = train["label"]
train = train.drop("label",1)


# In[5]:

# Understand more on the data
print("Check the label of training data:\n{}".format(target.value_counts(sort=True)))

# What about the actual pixel variables?
figure(figsize(10,10))
for digit_num in range(0,4):
    subplot(2,2,digit_num+1)
    plt.hist(train.iloc[digit_num])
    suptitle("Before transform")

train /= 255

figure(figsize(10,10))
for digit_num in range(0,4):
    subplot(2,2,digit_num+1)
    plt.hist(train.iloc[digit_num])
    suptitle("After transform")


# ## Visualise the sample 
# #### The pixel can be plot to number by using the concept below.
# 
# 000 001 002 003 ... 026 027
# 
# 028 029 030 031 ... 054 055
# 
# 056 057 058 059 ... 082 083
# 
#     ... blah blah blah ...
#  
# 728 729 730 731 ... 754 755
# 
# 756 757 758 759 ... 782 783
# 
# Hence, we can reshape the pixel to 28 * 28.

# In[6]:

figure(figsize(5,5))
for digit_num in range(0,25):
    subplot(5,5,digit_num+1)
    grid_data = train.iloc[digit_num].as_matrix().reshape(28,28)  # reshape from 1d to 2d pixel array
    plt.imshow(grid_data, interpolation = "none", cmap = "winter")
    xticks([])
    yticks([])


# In[8]:

# Build Function for model building and score them
def eval_model_classifier(model, data, target, split_ratio):
    trainX, testX, trainY, testY = train_test_split(data, target, train_size=split_ratio, random_state=0)
    model.fit(trainX, trainY) # Build model using training data
    return model.score(testX,testY)


# In[11]:

# Initialise values
num_estimators_array = np.array([1,5,10,50,100,200,500]) 
num_smpl = 10 # Test run the model according to samples_number
num_grid = len(num_estimators_array)
score_array_mu = np.zeros(num_grid) # Keep mean
score_array_sigma = np.zeros(num_grid) # Keep Standard deviation 
j=0

print("{}: RandomForestClassification Starts!".format(now()))
for n_estimators in num_estimators_array:
    score_array = np.zeros(num_smpl) # Initialize
    for i in range(0,num_smpl):
        rf_class = RandomForestClassifier(n_estimators = n_estimators, n_jobs=1, criterion="gini")
        score_array[i] = eval_model_classifier(rf_class, train.iloc[0:1000], target.iloc[0:1000], 0.8)
    score_array_mu[j], score_array_sigma[j] = mean(score_array), std(score_array)
    print("{}: Done try n_estimators = {} with mean score = {} and standard deviation = {}".format(now(),
                                                                                                   n_estimators, 
                                                                                                   round(score_array_mu[j], 2),
                                                                                                   round(score_array_sigma[j], 2)))
    j=j+1

print("{}: RandomForestClassification Done!".format(now()))

figure(figsize(7,3))
errorbar(num_estimators_array, score_array_mu, yerr=score_array_sigma, fmt='k.-')
xscale("log")
xlabel("number of estimators",size = 16)
ylabel("accuracy",size = 16)
xlim(0.9,600)
ylim(0.5,0.95)
title("Random Forest Classifier", size = 18)
grid(which="both")


# In[15]:

C_array = np.array([0.5, 0.1, 1, 5, 10])
score_array = np.zeros(len(C_array))
score_mu = np.zeros(len(C_array))
score_sigma = np.zeros(len(C_array))

print("{}: SVM Classifier (kernel = Linear) Starts!".format(now()))
i=0
for C_val in C_array:
    svc_class = svm.SVC(kernel='linear', random_state=1, C = C_val)
    score_array[i] = eval_model_classifier(svc_class, train.iloc[0:1000], target.iloc[0:1000], 0.8)
    score_mu[i], score_sigma[i] = mean(score_array[i]), std(score_array[i])
    print("{}: Try C = {}, mean score = {} and standard deviation = {}".format(now(),
                                                                               C_val,
                                                                               round(score_mu[i], 2),
                                                                               round(score_sigma[i], 2)))
    i=i+1

print("{}: SVM Classifier (kernel = Linear) Done!".format(now()))

figure(figsize(7,3))
errorbar(C_array, score_mu, yerr=score_sigma, fmt='k.-')
xlabel("C",size = 16)
ylabel("accuracy",size = 16)
title("SVM Classifier (Linear)", size = 18)
grid(which="both")


# In[13]:

gamma_array = np.array([0.001, 0.01, 0.1, 1, 10])
score_array = np.zeros(len(gamma_array))
score_mu = np.zeros(len(gamma_array))
score_sigma = np.zeros(len(gamma_array))

print("{}: SVM Classifier (kernel = RBF) Starts!".format(now()))
i=0
for gamma_val in gamma_array:
    svc_class = svm.SVC(kernel='rbf', random_state=1, gamma = gamma_val)
    score_array[i] = eval_model_classifier(svc_class, train.iloc[0:1000], target.iloc[0:1000], 0.8)
    score_mu[i], score_sigma[i] = mean(score_array[i]), std(score_array[i])
    print("{}: Try gamma = {}, mean score = {} and standard deviation = {}".format(now(),
                                                                                  gamma_val,
                                                                                  round(score_mu[i], 2),
                                                                                  round(score_sigma[i], 2)))
    i=i+1

print("{}: SVM Classifier (kernel = RBF) Done!".format(now()))

figure(figsize(10,5))
errorbar(gamma_array, score_mu, yerr=score_sigma, fmt='k.-')
xscale('log')
xlabel("Gamma",size = 16)
ylabel("accuracy",size = 16)
ylim(0.1, 1)
title("SVM Classifier (RBF)", size = 18)
grid(which="both")


# ### Using Principal Component Analysis (PCA)
# 
# As can be seen from the data, there are many features which represented each pixel. We will try to find the pattern while still retain the information (hence, minimal loss of information).

# In[19]:

n_components_array = np.array([2, 4, 8, 16, 32])
for n_components in n_components_array:
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
    pca.fit(train)
    transform = pca.transform(train)
    figure(figsize(15,10))
    plt.scatter(transform[:,0],transform[:,1], s=20, c = target, cmap = "nipy_spectral", edgecolor = "None")
    plt.colorbar()
    clim(0,9)
    xlabel("PC1")
    ylabel("PC2")
    title("With k={}".format(n_components))


# In[23]:

# Try to choose the minimum k (components) while retain the most information
n_components_array= np.array([2, 4, 8, 16, 25, 32, 36, 49, 64, 128])
vr = np.zeros(len(n_components_array))

i=0;
for n_components in n_components_array:
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
    pca.fit(train)
    vr[i] = sum(pca.explained_variance_ratio_)
    i=i+1 

figure(figsize(8,4))
plot(n_components_array,vr,'k.-')
xscale("log")
ylim(9e-2,1.1)
yticks(linspace(0.2,1.0,9))
xlim(1.5, 150)
grid(which="both")
xlabel("number of PCA components",size=16)
ylabel("variance ratio",size=16)


# In[26]:

n_components = 32
pca = PCA(n_components = n_components, svd_solver = 'randomized', whiten=True).fit(train)
train_pca = pca.transform(train)

plt.hist(pca.explained_variance_ratio_, bins=n_components, log=True)
pca.explained_variance_ratio_.sum()


# In[28]:

param_grid = {"C": [0.1, 1, 10]
              ,"gamma": [0.1, 1, 10]}

classifier = svm.SVC()
gs = GridSearchCV(estimator=classifier, param_grid=param_grid, scoring='accuracy', cv=2, n_jobs=-1, verbose=1)
gs = gs.fit(train_pca, target)


# In[33]:

print(gs.best_score_)
print(gs.best_params_)

param_to_use = gs.best_params_


# In[35]:

# Build model SVM using PCA as input

score_array = []
print("{}: Start run!".format(now()))
my_model = svm.SVC(kernel='rbf', C=param_to_use['C'], gamma=param_to_use['gamma'])
score_array = eval_model_classifier(my_model, train_pca, target, 0.8)
print("{}: output score = {}".format(now(), score_array))
print("{}: Done!!!".format(now()))


# In[64]:

test = pd.read_csv('test.csv')

test /= 255


# In[53]:

my_model = my_model.fit(train_pca, target)
print(my_model)


# In[65]:

pred = my_model.predict(pca.transform(test))
test['Label'] = pd.Series(pred)
test['ImageId'] = test.index +1
sub = test[['ImageId', 'Label']]
print(test['Label'].value_counts(sort=True))


# In[68]:

# Review our prediction
print(test['Label'].head(25))


# In[62]:

figure(figsize(5,5))
for digit_num in range(0,25):
    subplot(5,5,digit_num+1)
    grid_data = test.iloc[digit_num,0:784].as_matrix().reshape(28,28)  # reshape from 1d to 2d pixel array
    plt.imshow(grid_data, interpolation = "none", cmap = "winter")
    xticks([])
    yticks([])


# In[69]:

sub.to_csv('submission.csv', index=False)
# This yields score 0.97957 on the leaderboard... (first submission on 2-Mar-2017) 516th

