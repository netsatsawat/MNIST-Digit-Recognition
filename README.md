### This is the first try of MNIST classification for Kaggle competition

### This is the illustration of using PCA and other machine learning algorithms (i.e. RandomForest).

#### By N. Satsawat
#### Date: 24-Apr-2017

```python
# Import necessary library

import numpy as np
import pandas as pd
import datetime 
import seaborn as sb
sb.set_style("dark")

import matplotlib.pyplot as plt
%pylab inline

import os, sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn import svm
```


```python
def now():
    tmp = datetime.datetime.now().strftime("%m/%d/%y %H:%M:%S")
    return tmp

```



```python
# Read-in training dataset
train = pd.read_csv('train.csv')
print("The sizing of training data: {}".format(train.shape))
print("First 5 records: \n")
print(train.head(5))

# This will list the entire column name
# print(list(train))

target = train["label"]
train = train.drop("label",1)

```

    The sizing of training data: (42000, 785)
    First 5 records: 
    
       label  pixel0  pixel1  pixel2  pixel3  pixel4  pixel5  pixel6  pixel7  \
    0      1       0       0       0       0       0       0       0       0   
    1      0       0       0       0       0       0       0       0       0   
    2      1       0       0       0       0       0       0       0       0   
    3      4       0       0       0       0       0       0       0       0   
    4      0       0       0       0       0       0       0       0       0   
    
       pixel8    ...     pixel774  pixel775  pixel776  pixel777  pixel778  \
    0       0    ...            0         0         0         0         0   
    1       0    ...            0         0         0         0         0   
    2       0    ...            0         0         0         0         0   
    3       0    ...            0         0         0         0         0   
    4       0    ...            0         0         0         0         0   
    
       pixel779  pixel780  pixel781  pixel782  pixel783  
    0         0         0         0         0         0  
    1         0         0         0         0         0  
    2         0         0         0         0         0  
    3         0         0         0         0         0  
    4         0         0         0         0         0  
    
    [5 rows x 785 columns]
    


```python
# Understand more on the data
print("Check the label of training data:\n{}".format(target.value_counts(sort=True)))

# What about the actual pixel variables?
figure(figsize(10,10))
for digit_num in range(0,4):
    subplot(2,2,digit_num+1)
    plt.hist(train.iloc[digit_num])
    suptitle("Before transform")

# Normalize the training data
train /= 255

figure(figsize(10,10))
for digit_num in range(0,4):
    subplot(2,2,digit_num+1)
    plt.hist(train.iloc[digit_num])
    suptitle("After transform")
```

    Check the label of training data:
    1    4684
    7    4401
    3    4351
    9    4188
    2    4177
    6    4137
    0    4132
    4    4072
    8    4063
    5    3795
    Name: label, dtype: int64
    


![png](output_3_1.png)



![png](output_3_2.png)



```python
# Convert pixel from 784x1 (1 dimension) to 28x28 (2 dimensions)
figure(figsize(5,5))
for digit_num in range(0,25):
    subplot(5,5,digit_num+1)
    grid_data = train.iloc[digit_num].as_matrix().reshape(28,28) 
    plt.imshow(grid_data, interpolation = "none", cmap = "bone")
    xticks([])
    yticks([])
```


![png](output_4_0.png)



```python
### Create function to evaluate the score of each classification model
def eval_model_classifier(model, data, target, split_ratio):
    trainX, testX, trainY, testY = train_test_split(data, target, train_size=split_ratio, random_state=0)
    model.fit(trainX, trainY)    
    return model.score(testX,testY)
```


```python
### 1st round: RandomForestClassification

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
        print("{}: Try {} with n_estimators = {} and score = {}".format(now(), i, n_estimators, score_array[i]))
    score_array_mu[j], score_array_sigma[j] = mean(score_array), std(score_array)
    j=j+1

print("{}: RandomForestClassification Done!".format(now()))
```

    03/03/17 16:04:05: RandomForestClassification Starts!
    03/03/17 16:04:05: Try 0 with n_estimators = 1 and score = 0.49
    03/03/17 16:04:05: Try 1 with n_estimators = 1 and score = 0.58
    03/03/17 16:04:05: Try 2 with n_estimators = 1 and score = 0.525
    03/03/17 16:04:05: Try 3 with n_estimators = 1 and score = 0.51
    03/03/17 16:04:05: Try 4 with n_estimators = 1 and score = 0.59
    03/03/17 16:04:05: Try 5 with n_estimators = 1 and score = 0.565
    03/03/17 16:04:05: Try 6 with n_estimators = 1 and score = 0.475
    03/03/17 16:04:05: Try 7 with n_estimators = 1 and score = 0.57
    03/03/17 16:04:05: Try 8 with n_estimators = 1 and score = 0.51
    03/03/17 16:04:05: Try 9 with n_estimators = 1 and score = 0.48
    03/03/17 16:04:05: Try 0 with n_estimators = 5 and score = 0.685
    03/03/17 16:04:05: Try 1 with n_estimators = 5 and score = 0.73
    03/03/17 16:04:05: Try 2 with n_estimators = 5 and score = 0.655
    03/03/17 16:04:05: Try 3 with n_estimators = 5 and score = 0.655
    03/03/17 16:04:05: Try 4 with n_estimators = 5 and score = 0.725
    03/03/17 16:04:06: Try 5 with n_estimators = 5 and score = 0.74
    03/03/17 16:04:06: Try 6 with n_estimators = 5 and score = 0.74
    03/03/17 16:04:06: Try 7 with n_estimators = 5 and score = 0.68
    03/03/17 16:04:06: Try 8 with n_estimators = 5 and score = 0.725
    03/03/17 16:04:06: Try 9 with n_estimators = 5 and score = 0.64
    03/03/17 16:04:06: Try 0 with n_estimators = 10 and score = 0.83
    03/03/17 16:04:06: Try 1 with n_estimators = 10 and score = 0.83
    03/03/17 16:04:06: Try 2 with n_estimators = 10 and score = 0.76
    03/03/17 16:04:06: Try 3 with n_estimators = 10 and score = 0.785
    03/03/17 16:04:06: Try 4 with n_estimators = 10 and score = 0.815
    03/03/17 16:04:06: Try 5 with n_estimators = 10 and score = 0.76
    03/03/17 16:04:06: Try 6 with n_estimators = 10 and score = 0.815
    03/03/17 16:04:06: Try 7 with n_estimators = 10 and score = 0.775
    03/03/17 16:04:06: Try 8 with n_estimators = 10 and score = 0.795
    03/03/17 16:04:06: Try 9 with n_estimators = 10 and score = 0.8
    03/03/17 16:04:07: Try 0 with n_estimators = 50 and score = 0.87
    03/03/17 16:04:07: Try 1 with n_estimators = 50 and score = 0.875
    03/03/17 16:04:07: Try 2 with n_estimators = 50 and score = 0.885
    03/03/17 16:04:07: Try 3 with n_estimators = 50 and score = 0.865
    03/03/17 16:04:08: Try 4 with n_estimators = 50 and score = 0.87
    03/03/17 16:04:08: Try 5 with n_estimators = 50 and score = 0.86
    03/03/17 16:04:08: Try 6 with n_estimators = 50 and score = 0.885
    03/03/17 16:04:09: Try 7 with n_estimators = 50 and score = 0.88
    03/03/17 16:04:09: Try 8 with n_estimators = 50 and score = 0.88
    03/03/17 16:04:09: Try 9 with n_estimators = 50 and score = 0.875
    03/03/17 16:04:10: Try 0 with n_estimators = 100 and score = 0.885
    03/03/17 16:04:10: Try 1 with n_estimators = 100 and score = 0.9
    03/03/17 16:04:11: Try 2 with n_estimators = 100 and score = 0.88
    03/03/17 16:04:12: Try 3 with n_estimators = 100 and score = 0.89
    03/03/17 16:04:12: Try 4 with n_estimators = 100 and score = 0.9
    03/03/17 16:04:13: Try 5 with n_estimators = 100 and score = 0.89
    03/03/17 16:04:13: Try 6 with n_estimators = 100 and score = 0.895
    03/03/17 16:04:14: Try 7 with n_estimators = 100 and score = 0.89
    03/03/17 16:04:14: Try 8 with n_estimators = 100 and score = 0.88
    03/03/17 16:04:15: Try 9 with n_estimators = 100 and score = 0.88
    03/03/17 16:04:16: Try 0 with n_estimators = 200 and score = 0.9
    03/03/17 16:04:17: Try 1 with n_estimators = 200 and score = 0.91
    03/03/17 16:04:18: Try 2 with n_estimators = 200 and score = 0.89
    03/03/17 16:04:19: Try 3 with n_estimators = 200 and score = 0.88
    03/03/17 16:04:20: Try 4 with n_estimators = 200 and score = 0.895
    03/03/17 16:04:21: Try 5 with n_estimators = 200 and score = 0.885
    03/03/17 16:04:22: Try 6 with n_estimators = 200 and score = 0.905
    03/03/17 16:04:23: Try 7 with n_estimators = 200 and score = 0.9
    03/03/17 16:04:24: Try 8 with n_estimators = 200 and score = 0.9
    03/03/17 16:04:26: Try 9 with n_estimators = 200 and score = 0.88
    03/03/17 16:04:28: Try 0 with n_estimators = 500 and score = 0.895
    03/03/17 16:04:31: Try 1 with n_estimators = 500 and score = 0.88
    03/03/17 16:04:34: Try 2 with n_estimators = 500 and score = 0.91
    03/03/17 16:04:37: Try 3 with n_estimators = 500 and score = 0.91
    03/03/17 16:04:40: Try 4 with n_estimators = 500 and score = 0.905
    03/03/17 16:04:43: Try 5 with n_estimators = 500 and score = 0.925
    03/03/17 16:04:47: Try 6 with n_estimators = 500 and score = 0.89
    03/03/17 16:04:50: Try 7 with n_estimators = 500 and score = 0.91
    03/03/17 16:04:54: Try 8 with n_estimators = 500 and score = 0.895
    03/03/17 16:04:57: Try 9 with n_estimators = 500 and score = 0.9
    03/03/17 16:04:57: RandomForestClassification Done!
    


```python
figure(figsize(7,3))
errorbar(num_estimators_array, score_array_mu, yerr=score_array_sigma, fmt='k.-')
xscale("log")
xlabel("number of estimators",size = 16)
ylabel("accuracy",size = 16)
xlim(0.9,600)
ylim(0.5,0.95)
title("Random Forest Classifier", size = 18)
grid(which="both")
```


![png](output_7_0.png)



```python

C_array = np.array([0.5, 0.1, 1, 5, 10])
score_array = np.zeros(len(C_array))
i=0
for C_val in C_array:
    svc_class = svm.SVC(kernel='linear', random_state=1, C = C_val)
    score_array[i] = eval_model_classifier(svc_class, train.iloc[0:1000], target.iloc[0:1000], 0.8)
    i=i+1

score_mu, score_sigma = mean(score_array), std(score_array)

figure(figsize(7,3))
errorbar(C_array, score_array, yerr=score_sigma, fmt='k.-')
xlabel("C assignment",size = 16)
ylabel("accuracy",size = 16)
title("SVM Classifier (Linear)", size = 18)
grid(which="both")
```


![png](output_8_0.png)



```python
# Note: 
# Gamma: Kernel coefficient - the higher, it will try to exact fit to the training data, hence, can cause overfitting

gamma_array = np.array([0.001, 0.01, 0.1, 1, 10])
score_array = np.zeros(len(gamma_array))
score_mu = np.zeros(len(gamma_array))
score_sigma = np.zeros(len(gamma_array))
i=0
for gamma_val in gamma_array:
    svc_class = svm.SVC(kernel='rbf', random_state=1, gamma = gamma_val)
    score_array[i] = eval_model_classifier(svc_class, train.iloc[0:1000], target.iloc[0:1000], 0.8)
    score_mu[i], score_sigma[i] = mean(score_array[i]), std(score_array[i])
    i=i+1


figure(figsize(10,5))
errorbar(gamma_array, score_mu, yerr=score_sigma, fmt='k.-')
xscale('log')
xlabel("Gamma",size = 16)
ylabel("accuracy",size = 16)
ylim(0.1, 1)
title("SVM Classifier (RBF)", size = 18)
grid(which="both")
```


![png](output_9_0.png)



```python
# Some PCA

pca = PCA(n_components=16)
pca.fit(train)
transform = pca.transform(train)

figure(figsize(15,10))
plt.scatter(transform[:,0],transform[:,1], s=20, c = target, cmap = "nipy_spectral", edgecolor = "None")
plt.colorbar()
clim(0,9)

xlabel("PC1")
ylabel("PC2")
```




    <matplotlib.text.Text at 0x1ad8eda99b0>




![png](output_10_1.png)



```python
n_components_array= np.array([2, 4, 8, 16, 32, 64, 128])
vr = np.zeros(len(n_components_array))

i=0;
for n_components in n_components_array:
    pca = PCA(n_components=n_components)
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
```




    <matplotlib.text.Text at 0x1ad8e6afeb8>




![png](output_11_1.png)



```python
# K-nearest
n_neighbors_array = range(2,20)
weight_array = ['uniform','distance']
score_array = np.zeros(len(n_neighbors_array))

for weight in weight_array:
    i=0
    for n in n_neighbors_array:
        nbrs = KNeighborsClassifier(n_neighbors=n, weights=weight)
        score_array[i] = eval_model_classifier(nbrs, transform, target, 0.8)
        print("{}: for n_neighbors = {} and weight ={} produces score = {}".format(now(), n, weight, score_array[i]))
        i+=1

print("Done")
```

    03/03/17 16:24:59: for n_neighbors = 2 and weight =uniform produces score = 0.9508333333333333
    03/03/17 16:25:03: for n_neighbors = 3 and weight =uniform produces score = 0.96
    03/03/17 16:25:08: for n_neighbors = 4 and weight =uniform produces score = 0.9607142857142857
    03/03/17 16:25:13: for n_neighbors = 5 and weight =uniform produces score = 0.9603571428571429
    03/03/17 16:25:19: for n_neighbors = 6 and weight =uniform produces score = 0.9592857142857143
    03/03/17 16:25:25: for n_neighbors = 7 and weight =uniform produces score = 0.9591666666666666
    03/03/17 16:25:31: for n_neighbors = 8 and weight =uniform produces score = 0.9582142857142857
    03/03/17 16:25:37: for n_neighbors = 9 and weight =uniform produces score = 0.958452380952381
    03/03/17 16:25:44: for n_neighbors = 10 and weight =uniform produces score = 0.9576190476190476
    03/03/17 16:25:50: for n_neighbors = 11 and weight =uniform produces score = 0.9570238095238095
    03/03/17 16:25:57: for n_neighbors = 12 and weight =uniform produces score = 0.9578571428571429
    03/03/17 16:26:04: for n_neighbors = 13 and weight =uniform produces score = 0.9553571428571429
    03/03/17 16:26:11: for n_neighbors = 14 and weight =uniform produces score = 0.955952380952381
    03/03/17 16:26:19: for n_neighbors = 15 and weight =uniform produces score = 0.9530952380952381
    03/03/17 16:26:26: for n_neighbors = 16 and weight =uniform produces score = 0.9544047619047619
    03/03/17 16:26:34: for n_neighbors = 17 and weight =uniform produces score = 0.9545238095238096
    03/03/17 16:26:42: for n_neighbors = 18 and weight =uniform produces score = 0.9536904761904762
    03/03/17 16:26:51: for n_neighbors = 19 and weight =uniform produces score = 0.9532142857142857
    03/03/17 16:26:56: for n_neighbors = 2 and weight =distance produces score = 0.9557142857142857
    03/03/17 16:27:03: for n_neighbors = 3 and weight =distance produces score = 0.9611904761904762
    03/03/17 16:27:08: for n_neighbors = 4 and weight =distance produces score = 0.9615476190476191
    03/03/17 16:27:14: for n_neighbors = 5 and weight =distance produces score = 0.9614285714285714
    03/03/17 16:27:19: for n_neighbors = 6 and weight =distance produces score = 0.9610714285714286
    03/03/17 16:27:24: for n_neighbors = 7 and weight =distance produces score = 0.9602380952380952
    03/03/17 16:27:31: for n_neighbors = 8 and weight =distance produces score = 0.9608333333333333
    03/03/17 16:27:38: for n_neighbors = 9 and weight =distance produces score = 0.9598809523809524
    03/03/17 16:27:44: for n_neighbors = 10 and weight =distance produces score = 0.9604761904761905
    03/03/17 16:27:50: for n_neighbors = 11 and weight =distance produces score = 0.9582142857142857
    03/03/17 16:27:57: for n_neighbors = 12 and weight =distance produces score = 0.9596428571428571
    03/03/17 16:28:06: for n_neighbors = 13 and weight =distance produces score = 0.9572619047619048
    03/03/17 16:28:13: for n_neighbors = 14 and weight =distance produces score = 0.9579761904761904
    03/03/17 16:28:20: for n_neighbors = 15 and weight =distance produces score = 0.9547619047619048
    03/03/17 16:28:28: for n_neighbors = 16 and weight =distance produces score = 0.955595238095238
    03/03/17 16:28:35: for n_neighbors = 17 and weight =distance produces score = 0.955595238095238
    03/03/17 16:28:44: for n_neighbors = 18 and weight =distance produces score = 0.955
    03/03/17 16:28:53: for n_neighbors = 19 and weight =distance produces score = 0.955
    Done
    


```python
# Train all model using testing data

pca = PCA(n_components=32)
pca.fit(train, target)
transform = pca.transform(train)

KNNmodel = KNeighborsClassifier(n_neighbors=4, weights='distance').fit(transform, target)
print(KNNmodel)
```

    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=4, p=2,
               weights='distance')
    


```python
test = pd.read_csv("test.csv")
test /=255
```


```python
test_transform = pca.transform(test)
```


```python
y_pred = KNNmodel.predict(test_transform)
```

```python
test['Label'] = pd.Series(y_pred)
test['ImageId'] = test.index +1
sub = test[['ImageId', 'Label']]
print(test['Label'].value_counts(sort=True))
```

    1    3233
    7    2906
    9    2805
    2    2805
    0    2779
    6    2762
    4    2741
    3    2729
    8    2700
    5    2540
    Name: Label, dtype: int64
    


```python
sub.to_csv('submission_knn.csv', index=False)
```
