# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 22:53:22 2022

@author: Sara Morani
"""

# import library
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt

#For Data  Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#For Machine Learning Algorithm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



df = pd.read_excel('data_v2.xlsx')


df.head(5)
df.info()

df.isnull().sum()

df= df.dropna(subset=['billing_region1_name'])

df.duplicated().sum()

df.describe()

df_1= df.drop(['billing_region1_name','billing_region2_name','billing_region3_name'],axis=1)

for col in df_1.columns:
    print(col)

df_2 = df_1.drop(['fasion_items','digital_goods_items','gm_items', "el_items",'fmcg_items','total_nmv'],axis = 1)

df_3=df_2[(df_2['total_items']>0) & (df_2['total_gmv']>0)]
df_3.describe() 

for col in df_3.columns:
    print(col)



#Cohort Analysis
def get_month(x) : return dt.datetime(x.year,1,x.month)
df_3['InvoiceMonth'] = df['order_date'].apply(get_month)
grouping = df_3.groupby('buyer_id')['InvoiceMonth']
df_3['CohortMonth'] = grouping.transform('min')
df.tail()



def get_month_int (dframe,column):
    year = dframe[column].dt.year
    month = dframe[column].dt.month
    day = dframe[column].dt.day
    return year, day , month

invoice_year,invoice_month,_ = get_month_int(df_3,'InvoiceMonth')
cohort_year,cohort_month,_ = get_month_int(df_3,'CohortMonth')

year_diff = invoice_year - cohort_year 
month_diff = invoice_month - cohort_month 

df_3['CohortIndex'] = year_diff * 12 + month_diff + 1 


#Count monthly active customers from each cohort
grouping = df_3.groupby(['CohortMonth', 'CohortIndex'])
cohort_data = grouping['buyer_id'].apply(pd.Series.nunique)
# Return number of unique elements in the object.
cohort_data = cohort_data.reset_index()
cohort_counts = cohort_data.pivot(index='CohortMonth',columns='CohortIndex',values='buyer_id')
ret = cohort_counts

cohort_data.head()
print(df_3.shape)
print(cohort_data.shape)

# Retention table
cohort_size = cohort_counts.iloc[:,0]
retention = cohort_counts.divide(cohort_size,axis=0) #axis=0 to ensure the divide along the row axis 
retention1 = retention.round(3) * 100 #to show the number as percentage

#Build the heatmap
plt.figure(figsize=(15, 8))
plt.title('Retention rates')
sns.heatmap(data=retention,annot = True,fmt = '.0%',vmin = 0.0,vmax = 0.5,cmap="BuPu_r")
plt.show()


# Calculate RFM metrics

snapshot_date = df_3['order_date'].max() + dt.timedelta(days=1)
snapshot_date


rfm = df_3.groupby(['buyer_id']).agg({'order_date': lambda x : (snapshot_date - x.max()).days,
                                      'sales_order_id':'count','total_gmv': 'sum'})
#Function Lambdea: it gives the number of days between hypothetical today and the last transaction

#Rename columns
rfm.rename(columns={'order_date':'Recency','sales_order_id':'Frequency','total_gmv':'MonetaryValue'}
           ,inplace= True)

#Final RFM values
rfm.head()


for col in rfm.columns:
    print(col)
    
    
# # create labels and assign them to tree percentile groups 
# r_labels = range(4, 0, -1)
# r_quartiles = pd.qcut(rfm.Recency, q = 4, labels = r_labels)
# f_labels = range(1, 5)
# f_quartiles = pd.qcut(rfm.Frequency, q = 4, labels = f_labels)
# m_labels = range(1, 5)
# m_quartiles = pd.qcut(rfm.MonetaryValue, q = 4, labels = m_labels)

#checking the quartiles

quartiles = rfm.quantile(q=[0.25, 0.5, 0.75])
quartiles

#creating quartiles
def recency_score (data):
    if data <= 24:
        return 4
    elif data <= 65:
        return 3
    elif data <= 174:
        return 2
    else:
        return 1

def frequency_score (data):
    if data <= 1:
        return 1
    elif data <= 3:
        return 2
    elif data <= 8:
        return 3
    else:
        return 4

def monetary_value_score (data):
    if data <= 1299:
        return 1
    elif data <= 4011:
        return 2
    elif data <= 13785:
        return 3
    else:
        return 4

rfm['R'] = rfm['Recency'].apply(recency_score )
rfm['F'] = rfm['Frequency'].apply(frequency_score)
rfm['M'] = rfm['MonetaryValue'].apply(monetary_value_score)
rfm.head()

#calcualte overall RFM score
rfm['RFM_score'] = rfm[['R', 'F', 'M']].sum(axis=1)
rfm.head()

#label the RFM score
rfm['label'] = 'Bronze' 
rfm.loc[rfm['RFM_score'] > 4, 'label'] = 'Silver' 
rfm.loc[rfm['RFM_score'] > 6, 'label'] = 'Gold'
rfm.loc[rfm['RFM_score'] > 8, 'label'] = 'Platinum'
rfm.loc[rfm['RFM_score'] > 10, 'label'] = 'Diamond'

rfm.head()

#segmentation visualtization

barplot = dict(rfm['label'].value_counts())
bar_names = list(barplot.keys())
bar_values = list(barplot.values())
plt.bar(bar_names,bar_values)
print(pd.DataFrame(barplot, index=[' ']))

#Data Pre-Processing for Kmeans Clustering
rfm_rfm = rfm[['Recency','Frequency','MonetaryValue']]
print(rfm_rfm.describe())

# plot the distribution of RFM values
f,ax = plt.subplots(figsize=(10, 12))
plt.subplot(3, 1, 1); sns.distplot(rfm.Recency, label = 'Recency')
plt.subplot(3, 1, 2); sns.distplot(rfm.Frequency, label = 'Frequency')
plt.subplot(3, 1, 3); sns.distplot(rfm.MonetaryValue, label = 'Monetary Value')
plt.style.use('fivethirtyeight')
plt.tight_layout()
plt.show()

#Unskew the data with log transformation
rfm_log = rfm[['Recency', 'Frequency', 'MonetaryValue']].apply(np.log, axis = 1).round(3)
#or rfm_log = np.log(rfm_rfm)


# plot the distribution of RFM values
f,ax = plt.subplots(figsize=(10, 12))
plt.subplot(3, 1, 1); sns.distplot(rfm_log.Recency, label = 'Recency')
plt.subplot(3, 1, 2); sns.distplot(rfm_log.Frequency, label = 'Frequency')
plt.subplot(3, 1, 3); sns.distplot(rfm_log.MonetaryValue, label = 'Monetary Value')
plt.style.use('fivethirtyeight')
plt.tight_layout()
plt.show()

#%%Implementation of K-Means Clustering

#1. Data Pre-Processing
#Normalize the variables with StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(rfm_log)
#Store it separately for clustering
rfm_normalized= scaler.transform(rfm_log)


# Choosing a Number of Clusters

from sklearn.cluster import KMeans

#First : Get the Best KMeans 
ks = range(1,8)
inertias=[]
for k in ks :
    # Create a KMeans clusters
    kc = KMeans(n_clusters=k,random_state=1)
    kc.fit(rfm_normalized)
    inertias.append(kc.inertia_)

# Plot ks vs inertias
f, ax = plt.subplots(figsize=(15, 8))
plt.plot(ks, inertias, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(ks)
plt.style.use('ggplot')
plt.title('What is the Best Number for KMeans ?')
plt.show()

#silhouette method

from sklearn.metrics import silhouette_score
wcss_silhouette = []
for i in range(2,12):
    km = KMeans(n_clusters=i, random_state=0,init='k-means++').fit(rfm_normalized)
    preds = km.predict(rfm_normalized)    
    silhouette = silhouette_score(rfm_normalized,preds)
    wcss_silhouette.append(silhouette)
    print("Silhouette score for number of cluster(s) {}: {}".format(i,silhouette))

plt.figure(figsize=(10,5))
plt.title("The silhouette coefficient method \nfor determining number of clusters\n",fontsize=16)
plt.scatter(x=[i for i in range(2,12)],y=wcss_silhouette,s=150,edgecolor='k')
plt.grid(True)
plt.xlabel("Number of clusters",fontsize=14)
plt.ylabel("Silhouette score",fontsize=15)
plt.xticks([i for i in range(2,12)],fontsize=14)
plt.yticks(fontsize=15)
plt.show()


# clustering
kc = KMeans(n_clusters= 5, random_state=1)
kc.fit(rfm_normalized)

#Create a cluster label column in the original DataFrame
cluster_labels = kc.labels_

#Calculate average RFM values and size for each cluster:
rfm_rfm_k3 = rfm_rfm.assign(K_Cluster = cluster_labels)

#Calculate average RFM values and sizes for each cluster:
rfm_rfm_k3.groupby('K_Cluster').agg({'Recency': 'mean','Frequency': 'mean',
                                         'MonetaryValue': ['mean', 'count'],}).round(0)


rfm_normalized = pd.DataFrame(rfm_normalized,index=rfm_rfm.index,columns=rfm_rfm.columns)
rfm_normalized['K_Cluster'] = kc.labels_
rfm_normalized['label'] = rfm['label']
rfm_normalized.reset_index(inplace = True)

#Melt the data into a long format so RFM values and metric names are stored in 1 column each
rfm_melt = pd.melt(rfm_normalized,id_vars=['buyer_id','label','K_Cluster'],value_vars=['Recency', 'Frequency', 'MonetaryValue'],
var_name='Metric',value_name='Value')
rfm_melt.head()



#Snake Plot and Heatmap
f, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 8))
sns.lineplot(x = 'Metric', y = 'Value', hue = 'label', data = rfm_melt,ax=ax1)

# a snake plot with K-Means
sns.lineplot(x = 'Metric', y = 'Value', hue = 'K_Cluster', data = rfm_melt,ax=ax2)

plt.suptitle("Snake Plot of RFM",fontsize=24) #make title fontsize subtitle 
plt.show()

#%%Building Heatmap

# The further a ratio is from 0, the more important that attribute is for a segment relative to the total population
cluster_avg = rfm_rfm_k3.groupby(['K_Cluster']).mean()
population_avg = rfm_rfm.mean()
relative_imp = cluster_avg / population_avg - 1
relative_imp.round(2)


# the mean value in total 
total_avg = rfm.iloc[:, 0:3].mean()
# calculate the proportional gap with total mean
cluster_avg = rfm.groupby('label').mean().iloc[:, 0:3]
prop_rfm = cluster_avg/total_avg - 1
prop_rfm.round(2)

# heatmap with RFM
f, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 5))
sns.heatmap(data=relative_imp, annot=True, fmt='.2f', cmap='Blues',ax=ax1)
ax1.set(title = "Heatmap of K-Means")

# a snake plot with K-Means
sns.heatmap(prop_rfm, cmap= 'Oranges', fmt= '.2f', annot = True,ax=ax2)
ax2.set(title = "Heatmap of RFM quantile")

plt.suptitle("Heat Map of RFM",fontsize=20) #make title fontsize subtitle 

plt.show()

for col in rfm_rfm_k3.columns:
    print(col)
   
    
#%% Prediction using Decision Tree & Random Forest Classifier
# Decision tree & Random Forest to Predict Customers
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

X = rfm_rfm_k3.drop(['K_Cluster'],axis=1)
y = rfm_rfm_k3['K_Cluster']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train.shape , X_test.shape
y_train.shape , y_test.shape

#Creation of Decision Tree using Gini Impurity

#Using the Decision Tree Classifier with splitting criterion as Gini impurity, the maximum depth of the tree is 10.
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=1)


# fit the model
clf_gini.fit(X_train, y_train)


#Plot the tree
plt.figure(figsize=(12,8))

tree.plot_tree(clf_gini.fit(X_train, y_train)) 


#Predict the values 
y_pred_gini = clf_gini.predict(X_test)

#Predict the value using X train for accuracy comparision 
y_pred_train_gini = clf_gini.predict(X_train)

y_pred_train_gini


#Determine the accuracy score
print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))
#Accuracy Score for training set
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))


#Creation of Decision Tree using with entropy

clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=1)

# fit the model
clf_en.fit(X_train, y_train)


#figure
plt.figure(figsize=(12,8))
tree.plot_tree(clf_en.fit(X_train, y_train)) 



#Predict the values 
y_pred_en = clf_en.predict(X_test)


#Predict the value using X train for accuracy comparision
y_pred_train_en = clf_en.predict(X_train)


print('Model accuracy score with criterion entropy: {0:0.4f}'. format(accuracy_score(y_test, y_pred_en)))
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_en)))


#classificarion Report using gini imuprity

print(classification_report(y_test, y_pred_gini))

#classificarion Report using entropy

print(classification_report(y_test, y_pred_en))


#Decision-Tree Classifier model using both gini index and entropy have only
# very very small difference in model accuracy and training set accuracy,
# so there is no sign of overfitting.


#%%prediction using random forest

# Split the data into training and testing sets

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

train_features, test_features, train_labels, test_labels = train_test_split(X,
                                                                            y,
                                                                            test_size = 0.20,
                                                                            random_state = 1)

rf_1 = RandomForestClassifier()      
rf_2 = rf_1.fit(train_features,train_labels)
ac_2 = accuracy_score(test_labels,rf_1.predict(test_features))
print('Accuracy is: ',ac_2)
