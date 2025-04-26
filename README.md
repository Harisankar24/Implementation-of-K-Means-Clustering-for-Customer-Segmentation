# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the necessary packages using import statement.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Import KMeans and use for loop to cluster the data.

4.Predict the cluster and plot data graphs.

5.Print the outputs and end the program 
```
## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by:Harisankar.S 
RegisterNumber:  212224240051
*/

```
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv('/content/Mall_Customers.csv')

print("Dataset Head:\n", df.head())
```
```

X = df.iloc[:, [3, 4]].values  
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()
```
```
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(5):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=colors[i], label=f'Cluster {i+1}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='yellow', label='Centroids', edgecolor='black')
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()
```

## Output:
![Screenshot 2025-04-26 104331](https://github.com/user-attachments/assets/7ce9681f-6bf8-45d6-9c42-b457a689295a)

![Screenshot 2025-04-26 104347](https://github.com/user-attachments/assets/5ab2a143-5a72-423f-9dd8-e2dc2737675c)

![Screenshot 2025-04-26 104403](https://github.com/user-attachments/assets/a8be4f4c-f93d-4251-9730-f05be0850cf3)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
