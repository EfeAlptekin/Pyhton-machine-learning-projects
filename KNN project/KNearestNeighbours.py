#Algorithm process is like that ...
#Checking the point that already exist when we add a new point
#Founding the K neighbour points
#It's accepted according to most common neighbour point group

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Outcome = 1 Diabetic
# Outcome = 0 Fit
data = pd.read_csv("diabetes.csv")
data.head()

Diabetics = data[data.Outcome == 1]
Fits = data[data.Outcome == 0]


plt.scatter(Fits.Age, Fits.Glucose, color="green", label="Healthy", alpha = 0.4)
plt.scatter(Diabetics.Age, Diabetics.Glucose, color="red", label="Diabet", alpha = 0.4)
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.legend()
plt.show()


#When values divided far from too many distances , it harder to create euclid's triangle method
# so we using the normalization method.


y = data.Outcome.values
x_basedata = data.drop(["Outcome"],axis=1)   
# Rebuilding outcome axis to dependent variable because willing normalizate the other values
# its change the values beetween 0 and 1 


x = (x_basedata - np.min(x_basedata))/(np.max(x_basedata)-np.min(x_basedata))


print("The Values before the normalization:\n")
print(x_basedata.head())



print("\n\n\n The values after normalization for the KNN Train-Test:\n")
print(x.head())
    

# Dividing train data and test data
# Train Data --> Detection (sick)patients and healthy people with machine learning
# Test Data --> Our machine learn detect %? with truth ?

#Dividing test and train values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state=1)

#Classification of KNN model.
counter = 1
for k in range(1,11):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train,y_train)
    print(counter, "  ", "Accuracy Rate: %", knn.score(x_test,y_test)*100)
    counter += 1