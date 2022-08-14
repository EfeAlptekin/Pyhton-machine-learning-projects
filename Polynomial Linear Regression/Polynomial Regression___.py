import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("polynomial.csv",sep = ";")

# call the PolynomialFeatures function to create a polynomial regression object
# When calling this function, we specify the degree (N) of the polynomial:
polynomial_regression = PolynomialFeatures(degree = 4)

x_polynomial = polynomial_regression.fit_transform(df[['deneyim']])

#Create the reg object
#And using its fit method,so we train our regression model with available real data
reg = LinearRegression()
reg.fit(x_polynomial,df['maas'])


y_head = reg.predict(x_polynomial)



plt.plot(df['deneyim'],y_head,color="green",label="polynomial regression")
plt.legend()
plt.scatter(df['deneyim'],df['maas'])
plt.show()

#Calculating the salaries.
x_polynomial1 = polynomial_regression.fit_transform([[7]])
reg.predict(x_polynomial1)
print(x_polynomial1)






