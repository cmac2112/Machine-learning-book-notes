import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")

x = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

for i in range(len(x)):
    print(x[i])
#visualize
lifesat.plot(kind='scatter', grid=True, x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 100_500, 4, 9])
plt.show()


#create a model
model = LinearRegression()
#train model
model.fit(x, y)

#visualize the prediction points created by the mode

x_line = np.array([[23_500],[100_500]])
y_line = model.predict(x_line)
plt.plot(x_line, y_line, 'r-', linewidth=2, label='Linear Regression Line')

#make prediction using cyprus gdp per capita
x_new = [[37_655.2]] 
print(model.predict(x_new))
plt.savefig('plt.png')
plt.show()