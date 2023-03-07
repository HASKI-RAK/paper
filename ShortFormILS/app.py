import pandas as pd
import statsmodels.api as sm
from stepwise_regression import step_reg
from pprint import pprint
from sklearn import linear_model


# Load data
# file = "./Lifeexpetancy.csv"
file = "./D1SMRC.xlsx"
# df = pd.read_csv(file, sep=",")
df = pd.read_excel(file, sheet_name="HASKI Pretest (ILS, LIST-K & BF")
pprint(df.head())

reg = linear_model.LinearRegression()

# Value Wert can be max 21
# Select every column except the last one
X = df.iloc[:, :-1]
# add constant
# Select the last column
y = df.iloc[:, -1]
pprint(X.head())
pprint(y.head())
# Add constant
X = sm.add_constant(X)

# Define the model
model = sm.OLS(y, X)
# Fit the model
results = model.fit()
# Print the summary
print(results.summary())
backselect = step_reg.forward_regression(X, y, 0.05,verbose=False)
print(backselect)