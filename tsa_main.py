import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import holidays

#Load Data
train = pd.read_csv("tsa_train.csv")
test = pd.read_csv("tsa_test.csv")

#converting Date column to Datetime
train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])

train = train.sort_values('Date')
test = test.sort_values('Date')

#feature engineering
def create_time_features(df):
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['year'] = df['Date'].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    return df

def add_holiday_feature(df):
    us_holidays = holidays.US()
    df['is_holiday'] = df['Date'].isin(us_holidays).astype(int)
    return df

train = create_time_features(train)
test = create_time_features(test)

train = add_holiday_feature(train)
test = add_holiday_feature(test)

#feature columns
features = [
    "day_of_week",
    "month",
    "day_of_year",
    "year",
    "is_weekend",
    "is_holiday"
]

target = "Volume"

X_train = train[features]
y_train = train[target]

X_test = test[features]

#train model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train,y_train)

print("Model training complete")

#predictions
predictions = model.predict(X_test)

test["prediction"] = predictions

print("Forecasting complete")

#plot predictions
plt.figure(figsize=(12, 6))
plt.plot(test["Date"], test["prediction"], label="Predicted")

if "Volume" in test.columns:
    plt.plot(test["Date"], test["Volume"], label="Actual")

plt.title("TSA Passenger Forecast - Beginner Model (No Recursion)")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.legend()
plt.tight_layout()
plt.show()

#evaluating model
if "Volume" in test.columns:
    mae = mean_absolute_error(test['Volume'], test["prediction"])

    print(f"Test MAE: {mae:,.2f}")
