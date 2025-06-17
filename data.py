import pandas as pd #for CSV's
import numpy as np # used for tensors in our data model
import torch as pyt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# these libraries help us make plots
import matplotlib.pyplot as plt
import seaborn as sns

# ML model FEATURES: Location, Date+Time, Temp(C), Humidity, Precip(mm), Wind (km/h)
# Cities: ['San Diego' 'Philadelphia' 'San Antonio' 'San Jose' 'New York' 'Houston'
# 'Dallas' 'Chicago' 'Los Angeles' 'Phoenix']

dataframe = pd.read_csv('weather_data.csv')
cities_list = dataframe['Location'].unique()
dates_list = dataframe['Date_Time']
numerical_cols = [col for col in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[col].dtype)]
curr_city_data = []


def main():
    """To Display the cleaned up data (for now)
    """
    cleanData()
    
    print("Hello! Welcome to MeteorSpeculator\n")
    print("There are 2 modes you can use for weather prediction: Linear Regression or RNN\n")
    print("With the Lin Regression mode, you can predict one unknown weather condition given all the other features.\n")
    print("And with RNN mode, you can view a prediction of tomorrow's likely conditions.\n")
    
    
    choice = input("Enter your choice (Linear or RNN): ")
    
    while choice != "Linear" or choice != "RNN":
        if choice == "Linear":
            print("City choices: {}".format(cities_list))
            city_choice = input("Enter your city choice: ")
            print("\nWeather conditions you can predict:{}".format(numerical_cols))
            unknownVar = input("Enter the weather condition you'd like to estimate: ")
            lin_model = train_linreg_model(city=city_choice, unknownVariable=unknownVar)
            print("\nNow, you will need to enter the numerical values of the other conditions:")
            X = [col for col in numerical_cols if col != unknownVar]
            feature_vals = [[]]
            for feature in X:
                # Gotta check if the features are appending the right order
                val = float(input(f"{feature}: "))
                feature_vals[0].append(val)
            vals_dataframe = pd.DataFrame(feature_vals,columns=X)
            print("\nPredicted value for {}: {}".format(unknownVar, lin_model.predict(vals_dataframe)[0]))
            
            # Plot the data:
            # display_data = sns.load_dataset('weather_data.csv')
            sns.scatterplot(data=curr_city_data, x=X[0], y=X[1], hue=X[2], size=unknownVar)
            
            
            # sns.lmplot(x=X[0], y=X[1], hue=unknownVar, data=curr_city_data) 

            plt.show()
            break
        elif choice == "RNN":
            print("Not completed yet!")
            break
        else:
            choice = input("Please enter Linear or RNN to continue: ")

    # print(lin_model.predict())
    # Afterwards, display the lin model using seaborn and matplot lib
    
    
    
    
    
def cleanData():
    global dataframe
    global cities_list
    
    # Remove null/nonsense values
    dataframe.dropna(how='any', inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    
    # Remove Duplicates
    dataframe.drop_duplicates(inplace=True) # entire row's value must be same to be removed
    dataframe.reset_index(drop=True, inplace=True)
    
    # Sort by city
    dataframe.sort_values(inplace=True, by='Location')
    dataframe.reset_index(drop=True, inplace=True)
    
    # Convert Dtype of 'Location' col to str
    dataframe['Location'] = dataframe['Location'].astype(str)
    
    # Convert Dtype of Date_Time to a Date/Time class in python as needed
        
    
    # Scaling (to unit variance) - will add this if needed
    
def removeRow(row: int):
    dataframe.drop(row, inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    
# For now, use Lin Reg. If time, then try KNN, RNN, or Random Forest (time not needed for now)
def train_linreg_model(city, unknownVariable):
    assert city in cities_list
    assert str(unknownVariable) in numerical_cols # unknown can't be non numerical
    
    global curr_city_data
    
    start_index = dataframe.loc[dataframe['Location'] == city].index[0]
    end_index = dataframe.loc[dataframe['Location'] == city].index[-1]
    city_data = dataframe.iloc[start_index:end_index + 1]
    
    curr_city_data = city_data
    
    X = city_data[numerical_cols].drop(columns=unknownVariable)
    y = city_data[unknownVariable]
    
    # print("X dataframe:")
    # X.info()
    
    # print("Y columns:", y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("R^2 value for the test data:", model.score(X_test, y_test))
    # print("Coefficients used in model equation:", model.coef_)
    
    # Display the regression
    
    return model

def RNNModel():
    """Intended to have a higher accuracy than the linreg_model and predict all other 
    values based on time and location"""
    # Couldn't get to this—hopefully can finish in the future!
    pass

main()