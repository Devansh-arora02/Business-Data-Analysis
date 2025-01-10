import pandas as pd

import joblib

import os
import logging
import sys
from Discount_mapping import create_discount_mapping

path = '' # Model Path
directory = f"{path}" 

if not os.path.exists(directory):
    raise Exception(f"Directory {path} not found")

log_file = os.path.join(directory, "test_file.log")

# Configure logging
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,  # Log level (INFO, DEBUG, ERROR, etc.)
    format='%(asctime)s - %(message)s'
)

# Redirect print statements to the log file
class PrintLogger:
    def write(self, message):
        if message.strip() != "":
            logging.info(message.strip())
    def flush(self):
        pass

sys.stdout = PrintLogger()  # Redirect stdout (print output)
sys.stderr = PrintLogger()  # Redirect stderr (error output)
print('Testing Results : ')

# Function to apply customized discounts
def apply_custom_discounts(original_data, products_data, discount_mapping):
    # Copy the original dataset
    discounted_data = original_data.copy()

    discounted_data['BillTime'] = pd.to_datetime(discounted_data['BillTime'], format='%I:%M %p', errors='coerce')
    discounted_data['Hour'] = discounted_data['BillTime'].dt.hour
    discounted_data['DayOfWeek'] = discounted_data['BillTime'].dt.dayofweek
    discounted_data['IsWeekend'] = (discounted_data['BillTime'].dt.weekday >= 5).astype(int)

    # Iterate through the discount mapping and apply discounts
    for product, discount in discount_mapping.items():
        MRP = products_data.loc[products_data['ProductName'] == product, 'Mean_MRP']
        discounted_data.loc[discounted_data['ProductName'] == product, 'ProductLevelDisc%'] = discount
        discounted_data.loc[discounted_data['ProductName'] == product, 'ProductLevelDiscAmount'] = MRP * (discount / 100)
        discounted_data.loc[discounted_data['ProductName'] == product, 'BaseValue'] = (MRP * (1 - (discount / 100)))
        discounted_data.loc[discounted_data['ProductName'] == product, 'SGST'] = discounted_data.loc[discounted_data['ProductName'] == product, 'BaseValue']*0.18
        discounted_data.loc[discounted_data['ProductName'] == product, 'CGST'] = discounted_data.loc[discounted_data['ProductName'] == product, 'BaseValue']*0.18
        discounted_data.loc[discounted_data['ProductName'] == product, 'Amount'] = discounted_data.loc[discounted_data['ProductName'] == product, 'BaseValue']*(1-2*0.18)
        discounted_data.loc[discounted_data['ProductName'] == product, 'MRP'] = MRP 


    # Set default discount to 0 for products not in the mapping
    discounted_data['ProductLevelDisc%'].fillna(0, inplace=True)
    discounted_data['ProductLevelDiscAmount'].fillna(0, inplace=True)

    return discounted_data

def predictor(df, products_data, discount_mapping, path):
    # Calculate original total profit
    df = pd.read_csv(df)

    original_total_profit = df['Profit'].sum()
    original_total_quantity = df['Quantity'].sum()

    # Create the dataset with the new discounts applied
    discounted_data = apply_custom_discounts(df, products_data, discount_mapping)

    # Load the trained model
    loaded_model = joblib.load(path)

    # Prepare the features for predictions (same as before)
    X_discounted = discounted_data[['StoreName', 'MRP', 'ProductLevelDisc%', 'ProductLevelDiscAmount', 'ProductName', 
                                    'ProductGroup', 'DeliveryType', 'New customer Type', 'Hour', 'DayOfWeek', 'IsWeekend', 'Amount']]

    # Make predictions with the loaded model to ensure consistent transformations
    profit_quantity_predictions = loaded_model.predict(X_discounted)

    quantity_predictions = profit_quantity_predictions[:, 1]  # Predictions for Quantity

    # Print the results
    print(f"Original Total Profit: {original_total_profit}")

    # Calculate theoretical profit generated with discounted prices and predicted quantities
    discounted_data['PredictedQuantity'] = quantity_predictions  # Add the predicted quantities to the dataframe

    # Calculate profit per unit for each product after applying discount
    discounted_data['UnitProfit'] = discounted_data['Amount'] - discounted_data['BaseValue']

    # Calculate total profit for each row by multiplying unit profit by predicted quantity
    discounted_data['TheoreticalProfit'] = discounted_data['UnitProfit'] * discounted_data['PredictedQuantity']

    # Calculate the total theoretical profit
    total_theoretical_profit = discounted_data['TheoreticalProfit'].sum()

    percentage_increase_profit_th = ((total_theoretical_profit - original_total_profit) / original_total_profit) * 100

    print(f"Theoretical Profit with Predicted Quantities and Customized Discounts: {total_theoretical_profit}")
    print(f"Theoretical Percentage Increase in Profit: {percentage_increase_profit_th:.2f}%")


    # Additionally, analyze predicted quantities
    new_total_quantity = round(quantity_predictions.sum())

    percentage_increase_quantity = ((new_total_quantity - original_total_quantity) / original_total_quantity) * 100

    print(f"Original Total Quantity: {original_total_quantity}")
    print(f"Total Predicted Quantity with Customized Discounts: {new_total_quantity}")
    print(f"Expected Percentage Increase in Quantity: {percentage_increase_quantity:.2f}%")

products_data = pd.read_csv('product_sales_summary.csv')

insights = pd.read_csv('insights_2.csv')
discount = create_discount_mapping(insights)
discount_mapping = [discount]
# for index, row in insights.iterrows():
#     discount_mapping.append(create_discount_mapping(row))
file_paths = ['combined_files.csv', 'Cluster_0_Dataset.csv', 'Cluster_1_Dataset.csv', 'Cluster_2_Dataset.csv']
for mapping in discount_mapping:
    print(f'\n{"="*30}\n')
    print('Discount Mapping : ')
    print(mapping)
    for df in file_paths:
        if '_Dataset.csv' in df:
            folder_name = df.replace('_Dataset.csv', '')
            file_name = f"{folder_name.replace('_',' ')}_model.pkl"
        else:
            folder_name = 'Overall'
            file_name = 'Overall_model.pkl'
        print(f'{folder_name} Analysis - ')
        model_path = f'{directory}\\{folder_name}\\{file_name}'
        predictor(df, products_data, mapping, model_path)
        print(f'\n{"-"*30}\n')
