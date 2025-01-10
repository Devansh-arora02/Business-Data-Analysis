import pandas as pd
import joblib
import logging
import os
import sys

directory = r"" # Model Path

if not os.path.exists(directory):
    raise Exception(f"Directory not found")

log_file = os.path.join(directory, "discount_file.log")

# Configure logging
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,  # Log level (INFO, DEBUG, ERROR, etc.)
    format='%(message)s'
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

def simulate_discount_per_product(input_df, model, discount_mapping):
    
    input_df['BillTime'] = pd.to_datetime(input_df['BillTime'], format='%I:%M %p', errors='coerce')
    input_df['Hour'] = input_df['BillTime'].dt.hour
    input_df['DayOfWeek'] = input_df['BillTime'].dt.dayofweek
    input_df['IsWeekend'] = (input_df['BillTime'].dt.weekday >= 5).astype(int)
    
    original_profit = input_df['Profit'].sum()
    profit_results = []
    for product in input_df['ProductName'].unique():
        df = input_df.copy()
        for discount in discount_mapping:
            # Apply discount individually to each product
            df.loc[df['ProductName'] == product,'ProductLevelDisc%'] = discount  # Apply the discount percentage to the column
            df.loc[df['ProductName'] == product, 'ProductLevelDiscAmount'] = df.loc[df['ProductName'] == product, 'MRP'] * (discount / 100)  # Calculate discount amount
            df.loc[df['ProductName'] == product, 'DiscountedMRP'] = df.loc[df['ProductName'] == product, 'MRP'] - df.loc[df['ProductName'] == product, 'ProductLevelDiscAmount']  # Calculate discounted MRP
            df.loc[df['ProductName'] == product, 'Amount'] = df.loc[df['ProductName'] == product, 'DiscountedMRP'] * 1.36  # Final amount after tax

            # Select the features that match the training model
            features = ['StoreName', 'MRP', 'ProductLevelDisc%', 'ProductLevelDiscAmount', 'ProductName', 'New customer Type', 'Hour', 'DayOfWeek', 'IsWeekend', 'Amount']
            X = df[features]

            # Get predicted profit from the model
            predicted_profit = model.predict(X)[:, 0].sum()


            # Calculate percentage increase in profit
            if original_profit != 0:
                percentage_increase = ((predicted_profit - original_profit) / original_profit) * 100
            else:
                percentage_increase = 0

            # Store results for the current discount
            profit_results.append({
                'Product': product,
                'ProfitWithDiscount': f'{predicted_profit:.3f}',
                'OriginalProfit': f'{original_profit:.3f}',
                '%Increase': f'{percentage_increase:.3f}'
            })

    # Convert results to DataFrame
    profit_results_df = pd.DataFrame(profit_results)

    return profit_results_df


file_paths = ['combined_files.csv', 'Cluster_0_Dataset.csv', 'Cluster_1_Dataset.csv', 'Cluster_2_Dataset.csv']
discount_mapping = [0, 5, 10, 15, 20, 25, 30, 35, 40, 100]  # Example of discount percentages

for mapping in discount_mapping:
    print(f'\n{"="*30}\n')
    print(f'Discount Percentage : {mapping}')
    for df in file_paths:
        if '_Dataset.csv' in df:
            folder_name = df.replace('_Dataset.csv', '')
            file_name = f"{folder_name.replace('_',' ')}_model.pkl"
        else:
            folder_name = 'Overall'
            file_name = 'Overall_model.pkl'
        
        print(f'{folder_name} Analysis - ')
        model_path = f'{directory}\\{folder_name}\\{file_name}'
        
        # Load the dataset
        df = pd.read_csv(df)
        
        # Load the model
        model = joblib.load(model_path)
        
        # Call the simulate_discount_per_product function and output results
        profit_results_df = simulate_discount_per_product(df, model, [mapping])
        profit_result_df = profit_results_df.sort_values(by = '%Increase', ascending = False)
        
        # Print the results for this discount level
        print(profit_result_df.head(10))

        print(f'\n{"-"*30}\n')
