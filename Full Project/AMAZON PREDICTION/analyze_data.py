import pandas as pd
import os

def analyze_all_sales():
    files = {
        'Main': 'c:/Full Project/AMAZON PREDICTION/data/amazon_sales_cleaned.csv',
        'Electronics': 'c:/Full Project/AMAZON PREDICTION/data/electronics_cleaned.csv',
        'Grocery': 'c:/Full Project/AMAZON PREDICTION/data/grocery_sales_data_cleaned.csv'
    }

    for name, file_path in files.items():
        print(f"\n{'='*20} {name} Analysis {'='*20}")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        try:
            # Read only first 10000 rows for speed if file is huge, or full if reasonable
            if name == 'Electronics': # This file was 300MB+, might be slow
                df = pd.read_csv(file_path, nrows=50000) 
                print("Loaded partial data for performance.")
            elif name == 'Grocery': # This file was 1GB+, definitely partial
                df = pd.read_csv(file_path, nrows=50000)
                print("Loaded partial data for performance.")
            else:
                df = pd.read_csv(file_path) # Main is 22MB, fine to load all
                
        except Exception as e:
            print(f"Error loading CSV: {e}")
            continue

        if name == 'Main':
            # ... existing logic ...
            if df['Amount'].dtype == 'object':
                 df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            total_rev = df['Amount'].sum()
            print(f"Total Revenue: {total_rev}")
            print(f"Avg Order Value: {total_rev/len(df):.2f}")
            print("Top Categories:", df['Category'].value_counts().head(3).index.tolist())

        elif name == 'Electronics':
            # Columns: item_id, user_id, rating, timestamp, category, brand, year...
            # KPIs: Avg Rating, Top Brand, Top Category by Rating Count
            print(f"Avg Rating: {df['rating'].mean():.2f}")
            print(f"Total Ratings: {len(df)}")
            if 'category' in df.columns:
                print("Top Categories:", df['category'].value_counts().head(3).index.tolist())
            if 'brand' in df.columns:
                print("Top Brands:", df['brand'].value_counts().head(3).index.tolist())

        elif name == 'Grocery':
            # Columns: Sales Amount, Sales Quantity, Item, ...
            # KPIs: Total Sales, Avg Sales per Txn, Top Items
            if 'Sales Amount' in df.columns:
                total_sales = df['Sales Amount'].sum()
                print(f"Total Sales (Sample): {total_sales}")
                print(f"Avg Ticket: {total_sales/len(df):.2f}")
            if 'Item' in df.columns:
                print("Top Items:", df['Item'].value_counts().head(3).index.tolist())
            if 'Sales Quantity' in df.columns:
                print(f"Total Quantity (Sample): {df['Sales Quantity'].sum()}")

if __name__ == "__main__":
    analyze_all_sales()
