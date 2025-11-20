"""
Mock Sales Data Generator
-------------------------
Generates realistic synthetic sales data for demo purposes.

Usage:
    python generate_mock_data.py

This will create data/sales_data.xlsx with synthetic transactions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

def generate_mock_sales_data(num_records=10000):
    """
    Generate realistic mock sales data.
    
    Args:
        num_records: Number of sales transactions to generate
        
    Returns:
        DataFrame with mock sales data
    """
    print(f"🎨 Generating {num_records:,} mock sales records...")
    
    # Product categories and base prices
    categories = {
        'Electronics': {'base_price': 200, 'variance': 100, 'skus': 1000},
        'Clothing': {'base_price': 50, 'variance': 30, 'skus': 800},
        'Home & Garden': {'base_price': 80, 'variance': 50, 'skus': 600},
        'Books': {'base_price': 25, 'variance': 15, 'skus': 1200},
        'Sports': {'base_price': 100, 'variance': 60, 'skus': 500},
        'Toys': {'base_price': 40, 'variance': 25, 'skus': 700},
    }
    
    # Generate company names (simple mock companies)
    company_prefixes = ['Global', 'Tech', 'Smart', 'Prime', 'Alpha', 'Beta', 
                       'Delta', 'Omega', 'Star', 'Sky', 'Blue', 'Green']
    company_suffixes = ['Corp', 'Inc', 'LLC', 'Ltd', 'Solutions', 'Systems',
                       'Industries', 'Enterprises', 'Group', 'Partners']
    companies = [f"{random.choice(company_prefixes)} {random.choice(company_suffixes)}" 
                 for _ in range(50)]
    
    # Date range: Last 5 years
    start_date = datetime.now() - timedelta(days=5*365)
    end_date = datetime.now()
    
    records = []
    
    for i in range(num_records):
        # Random category
        category = random.choice(list(categories.keys()))
        cat_info = categories[category]
        
        # Generate SKU
        sku = random.randint(10000, 10000 + cat_info['skus'])
        
        # Random date with seasonal patterns
        days_offset = random.randint(0, (end_date - start_date).days)
        date = start_date + timedelta(days=days_offset)
        
        # Seasonal multiplier (higher in Q4 for holidays)
        month = date.month
        if month in [11, 12]:  # Black Friday, Christmas
            seasonal_multiplier = 1.5
        elif month in [6, 7]:  # Summer
            seasonal_multiplier = 1.2
        else:
            seasonal_multiplier = 1.0
        
        # Price with some randomness
        base_price = cat_info['base_price']
        variance = cat_info['variance']
        price_per_unit = max(5, base_price + np.random.normal(0, variance))
        
        # Quantity sold (influenced by price)
        if price_per_unit < 50:
            qty = random.randint(1, 500)  # Cheap items sell more
        elif price_per_unit < 100:
            qty = random.randint(1, 200)
        else:
            qty = random.randint(1, 100)
        
        # Apply seasonal multiplier
        qty = int(qty * seasonal_multiplier)
        
        # Total amount
        amount = price_per_unit * qty
        
        # Random customer
        customer = random.choice(companies)
        
        # Product name
        product_names = {
            'Electronics': ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Camera', 'Monitor'],
            'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Shoes', 'Dress', 'Sweater'],
            'Home & Garden': ['Lamp', 'Chair', 'Table', 'Vase', 'Plant', 'Cushion'],
            'Books': ['Novel', 'Textbook', 'Cookbook', 'Biography', 'Guide', 'Manual'],
            'Sports': ['Basketball', 'Yoga Mat', 'Dumbbells', 'Tennis Racket', 'Bike', 'Shoes'],
            'Toys': ['Action Figure', 'Board Game', 'Puzzle', 'Doll', 'Building Blocks', 'Car'],
        }
        product = random.choice(product_names[category])
        
        record = {
            'Date': date.strftime('%Y-%m-%d'),
            'SKU': sku,
            'Product': f"{category} - {product}",
            'Name': customer,  # Customer name
            'Qty': qty,
            'Amount': round(amount, 2),
            'Price_per_Unit': round(price_per_unit, 2),
        }
        
        records.append(record)
        
        if (i + 1) % 2000 == 0:
            print(f"  ✓ Generated {i + 1:,} records...")
    
    df = pd.DataFrame(records)
    
    # Sort by date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"\n✅ Generated {len(df):,} records!")
    print(f"📊 Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"💰 Total revenue: ${df['Amount'].sum():,.2f}")
    print(f"🏢 Unique customers: {df['Name'].nunique()}")
    print(f"📦 Unique products: {df['SKU'].nunique()}")
    
    return df


def main():
    """Generate and save mock data."""
    print("🚀 Mock Sales Data Generator\n")
    print("=" * 60)
    
    # Generate data
    df = generate_mock_sales_data(num_records=10000)
    
    # Save to Excel
    output_path = 'data/sales_data.xlsx'
    print(f"\n💾 Saving to {output_path}...")
    df.to_excel(output_path, index=False, sheet_name='Sales')
    
    print(f"\n✅ Done! Mock data saved to {output_path}")
    print("\n📋 Sample records:")
    print(df.head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("🎉 Ready to run the RAG system!")
    print("\nNext steps:")
    print("1. streamlit run web_ui.py")
    print("2. Click 'Initialize System'")
    print("3. Start asking questions!")


if __name__ == "__main__":
    main()