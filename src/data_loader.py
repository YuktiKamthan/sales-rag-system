"""
Data Loader Module
------------------
This module loads and preprocesses sales data for the RAG pipeline.
It handles data cleaning, feature engineering, and preparation for embeddings.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SalesDataLoader:
    """
    Handles loading and preprocessing of sales data.
    
    Think of this as your data chef - it takes raw ingredients (sales data)
    and prepares them for the pipeline to consume.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the data loader.
        
        Args:
            file_path: Path to the Excel file with sales data
        """
        self.file_path = file_path
        self.df = None
        self.processed_df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the Excel file into a pandas DataFrame.
        
        Returns:
            Raw DataFrame with sales data
        """
        logger.info(f"Loading data from {self.file_path}")
        self.df = pd.read_excel(self.file_path)
        logger.info(f"Loaded {len(self.df)} records")
        return self.df
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the data by handling missing values and outliers.
        
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        
        # Remove rows where Amount is missing
        self.df = self.df.dropna(subset=['Amount'])
        
        # Convert Date to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(self.df['Date']):
            self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Sort by date for time series analysis
        self.df = self.df.sort_values('Date')
        
        logger.info(f"After cleaning: {len(self.df)} records")
        return self.df
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Create useful features from the data.
        
        This adds context that helps the RAG system understand patterns:
        - Year, Month, Day of week
        - Revenue metrics
        - Seasonal indicators
        
        Returns:
            DataFrame with additional features
        """
        logger.info("Engineering features...")
        
        # Time-based features
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Quarter'] = self.df['Date'].dt.quarter
        self.df['DayOfWeek'] = self.df['Date'].dt.day_name()
        self.df['WeekOfYear'] = self.df['Date'].dt.isocalendar().week
        
        # Revenue metrics
        self.df['Revenue'] = self.df['Amount']
        self.df['UnitsSold'] = self.df['Qty']
        
        # Season mapping
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                     3: 'Spring', 4: 'Spring', 5: 'Spring',
                     6: 'Summer', 7: 'Summer', 8: 'Summer',
                     9: 'Fall', 10: 'Fall', 11: 'Fall'}
        self.df['Season'] = self.df['Month'].map(season_map)
        
        return self.df
    
    def create_text_descriptions(self) -> pd.DataFrame:
        """
        Create natural language descriptions of each transaction.
        
        This is KEY for RAG! We convert structured data into text
        that can be embedded and searched semantically.
        
        Returns:
            DataFrame with 'text_description' column
        """
        logger.info("Creating text descriptions for RAG...")
        
        def create_description(row):
            """Create a readable description of each sale."""
            # Get price per unit - handle both column names
            price_per_unit = row.get('Price_per_Unit', row.get('Sales Price', 0))
            
            # Build description with available columns
            desc_parts = [
                f"Sale Transaction:",
                f"- Product SKU: {row['SKU']}",
                f"- Date: {row['Date'].strftime('%Y-%m-%d')}"
            ]
            
            # Add optional fields if they exist
            if 'DayOfWeek' in row:
                desc_parts[2] += f" ({row['DayOfWeek']})"
            
            if 'Season' in row and 'Year' in row and 'Quarter' in row:
                desc_parts.append(f"- Time Period: {row['Season']} {row['Year']}, Quarter {row['Quarter']}")
            elif 'Year' in row:
                desc_parts.append(f"- Year: {row['Year']}")
            
            # Add product info if available
            if 'Product' in row:
                desc_parts.append(f"- Product: {row['Product']}")
            
            desc_parts.extend([
                f"- Customer: {row['Name']}",
                f"- Quantity Sold: {row['Qty']} units",
                f"- Price per Unit: ${price_per_unit:.2f}",
                f"- Total Amount: ${row['Amount']:.2f}"
            ])
            
            # Add type if it exists
            if 'Type' in row:
                desc_parts.append(f"- Type: {row['Type']}")
            
            return "\n".join(desc_parts)
        
        self.df['text_description'] = self.df.apply(create_description, axis=1)
        return self.df
    
    def get_aggregated_stats(self) -> Dict:
        """
        Get summary statistics for monitoring and validation.
        
        Returns:
            Dictionary with key statistics
        """
        stats = {
            'total_records': len(self.df),
            'date_range': {
                'start': self.df['Date'].min().strftime('%Y-%m-%d'),
                'end': self.df['Date'].max().strftime('%Y-%m-%d')
            },
            'total_revenue': self.df['Amount'].sum(),
            'average_transaction': self.df['Amount'].mean(),
            'unique_products': self.df['SKU'].nunique(),
            'unique_customers': self.df['Name'].nunique()
        }
        return stats
    
    def prepare_for_rag(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete preprocessing pipeline for RAG.
        
        This runs all steps in sequence:
        1. Load data
        2. Clean data
        3. Engineer features
        4. Create text descriptions
        
        Returns:
            Tuple of (processed DataFrame, statistics)
        """
        self.load_data()
        self.clean_data()
        self.engineer_features()
        self.create_text_descriptions()
        
        stats = self.get_aggregated_stats()
        
        logger.info("Data preparation complete!")
        logger.info(f"Stats: {stats}")
        
        self.processed_df = self.df
        return self.processed_df, stats


if __name__ == "__main__":
    # Example usage
    loader = SalesDataLoader("data/sales_data.xlsx")
    df, stats = loader.prepare_for_rag()
    print("\nData prepared successfully!")
    print(f"\nSample text description:\n{df['text_description'].iloc[0]}")