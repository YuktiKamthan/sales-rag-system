"""
Sales Predictor Module
----------------------
Time series forecasting for sales prediction.

Uses Prophet for robust forecasting with seasonality detection.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
import logging
from .telemetry import telemetry, metrics
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SalesPredictor:
    """
    Predicts future sales using historical data.
    
    This uses Facebook Prophet which handles:
    - Seasonality (monthly, yearly patterns)
    - Trends (growing or declining)
    - Holidays and special events
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the predictor.
        
        Args:
            df: DataFrame with sales data (must have Date and Amount columns)
        """
        self.df = df.copy()
        self.model = None
        self.forecast_result = None
        
        logger.info("Sales Predictor initialized")
    
    def prepare_data_for_prophet(self) -> pd.DataFrame:
        """
        Prepare data in Prophet's required format.
        
        Prophet needs:
        - 'ds' column (date)
        - 'y' column (value to predict)
        
        Returns:
            DataFrame ready for Prophet
        """
        # Aggregate sales by date
        daily_sales = self.df.groupby('Date')['Amount'].sum().reset_index()
        
        # Rename columns for Prophet
        daily_sales = daily_sales.rename(columns={
            'Date': 'ds',
            'Amount': 'y'
        })
        
        logger.info(f"Prepared {len(daily_sales)} daily records for forecasting")
        
        return daily_sales
    
    @telemetry.trace_operation("train_model")
    def train(self, 
             seasonality_mode: str = 'multiplicative',
             changepoint_prior_scale: float = 0.05,
             yearly_seasonality: bool = True,
             weekly_seasonality: bool = True,
             daily_seasonality: bool = False) -> None:
        """
        Train the forecasting model.
        
        Args:
            seasonality_mode: 'additive' or 'multiplicative'
            changepoint_prior_scale: Flexibility of trend (0.001-0.5)
            yearly_seasonality: Include yearly patterns
            weekly_seasonality: Include weekly patterns
            daily_seasonality: Include daily patterns
        """
        start_time = time.time()
        
        try:
            # Prepare data
            prophet_df = self.prepare_data_for_prophet()
            
            # Create and configure model
            self.model = Prophet(
                seasonality_mode=seasonality_mode,
                changepoint_prior_scale=changepoint_prior_scale,
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality
            )
            
            # Train model
            logger.info("Training forecasting model...")
            self.model.fit(prophet_df)
            
            duration = time.time() - start_time
            logger.info(f"Model trained in {duration:.2f} seconds")
            metrics.record('prediction_training', duration, success=True)
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            metrics.record('prediction_training', time.time() - start_time, success=False)
            raise
    
    @telemetry.trace_operation("generate_forecast")
    def forecast(self, periods: int = 30, freq: str = 'D') -> pd.DataFrame:
        """
        Generate future predictions.
        
        Args:
            periods: How many periods into the future
            freq: Frequency ('D'=daily, 'W'=weekly, 'M'=monthly)
            
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        start_time = time.time()
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(
                periods=periods,
                freq=freq
            )
            
            # Generate forecast
            self.forecast_result = self.model.predict(future)
            
            duration = time.time() - start_time
            logger.info(f"Forecast generated for {periods} periods in {duration:.2f} seconds")
            metrics.record('prediction_generation', duration, success=True)
            
            return self.forecast_result
            
        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            metrics.record('prediction_generation', time.time() - start_time, success=False)
            raise
    
    def get_future_predictions(self, num_days: int = 30) -> pd.DataFrame:
        """
        Get only the future predictions (not historical).
        
        Args:
            num_days: Number of days to predict
            
        Returns:
            DataFrame with future predictions only
        """
        if self.forecast_result is None:
            raise ValueError("No forecast available. Call forecast() method first.")
        
        # Get only future dates
        last_date = self.df['Date'].max()
        future_forecast = self.forecast_result[self.forecast_result['ds'] > last_date].copy()
        
        # Rename columns for clarity
        future_forecast = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        future_forecast.columns = ['Date', 'Predicted_Sales', 'Lower_Bound', 'Upper_Bound']
        
        return future_forecast
    
    def plot_forecast(self, save_path: Optional[str] = None) -> None:
        """
        Visualize the forecast.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.forecast_result is None:
            raise ValueError("No forecast available. Call forecast() first.")
        
        # Create plot
        fig = self.model.plot(self.forecast_result, figsize=(12, 6))
        plt.title('Sales Forecast', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Sales Amount ($)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.close()
        
        return fig
    
    def plot_components(self, save_path: Optional[str] = None) -> None:
        """
        Plot forecast components (trend, seasonality).
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.forecast_result is None:
            raise ValueError("No forecast available. Call forecast() first.")
        
        fig = self.model.plot_components(self.forecast_result, figsize=(12, 8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Components plot saved to {save_path}")
        
        plt.close()
        
        return fig
    
    def get_metrics(self, test_size: int = 30) -> Dict:
        """
        Calculate prediction accuracy metrics.
        
        Args:
            test_size: Number of recent days to use as test set
            
        Returns:
            Dictionary with MAE, RMSE, MAPE
        """
        # Prepare data
        prophet_df = self.prepare_data_for_prophet()
        
        # Split into train/test
        train_df = prophet_df[:-test_size]
        test_df = prophet_df[-test_size:]
        
        # Train model on training data
        temp_model = Prophet(
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )
        temp_model.fit(train_df)
        
        # Predict on test period
        test_forecast = temp_model.predict(test_df)
        
        # Calculate metrics
        actual = test_df['y'].values
        predicted = test_forecast['yhat'].values
        
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        metrics_dict = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'test_size': test_size
        }
        
        logger.info(f"Model Metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
        
        return metrics_dict
    
    def get_summary(self) -> Dict:
        """
        Get a summary of predictions and insights.
        
        Returns:
            Dictionary with key insights
        """
        if self.forecast_result is None:
            raise ValueError("No forecast available. Call forecast() first.")
        
        future_predictions = self.get_future_predictions(30)
        
        summary = {
            'next_30_days_total': future_predictions['Predicted_Sales'].sum(),
            'daily_average': future_predictions['Predicted_Sales'].mean(),
            'peak_day': {
                'date': future_predictions.loc[future_predictions['Predicted_Sales'].idxmax(), 'Date'],
                'amount': future_predictions['Predicted_Sales'].max()
            },
            'low_day': {
                'date': future_predictions.loc[future_predictions['Predicted_Sales'].idxmin(), 'Date'],
                'amount': future_predictions['Predicted_Sales'].min()
            }
        }
        
        return summary


if __name__ == "__main__":
    # Test the predictor
    print("Testing Sales Predictor...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    np.random.seed(42)
    amounts = np.random.rand(len(dates)) * 1000 + np.sin(np.arange(len(dates)) / 30) * 500
    
    test_df = pd.DataFrame({
        'Date': dates,
        'Amount': amounts
    })
    
    # Create predictor
    predictor = SalesPredictor(test_df)
    
    # Train
    predictor.train()
    
    # Forecast
    predictor.forecast(periods=30)
    
    # Get summary
    summary = predictor.get_summary()
    print(f"\nForecast Summary: {summary}")