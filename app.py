"""
Sales RAG System - Main Application
====================================
Enterprise-level RAG pipeline with sales prediction and monitoring.

Author: Yukti
GitHub: [Your GitHub URL]
"""

import argparse
import sys
from src.rag_pipeline import SalesRAGPipeline
from src.predictor import SalesPredictor
from src.telemetry import telemetry, metrics
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_rag_demo():
    """Run interactive RAG demo."""
    print("\n" + "="*60)
    print("🚀 Sales RAG System - Interactive Demo")
    print("="*60 + "\n")
    
    # Initialize pipeline
    print("Initializing RAG pipeline...")
    pipeline = SalesRAGPipeline(
        data_path="data/sales_data.xlsx",
        collection_name="sales_data"
    )
    
    pipeline.initialize()
    
    print("\n✅ Pipeline ready!")
    print("\nYou can now ask questions about your sales data.")
    print("Examples:")
    print("- What were the total sales in 2024?")
    print("- Show me transactions for Product SKU 00010")
    print("- Which months had the highest revenue?")
    print("\nType 'exit' to quit\n")
    
    # Interactive loop
    while True:
        try:
            question = input("\n💬 Your question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            if not question:
                continue
            
            # Query the pipeline
            print("\n🔍 Searching...")
            result = pipeline.query(
                question=question,
                top_k=5,
                use_llm=True  # Set to False if no Gemini API key
            )
            
            print("\n" + "="*60)
            print("📊 Answer:")
            print("="*60)
            print(result['answer'])
            print(f"\n⏱️  Execution time: {result['execution_time']:.2f} seconds")
            print(f"📁 Retrieved {result['retrieved_records']} relevant records")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\n❌ Error: {e}")


def run_prediction_demo():
    """Run sales prediction demo."""
    print("\n" + "="*60)
    print("📈 Sales Prediction Demo")
    print("="*60 + "\n")
    
    # Initialize pipeline to get data
    pipeline = SalesRAGPipeline(
        data_path="data/sales_data.xlsx",
        collection_name="sales_data"
    )
    
    print("Loading data...")
    pipeline.data_loader.load_data()
    pipeline.data_loader.clean_data()
    pipeline.data_loader.engineer_features()
    
    # Create predictor
    print("\nTraining prediction model...")
    predictor = SalesPredictor(pipeline.data_loader.df)
    predictor.train()
    
    # Generate forecast
    print("\nGenerating 30-day forecast...")
    predictor.forecast(periods=30, freq='D')
    
    # Get summary
    summary = predictor.get_summary()
    
    print("\n" + "="*60)
    print("📊 Forecast Summary (Next 30 Days)")
    print("="*60)
    print(f"\n💰 Total Predicted Sales: ${summary['next_30_days_total']:,.2f}")
    print(f"📊 Daily Average: ${summary['daily_average']:,.2f}")
    print(f"\n📈 Peak Sales Day:")
    print(f"   Date: {summary['peak_day']['date']}")
    print(f"   Amount: ${summary['peak_day']['amount']:,.2f}")
    print(f"\n📉 Lowest Sales Day:")
    print(f"   Date: {summary['low_day']['date']}")
    print(f"   Amount: ${summary['low_day']['amount']:,.2f}")
    
    # Save plots
    print("\n📊 Generating visualization...")
    predictor.plot_forecast(save_path='outputs/forecast.png')
    predictor.plot_components(save_path='outputs/forecast_components.png')
    
    print("\n✅ Plots saved to outputs/ directory")
    
    # Get metrics
    print("\n🎯 Model Performance Metrics:")
    metrics_dict = predictor.get_metrics(test_size=30)
    print(f"   MAE:  ${metrics_dict['MAE']:,.2f}")
    print(f"   RMSE: ${metrics_dict['RMSE']:,.2f}")
    print(f"   MAPE: {metrics_dict['MAPE']:.2f}%")


def show_stats():
    """Show system statistics."""
    print("\n" + "="*60)
    print("📊 System Statistics")
    print("="*60 + "\n")
    
    pipeline = SalesRAGPipeline(
        data_path="data/sales_data.xlsx",
        collection_name="sales_data"
    )
    
    pipeline.initialize()
    stats = pipeline.get_stats()
    
    # Data stats
    print("📁 Data Statistics:")
    print(f"   Total Records: {stats['data']['total_records']:,}")
    print(f"   Date Range: {stats['data']['date_range']['start']} to {stats['data']['date_range']['end']}")
    print(f"   Total Revenue: ${stats['data']['total_revenue']:,.2f}")
    print(f"   Avg Transaction: ${stats['data']['average_transaction']:,.2f}")
    print(f"   Unique Products: {stats['data']['unique_products']:,}")
    print(f"   Unique Customers: {stats['data']['unique_customers']:,}")
    
    # Vector store stats
    print(f"\n🗄️  Vector Store:")
    print(f"   Collection: {stats['vector_store']['collection_name']}")
    print(f"   Documents: {stats['vector_store']['total_documents']:,}")
    
    # Performance stats
    if stats['performance']:
        print(f"\n⚡ Performance Metrics:")
        for operation, perf in stats['performance'].items():
            print(f"   {operation}:")
            print(f"      Avg: {perf['avg_duration']:.3f}s")
            print(f"      Count: {perf['count']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Sales RAG System - Enterprise-level RAG pipeline'
    )
    
    parser.add_argument(
        'mode',
        choices=['rag', 'predict', 'stats', 'setup'],
        help='Operation mode'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'rag':
            run_rag_demo()
        elif args.mode == 'predict':
            run_prediction_demo()
        elif args.mode == 'stats':
            show_stats()
        elif args.mode == 'setup':
            print("Run ./setup.sh to install dependencies")
    
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        # Cleanup
        telemetry.shutdown()
        
        # Show performance bottlenecks
        bottlenecks = metrics.identify_bottlenecks(threshold=1.0)
        if bottlenecks:
            print("\n⚠️  Performance Bottlenecks Detected:")
            for b in bottlenecks:
                print(f"   {b['operation']}: {b['avg_duration']:.2f}s")
                print(f"      → {b['recommendation']}")


if __name__ == "__main__":
    main()
