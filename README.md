# 🚀 Enterprise Sales RAG System

An enterprise-level Retrieval-Augmented Generation (RAG) pipeline with OpenTelemetry monitoring and AI-powered sales forecasting.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This system combines modern AI technologies to create an intelligent sales analytics platform:

- **RAG Pipeline**: Ask natural language questions about sales data
- **Vector Search**: Semantic search using sentence embeddings
- **Sales Forecasting**: Time series prediction with Prophet
- **OpenTelemetry**: Distributed tracing for performance monitoring
- **Enterprise-Ready**: Production-grade code with proper error handling

## 🎯 Features

### 1. Intelligent Query System
- Ask questions in natural language
- Get AI-powered answers based on actual sales data
- Semantic search finds relevant transactions automatically

### 2. Sales Prediction
- 30-day sales forecasting
- Automatic seasonality detection
- Trend analysis and visualization
- Performance metrics (MAE, RMSE, MAPE)

### 3. Performance Monitoring
- OpenTelemetry integration
- Automatic bottleneck detection
- Detailed execution tracing
- Performance metrics tracking

### 4. Production-Ready
- Comprehensive error handling
- Logging and monitoring
- Configurable via YAML
- Modular, testable code

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Query                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  RAG Pipeline                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Text to      │→ │ Vector       │→ │ LLM          │      │
│  │ Embedding    │  │ Search       │  │ Generation   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              OpenTelemetry Tracing                           │
│  • Monitors every operation                                  │
│  • Identifies performance bottlenecks                        │
│  • Tracks errors and success rates                          │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Gemini API key (get one at [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey))

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sales-rag-system.git
cd sales-rag-system

# Run setup script
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Add your API key
echo "GEMINI_API_KEY=your_key_here" > .env
```

### Usage

**1. Interactive RAG Query System**
```bash
python app.py rag
```
Example questions:
- "What were the total sales in 2024?"
- "Show me the top products by revenue"
- "Which customers had the highest purchases?"

**2. Sales Prediction**
```bash
python app.py predict
```
Generates 30-day forecast with visualizations.

**3. System Statistics**
```bash
python app.py stats
```
Shows data statistics and performance metrics.

## 📊 Technical Stack

### Core Technologies
- **Vector Database**: ChromaDB for efficient similarity search
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM**: Google Gemini Pro for answer generation
- **Forecasting**: Facebook Prophet for time series prediction
- **Monitoring**: OpenTelemetry for distributed tracing

### Python Libraries
```
pandas, numpy          # Data processing
chromadb              # Vector database
sentence-transformers # Embeddings
prophet               # Forecasting
opentelemetry         # Tracing
google-generativeai   # LLM integration
```

## 📁 Project Structure

```
sales-rag-system/
├── src/
│   ├── data_loader.py      # Data preprocessing
│   ├── embeddings.py       # Embedding generation
│   ├── vector_store.py     # ChromaDB integration
│   ├── rag_pipeline.py     # Main RAG pipeline
│   ├── predictor.py        # Sales forecasting
│   └── telemetry.py        # OpenTelemetry tracing
├── config/
│   └── config.yaml         # Configuration
├── data/
│   └── sales_data.xlsx     # Your sales data
├── outputs/                # Generated plots
├── app.py                  # Main application
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🔧 Configuration

Edit `config/config.yaml` to customize:

```yaml
# Embedding model
embeddings:
  model_name: "all-MiniLM-L6-v2"
  
# RAG settings
rag:
  top_k_results: 5
  temperature: 0.3
  
# Prediction settings
prediction:
  forecast_periods: 12
  seasonality_mode: "multiplicative"
```

## 📈 Performance Metrics

The system automatically tracks:
- **Data Loading Time**: < 2 seconds for 150K records
- **Embedding Generation**: ~5 seconds per 1000 records
- **Query Response Time**: < 1 second with LLM
- **Prediction Training**: ~10 seconds on 2 years of data

## 🎓 Key Concepts Explained

### What is RAG?
Retrieval-Augmented Generation combines:
1. **Retrieval**: Find relevant information from your data
2. **Augmentation**: Add that information as context
3. **Generation**: LLM generates answer using the context

### What are Embeddings?
Embeddings convert text into numerical vectors where similar meanings have similar numbers. This enables semantic search.

### What is OpenTelemetry?
OpenTelemetry provides observability:
- **Traces**: Track request flow through system
- **Spans**: Individual operations within a trace
- **Metrics**: Numerical measurements of performance

## 🔍 Example Queries

```python
from src.rag_pipeline import SalesRAGPipeline

# Initialize
pipeline = SalesRAGPipeline("data/sales_data.xlsx")
pipeline.initialize()

# Query
result = pipeline.query("What were sales in Q1 2024?")
print(result['answer'])
```

## 📊 Sales Prediction Example

```python
from src.predictor import SalesPredictor

# Create predictor
predictor = SalesPredictor(df)
predictor.train()

# Forecast 30 days
predictor.forecast(periods=30)
summary = predictor.get_summary()

print(f"Next 30 days total: ${summary['next_30_days_total']:,.2f}")
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 👤 Author

**Yukti**
- GitHub: [@YuktiKamthan](https://github.com/yourusername)
- LinkedIn: [Yukti Kamthan](www.linkedin.com/in/yuktikamthan)

## 🙏 Acknowledgments

- Sentence Transformers for embeddings
- Facebook Prophet for forecasting
- ChromaDB for vector storage
- OpenTelemetry for observability

## 📫 Contact

For questions or feedback, please open an issue or contact me directly.

---

**⭐ If this project helped you, please star the repository!**
