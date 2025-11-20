# 🎯 Cursor IDE Setup Guide

## Quick Start in Cursor

### Step 1: Open Project in Cursor
```bash
# In your terminal
cd /path/to/sales-rag-system
cursor .
```

### Step 2: Set Up Python Environment

**In Cursor's terminal (Ctrl+`)**:
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Mac/Linux
# or
.\venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure API Key

Create `.env` file:
```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:
```
GEMINI_API_KEY=your_actual_key_here
```

Get a free key at: https://makersuite.google.com/app/apikey

### Step 4: Run the Application

**Option 1: Interactive RAG**
```bash
python app.py rag
```

**Option 2: Sales Prediction**
```bash
python app.py predict
```

**Option 3: View Statistics**
```bash
python app.py stats
```

## 🎨 Cursor-Specific Tips

### 1. Use Cursor's AI Features

**Ask Cursor to explain code:**
- Select any function
- Press `Cmd+K` (Mac) or `Ctrl+K` (Windows)
- Type: "Explain this function"

**Ask Cursor to modify code:**
- Select a code block
- Press `Cmd+K`
- Type: "Add error handling to this function"

### 2. Recommended Extensions

Install these in Cursor:
- **Python** (Microsoft)
- **Pylance** (Microsoft)
- **autoDocstring** (for documentation)
- **GitLens** (for Git integration)

### 3. Debugging in Cursor

Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: RAG Demo",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/app.py",
            "args": ["rag"],
            "console": "integratedTerminal"
        },
        {
            "name": "Python: Prediction",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/app.py",
            "args": ["predict"],
            "console": "integratedTerminal"
        }
    ]
}
```

Press `F5` to debug!

### 4. Cursor Composer for Quick Edits

Use `Cmd+I` (Mac) or `Ctrl+I` (Windows) to open Composer:
- "Add logging to this file"
- "Create a test for this function"
- "Improve the error messages"

### 5. Multi-File Editing

In Composer, you can say:
- "Update all files to use async/await"
- "Add type hints to all functions"
- "Improve documentation across the project"

## 📝 Testing Individual Components

### Test Data Loader
```bash
python -c "from src.data_loader import SalesDataLoader; loader = SalesDataLoader('data/sales_data.xlsx'); df, stats = loader.prepare_for_rag(); print(stats)"
```

### Test Embeddings
```bash
python src/embeddings.py
```

### Test Vector Store
```bash
python src/vector_store.py
```

### Test RAG Pipeline
```bash
python src/rag_pipeline.py
```

### Test Predictor
```bash
python src/predictor.py
```

## 🔧 Common Issues

### Issue 1: Import Errors
```bash
# Make sure you're in virtual environment
source venv/bin/activate

# Reinstall packages
pip install -r requirements.txt
```

### Issue 2: API Key Not Found
```bash
# Check .env file exists
ls -la .env

# Check content
cat .env

# Make sure it has: GEMINI_API_KEY=your_key
```

### Issue 3: ChromaDB Errors
```bash
# Clear the database
rm -rf chroma_db/

# Rebuild
python app.py stats
```

## 🚀 Next Steps

1. **Modify the code** - Use Cursor AI to add features
2. **Add your data** - Replace sales_data.xlsx with your own
3. **Customize queries** - Edit prompts in rag_pipeline.py
4. **Add more metrics** - Extend predictor.py
5. **Deploy** - Use FastAPI for web service

## 💡 Cursor Pro Tips

### Quick Commands
- `Cmd+P`: Quick file search
- `Cmd+Shift+P`: Command palette
- `Cmd+B`: Toggle sidebar
- `Cmd+J`: Toggle terminal
- `Cmd+K`: Cursor AI chat

### Use Cursor to Learn
Ask Cursor:
- "How does RAG work in this codebase?"
- "Explain the embeddings.py module"
- "What design patterns are used here?"
- "How can I improve performance?"

### Generate Documentation
Select code and ask:
- "Add docstrings"
- "Generate README section for this"
- "Create usage examples"

## 📚 Learning Resources

- **RAG**: Understanding the architecture
- **Embeddings**: How text becomes vectors
- **Vector Databases**: Similarity search
- **OpenTelemetry**: Performance monitoring
- **Time Series**: Prophet forecasting

Use Cursor to explore each concept!

## 🎓 For Your Portfolio

This project demonstrates:
✅ Enterprise-level code organization
✅ Modern AI/ML techniques (RAG, embeddings)
✅ Production monitoring (OpenTelemetry)
✅ Time series forecasting
✅ Clean, documented code
✅ Modular architecture

Perfect for:
- GitHub portfolio
- LinkedIn project showcase
- Job interviews
- Learning AI/ML engineering

---

**Happy Coding with Cursor! 🚀**
