"""
Web UI for Sales RAG System
----------------------------
Interactive web interface with visualizations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.rag_pipeline import SalesRAGPipeline
from src.predictor import SalesPredictor
from src.telemetry import metrics
import time

# Page config
st.set_page_config(
    page_title="Sales RAG System",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# Title
st.title("🚀 Enterprise Sales RAG System")
st.markdown("Ask questions about your sales data using natural language")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    if st.button("Initialize System", type="primary"):
        with st.spinner("Loading system..."):
            try:
                st.session_state.pipeline = SalesRAGPipeline(
                    data_path="data/sales_data.xlsx",
                    collection_name="sales_data"
                )
                st.session_state.pipeline.initialize()
                st.success("✓ System ready!")
            except Exception as e:
                st.error(f"Error initializing: {e}")
    
    st.markdown("---")
    st.header("📊 System Status")
    
    if st.session_state.pipeline:
        try:
            stats = st.session_state.pipeline.get_stats()
            st.metric("Total Records", f"{stats['data']['total_records']:,}")
            st.metric("Total Revenue", f"${stats['data']['total_revenue']:,.2f}")
            st.metric("Unique Products", f"{stats['data']['unique_products']:,}")
        except:
            st.info("Stats loading...")
    else:
        st.info("Click 'Initialize System' to start")

# Main tabs
tab1, tab2, tab3 = st.tabs(["💬 Query", "📈 Predictions", "🔍 Performance"])

# Tab 1: Query
with tab1:
    st.header("Ask Questions")
    
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - What were the total sales in 2024?
        - Which products had the highest revenue?
        - Show me sales by season
        """)
    
    question = st.text_input("Your question:", placeholder="What were sales in 2024?")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        query_button = st.button("🔍 Search", type="primary")
    with col2:
        use_llm = st.checkbox("Use AI", value=True)
    
    if query_button and question:
        if not st.session_state.pipeline:
            st.error("⚠️ Initialize system first")
        else:
            with st.spinner("Searching..."):
                try:
                    result = st.session_state.pipeline.query(
                        question=question,
                        top_k=5,
                        use_llm=use_llm
                    )
                    
                    st.markdown("### 📝 Answer")
                    st.info(result['answer'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Time", f"{result['execution_time']:.2f}s")
                    with col2:
                        st.metric("Records", result['retrieved_records'])
                    
                    st.session_state.query_history.append({
                        'question': question,
                        'answer': result['answer'],
                        'time': result['execution_time']
                    })
                except Exception as e:
                    st.error(f"Error: {e}")
    
    if st.session_state.query_history:
        st.markdown("---")
        st.subheader("📜 Recent Queries")
        for query in reversed(st.session_state.query_history[-3:]):
            with st.expander(f"{query['question']} ({query['time']:.2f}s)"):
                st.write(query['answer'])

# Tab 2: Predictions
with tab2:
    st.header("📈 Sales Forecast")
    
    if st.button("Generate 30-Day Forecast"):
        if not st.session_state.pipeline:
            st.error("⚠️ Initialize system first")
        else:
            try:
                # Check data availability
                if not hasattr(st.session_state.pipeline, 'df') or st.session_state.pipeline.df is None:
                    st.error("❌ Data not loaded. Click 'Initialize System' first.")
                    st.stop()
                
                st.success(f"✅ Data ready: {len(st.session_state.pipeline.df):,} records")
                
                with st.spinner("Step 1/3: Creating predictor..."):
                    predictor = SalesPredictor(st.session_state.pipeline.df)
                    st.success("✓ Predictor created")
                
                with st.spinner("Step 2/3: Training Prophet model..."):
                    predictor.train()
                    st.success("✓ Model trained")
                
                with st.spinner("Step 3/3: Generating 30-day forecast..."):
                    try:
                        forecast_result = predictor.forecast(periods=30, freq='D')
                        st.success(f"✓ Forecast generated: {len(forecast_result)} predictions")
                    except Exception as e:
                        st.error(f"❌ Forecast failed: {str(e)}")
                        st.exception(e)
                        st.stop()
                
                with st.spinner("Extracting future predictions..."):
                    try:
                        future_pred = predictor.get_future_predictions(30)
                        st.success(f"✓ Got {len(future_pred)} future predictions")
                    except Exception as e:
                        st.error(f"❌ Failed to extract predictions: {str(e)}")
                        st.exception(e)
                        st.stop()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=future_pred['Date'],
                        y=future_pred['Predicted_Sales'],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.update_layout(
                        title="30-Day Sales Forecast",
                        xaxis_title="Date",
                        yaxis_title="Sales ($)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    summary = predictor.get_summary()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("30-Day Total", f"${summary['next_30_days_total']:,.2f}")
                    with col2:
                        st.metric("Daily Average", f"${summary['daily_average']:,.2f}")
            except Exception as e:
                st.error(f"Error: {e}")

# Tab 3: Performance
with tab3:
    st.header("🔍 Performance")
    
    if st.button("Refresh Metrics"):
        summary = metrics.get_summary()
        
        if summary:
            df_perf = pd.DataFrame([
                {'Operation': k, 'Duration (s)': v['avg_duration'], 'Count': v['count']}
                for k, v in summary.items()
            ])
            
            fig = px.bar(df_perf, x='Operation', y='Duration (s)', 
                        title='Operation Performance',
                        color='Duration (s)',
                        color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
            
            bottlenecks = metrics.identify_bottlenecks(threshold=1.0)
            if bottlenecks:
                st.warning("⚠️ Bottlenecks Detected")
                for b in bottlenecks:
                    st.markdown(f"**{b['operation']}**: {b['avg_duration']:.2f}s")
            else:
                st.success("✓ No bottlenecks!")
        else:
            st.info("Run queries first to see metrics")

st.markdown("---")
st.caption("Built with Streamlit | Enterprise Sales RAG System")