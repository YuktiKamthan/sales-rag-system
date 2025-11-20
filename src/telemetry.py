"""
OpenTelemetry Tracing Module
-----------------------------
This module implements distributed tracing to monitor system performance
and identify bottlenecks (leakage points) in the RAG pipeline.

Think of this as a GPS tracker for your data - it shows you exactly 
where time is being spent and where problems occur.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
import time
from functools import wraps
from typing import Callable, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelemetryManager:
    """
    Manages OpenTelemetry tracing for the entire system.
    
    This tracks:
    - How long each operation takes
    - Where errors occur
    - Which parts of the pipeline are slow
    """
    
    def __init__(self, service_name: str = "sales-rag-system", 
                 jaeger_host: str = "localhost",
                 jaeger_port: int = 6831,
                 enable_console: bool = True):
        """
        Initialize the telemetry system.
        
        Args:
            service_name: Name of your service (appears in traces)
            jaeger_host: Jaeger server host for visualization
            jaeger_port: Jaeger server port
            enable_console: Also print traces to console for debugging
        """
        self.service_name = service_name
        self.tracer_provider = None
        self.tracer = None
        
        self._setup_tracing(jaeger_host, jaeger_port, enable_console)
    
    def _setup_tracing(self, jaeger_host: str, jaeger_port: int, 
                      enable_console: bool):
        """
        Set up the tracing infrastructure.
        
        This creates the "plumbing" that tracks all operations.
        """
        logger.info(f"Setting up OpenTelemetry for {self.service_name}")
        
        # Create a resource that identifies this service
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": "1.0.0"
        })
        
        # Create the tracer provider
        self.tracer_provider = TracerProvider(resource=resource)
        
        # Add console exporter for debugging (prints to terminal)
        if enable_console:
            console_exporter = ConsoleSpanExporter()
            console_processor = BatchSpanProcessor(console_exporter)
            self.tracer_provider.add_span_processor(console_processor)
        
        # Add Jaeger exporter for visualization (optional, needs Jaeger running)
        try:
            jaeger_exporter = JaegerExporter(
                agent_host_name=jaeger_host,
                agent_port=jaeger_port,
            )
            jaeger_processor = BatchSpanProcessor(jaeger_exporter)
            self.tracer_provider.add_span_processor(jaeger_processor)
            logger.info(f"Jaeger exporter configured at {jaeger_host}:{jaeger_port}")
        except Exception as e:
            logger.warning(f"Jaeger not available: {e}. Traces will only show in console.")
        
        # Set as global tracer
        trace.set_tracer_provider(self.tracer_provider)
        self.tracer = trace.get_tracer(__name__)
        
        logger.info("Telemetry system ready!")
    
    def trace_operation(self, operation_name: str):
        """
        Decorator to automatically trace any function.
        
        Usage:
            @telemetry.trace_operation("load_data")
            def load_data():
                # your code here
        
        This will automatically:
        - Track how long the function takes
        - Log any errors
        - Add the operation to the trace
        
        Args:
            operation_name: Descriptive name for this operation
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                # Start a new span (tracking section)
                with self.tracer.start_as_current_span(operation_name) as span:
                    start_time = time.time()
                    
                    try:
                        # Add context to the span
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("operation.type", operation_name)
                        
                        # Execute the actual function
                        result = func(*args, **kwargs)
                        
                        # Track success
                        duration = time.time() - start_time
                        span.set_attribute("duration.seconds", duration)
                        span.set_status(Status(StatusCode.OK))
                        
                        logger.info(f"✓ {operation_name} completed in {duration:.2f}s")
                        
                        return result
                        
                    except Exception as e:
                        # Track errors
                        duration = time.time() - start_time
                        span.set_attribute("duration.seconds", duration)
                        span.set_attribute("error.message", str(e))
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        
                        logger.error(f"✗ {operation_name} failed after {duration:.2f}s: {e}")
                        raise
            
            return wrapper
        return decorator
    
    def create_manual_span(self, span_name: str):
        """
        Create a manual span for custom tracking.
        
        Usage:
            with telemetry.create_manual_span("custom_operation"):
                # your code here
        
        Args:
            span_name: Name for this tracking section
        """
        return self.tracer.start_as_current_span(span_name)
    
    def add_event(self, event_name: str, attributes: dict = None):
        """
        Add an event to the current span.
        
        Use this to mark important moments:
        - "data_loaded"
        - "embedding_created"
        - "prediction_made"
        
        Args:
            event_name: Name of the event
            attributes: Additional data about the event
        """
        span = trace.get_current_span()
        if span:
            span.add_event(event_name, attributes or {})
    
    def shutdown(self):
        """
        Gracefully shutdown telemetry system.
        Call this when your application exits.
        """
        if self.tracer_provider:
            self.tracer_provider.shutdown()
            logger.info("Telemetry system shutdown complete")


# Create a global instance for easy import
telemetry = TelemetryManager()


# Performance metrics tracker
class PerformanceMetrics:
    """
    Track and analyze performance metrics across the pipeline.
    
    This helps identify "leakage" - places where performance degrades.
    """
    
    def __init__(self):
        self.metrics = {
            'data_loading': [],
            'embedding_creation': [],
            'vector_search': [],
            'llm_generation': [],
            'prediction': []
        }
    
    def record(self, operation: str, duration: float, success: bool = True):
        """
        Record a performance metric.
        
        Args:
            operation: Name of the operation
            duration: How long it took (seconds)
            success: Whether it succeeded
        """
        if operation in self.metrics:
            self.metrics[operation].append({
                'duration': duration,
                'success': success,
                'timestamp': time.time()
            })
    
    def get_summary(self) -> dict:
        """
        Get summary statistics for all operations.
        
        Returns:
            Dictionary with averages, mins, maxs for each operation
        """
        summary = {}
        for operation, records in self.metrics.items():
            if records:
                durations = [r['duration'] for r in records]
                summary[operation] = {
                    'count': len(records),
                    'avg_duration': sum(durations) / len(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'success_rate': sum(1 for r in records if r['success']) / len(records)
                }
        return summary
    
    def identify_bottlenecks(self, threshold: float = 1.0) -> list:
        """
        Identify operations that are slow (potential leakage points).
        
        Args:
            threshold: Duration in seconds to consider "slow"
        
        Returns:
            List of slow operations
        """
        summary = self.get_summary()
        bottlenecks = []
        
        for operation, stats in summary.items():
            if stats['avg_duration'] > threshold:
                bottlenecks.append({
                    'operation': operation,
                    'avg_duration': stats['avg_duration'],
                    'recommendation': self._get_optimization_tip(operation)
                })
        
        return bottlenecks
    
    def _get_optimization_tip(self, operation: str) -> str:
        """Get optimization suggestions for slow operations."""
        tips = {
            'data_loading': "Consider caching data or using chunked reading",
            'embedding_creation': "Batch embeddings or use a faster model",
            'vector_search': "Reduce top_k or optimize index",
            'llm_generation': "Reduce max_tokens or use a faster model",
            'prediction': "Cache predictions or simplify model"
        }
        return tips.get(operation, "Profile this operation for optimization")


# Global metrics tracker
metrics = PerformanceMetrics()


if __name__ == "__main__":
    # Example usage
    print("Testing telemetry system...")
    
    @telemetry.trace_operation("example_operation")
    def example_function():
        time.sleep(0.5)  # Simulate work
        return "Success!"
    
    result = example_function()
    print(f"Result: {result}")
    
    # Record some metrics
    metrics.record('data_loading', 0.5)
    metrics.record('embedding_creation', 2.3)
    metrics.record('vector_search', 0.1)
    
    print("\nPerformance Summary:")
    print(metrics.get_summary())
    
    print("\nBottlenecks:")
    print(metrics.identify_bottlenecks(threshold=1.0))
    
    telemetry.shutdown()
