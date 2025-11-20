"""
Trace Visualization
-------------------
Visualize OpenTelemetry traces and performance metrics.
"""

import matplotlib.pyplot as plt
from src.telemetry import metrics
import os

def visualize_performance():
    """Create performance visualizations."""
    summary = metrics.get_summary()
    
    if not summary:
        print("No metrics recorded. Run some queries first!")
        return
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Prepare data
    operations = list(summary.keys())
    durations = [summary[op]['avg_duration'] for op in operations]
    counts = [summary[op]['count'] for op in operations]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Duration
    colors = ['#ff6b6b' if d > 1.0 else '#51cf66' for d in durations]
    bars = ax1.barh(operations, durations, color=colors)
    ax1.set_xlabel('Duration (seconds)')
    ax1.set_title('Operation Performance', fontweight='bold')
    ax1.axvline(x=1.0, color='red', linestyle='--', label='1s threshold')
    
    for i, (bar, val) in enumerate(zip(bars, durations)):
        ax1.text(val + 0.05, i, f'{val:.2f}s', va='center')
    
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Counts
    ax2.bar(operations, counts, color='#4c6ef5')
    ax2.set_ylabel('Count')
    ax2.set_title('Operation Frequency', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    for i, (op, count) in enumerate(zip(operations, counts)):
        ax2.text(i, count + 0.5, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig('outputs/performance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: outputs/performance.png")
    plt.close()


def print_report():
    """Print bottleneck report."""
    bottlenecks = metrics.identify_bottlenecks(threshold=1.0)
    
    print("\n" + "="*60)
    print("🔍 PERFORMANCE ANALYSIS")
    print("="*60)
    
    if not bottlenecks:
        print("\n✓ No bottlenecks detected!")
    else:
        print(f"\n⚠️  Found {len(bottlenecks)} bottleneck(s):\n")
        for i, b in enumerate(bottlenecks, 1):
            print(f"{i}. {b['operation']}: {b['avg_duration']:.2f}s")
            print(f"   💡 {b['recommendation']}\n")
    
    summary = metrics.get_summary()
    if summary:
        total_ops = sum(s['count'] for s in summary.values())
        print(f"\n📊 Total Operations: {total_ops}")
        print("="*60)


def main():
    """Generate visualizations."""
    print("🎨 Generating visualizations...\n")
    
    visualize_performance()
    print_report()
    
    print("\n✅ Done! Check outputs/ directory")


if __name__ == "__main__":
    main()