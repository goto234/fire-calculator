"""
🔥 FIRE Calculator – Monte-Carlo Simulation (India Edition)
Complete implementation with enhanced visualizations and professional reporting
"""

import io
import base64
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # NEW: Added for enhanced formatting
import streamlit as st
from fire_engine import simulate_fire
from data_fetcher import get_preset_scenarios
import traceback


# -----------------------------------------------------------------------------
# App Configuration & Enhanced Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="🔥 FIRE Monte-Carlo Calculator",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Professional color palette (colorblind-friendly)
COLORS = {
    'primary': '#2E86AB',      # Professional blue
    'secondary': '#A23B72',    # Accent purple
    'success': '#1B5E20',      # Dark green
    'warning': '#F57C00',      # Orange
    'danger': '#C62828',       # Red
    'neutral': '#424242',      # Dark gray
    'light_blue': '#E3F2FD',   # Light background
    'percentiles': ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#607D8B']  # P10-P90
}

# Enhanced matplotlib configuration for presentation quality
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'axes.grid': True,
    'grid.color': '#E0E0E0',
    'grid.alpha': 0.7,
    'grid.linewidth': 0.5,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.autolayout': True,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white'
})

# -----------------------------------------------------------------------------
# Enhanced Helper Functions
# -----------------------------------------------------------------------------
def format_inr_axis(value, pos=None):
    """Format axis values in Indian notation (₹10L, ₹1Cr)."""
    if value >= 1e7:
        return f"₹{value/1e7:.1f}Cr"
    elif value >= 1e5:
        return f"₹{value/1e5:.0f}L"
    elif value >= 1000:
        return f"₹{value/1000:.0f}K"
    else:
        return f"₹{value:.0f}"


def format_inr(val):
    """Format currency in Indian notation for display."""
    if val >= 1e7:
        return f"₹{val/1e7:.2f} Cr"
    elif val >= 1e5:
        return f"₹{val/1e5:.2f} L"
    elif val >= 1000:
        return f"₹{val/1000:.0f} K"
    else:
        return f"₹{val:,.0f}"


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()


# -----------------------------------------------------------------------------
# Enhanced Chart Functions
# -----------------------------------------------------------------------------
def create_enhanced_corpus_evolution_chart(corpus_matrix, inputs, results):
    """Create professional corpus evolution chart with dynamic scaling and annotations."""
    # Calculate time parameters
    retirement_years = inputs['retirement_age'] - inputs['current_age']
    total_years = retirement_years + inputs['total_years']
    years = np.arange(corpus_matrix.shape[1]) / 12  # Convert months to years
    
    # Create figure with presentation aspect ratio
    fig, ax = plt.subplots(figsize=(12, 6.75))  # 16:9 aspect ratio
    
    # Sample simulation paths for visualization (avoid overcrowding)
    n_paths_to_show = min(100, corpus_matrix.shape[0])
    sample_indices = np.random.choice(corpus_matrix.shape[0], n_paths_to_show, replace=False)
    sampled_paths = corpus_matrix[sample_indices]
    
    # Plot individual simulation paths (semi-transparent)
    for i, path in enumerate(sampled_paths):
        alpha = 0.03 if i > 20 else 0.1  # Make first 20 paths slightly more visible
        ax.plot(years, path, color=COLORS['neutral'], alpha=alpha, linewidth=0.5)
    
    # Plot key percentile lines
    p10 = np.percentile(corpus_matrix, 10, axis=0)
    p50 = np.percentile(corpus_matrix, 50, axis=0)
    p90 = np.percentile(corpus_matrix, 90, axis=0)
    
    ax.plot(years, p50, color=COLORS['primary'], linewidth=3, label='Median (P50)', zorder=5)
    ax.plot(years, p10, color=COLORS['danger'], linewidth=2, linestyle='--', 
            label='Pessimistic (P10)', zorder=4)
    ax.plot(years, p90, color=COLORS['success'], linewidth=2, linestyle='--', 
            label='Optimistic (P90)', zorder=4)
    
    # Add retirement transition line
    if retirement_years > 0:
        ax.axvline(x=retirement_years, color=COLORS['warning'], linestyle=':', 
                   linewidth=2, alpha=0.8, zorder=3)
        # Add annotation for retirement
        y_pos = ax.get_ylim()[1] * 0.9
        ax.annotate('Retirement Starts', 
                    xy=(retirement_years, y_pos), 
                    xytext=(retirement_years + total_years*0.1, y_pos),
                    arrowprops=dict(arrowstyle='->', color=COLORS['warning']),
                    fontsize=10, color=COLORS['warning'], weight='bold')
    
    # Highlight accumulation vs retirement phases
    if retirement_years > 0:
        ax.axvspan(0, retirement_years, alpha=0.05, color=COLORS['success'], 
                   label='Accumulation Phase')
        ax.axvspan(retirement_years, max(years), alpha=0.05, color=COLORS['danger'], 
                   label='Retirement Phase')
    
    # Dynamic Y-axis scaling based on data
    y_max = np.percentile(corpus_matrix, 95) * 1.1  # Use P95 for better scaling
    ax.set_ylim(bottom=0, top=y_max)
    
    # Formatting
    ax.set_xlabel('Years from Start', fontweight='bold')
    ax.set_ylabel('Portfolio Value', fontweight='bold')
    ax.set_title(f'Portfolio Evolution: {n_paths_to_show} Sample Paths from {corpus_matrix.shape[0]:,} Simulations', 
                 pad=20, fontweight='bold')
    
    # Indian currency formatting for Y-axis
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_inr_axis))
    
    # Professional legend positioning
    legend = ax.legend(loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    return fig


def create_enhanced_burn_curve_chart(corpus_matrix, inputs, results):
    """Create professional burn curve showing percentile ranges with risk zones."""
    # Calculate time parameters
    retirement_years = inputs['retirement_age'] - inputs['current_age']
    total_years = retirement_years + inputs['total_years']
    years = np.arange(corpus_matrix.shape[1]) / 12
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6.75))
    
    # Calculate percentiles
    percentiles = [10, 25, 50, 75, 90]
    p_data = {}
    for p in percentiles:
        p_data[p] = np.percentile(corpus_matrix, p, axis=0)
    
    # Plot percentile bands
    ax.fill_between(years, p_data[10], p_data[90], alpha=0.2, color=COLORS['primary'], 
                    label='80% Confidence Band (P10-P90)')
    ax.fill_between(years, p_data[25], p_data[75], alpha=0.3, color=COLORS['primary'], 
                    label='50% Confidence Band (P25-P75)')
    
    # Plot percentile lines
    ax.plot(years, p_data[50], color=COLORS['primary'], linewidth=3, 
            label='Median (P50)', zorder=5)
    ax.plot(years, p_data[10], color=COLORS['danger'], linewidth=2, 
            linestyle='--', label='Pessimistic (P10)', zorder=4)
    ax.plot(years, p_data[90], color=COLORS['success'], linewidth=2, 
            linestyle='--', label='Optimistic (P90)', zorder=4)
    
    # Add retirement transition
    if retirement_years > 0:
        ax.axvline(x=retirement_years, color=COLORS['warning'], linestyle=':', 
                   linewidth=2, alpha=0.8, zorder=3, label='Retirement Starts')
    
    # Highlight ruin zone
    ruin_threshold = max(years) * 0.1  # Show if ruin occurs in last 10% of timeline
    if np.any(p_data[10][-int(len(years)*0.1):] <= 0):
        ax.axhspan(0, ax.get_ylim()[1]*0.05, alpha=0.3, color=COLORS['danger'], 
                   label='Ruin Risk Zone')
    
    # Add goal line if provided
    if 'retirement_goal' in inputs and inputs['retirement_goal'] > 0:
        goal_line_y = inputs['retirement_goal']
        if goal_line_y <= ax.get_ylim()[1]:
            ax.axhline(y=goal_line_y, color=COLORS['success'], linestyle=':', 
                       alpha=0.7, linewidth=2, label=f"Goal: {format_inr(goal_line_y)}")
    
    # Dynamic Y-axis scaling
    y_max = np.percentile(p_data[90], 95) * 1.1
    ax.set_ylim(bottom=0, top=y_max)
    
    # Formatting
    ax.set_xlabel('Years from Start', fontweight='bold')
    ax.set_ylabel('Portfolio Value', fontweight='bold')
    ax.set_title('Burn Curve: Portfolio Value Distribution Over Time', 
                 pad=20, fontweight='bold')
    
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_inr_axis))
    
    # Professional legend
    legend = ax.legend(loc='upper right', framealpha=0.9, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_enhanced_histogram_chart(final_dist, inputs, results):
    """Create professional final corpus distribution histogram with outcome analysis."""
    # Clean and prepare data
    final_dist = np.asarray(final_dist).flatten()
    final_dist = final_dist[np.isfinite(final_dist)]
    
    # Convert to crores for better readability
    final_cr = final_dist / 1e7
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6.75))
    
    # Determine optimal binning
    n_bins = min(50, max(20, len(final_cr) // 100))
    
    # Create histogram
    n, bins, patches = ax.hist(final_cr, bins=n_bins, color=COLORS['primary'], 
                               alpha=0.7, edgecolor='white', linewidth=0.5)
    
    # Color ruin scenarios (negative values) in red
    for i, patch in enumerate(patches):
        if bins[i] <= 0:
            patch.set_facecolor(COLORS['danger'])
            patch.set_alpha(0.8)
    
    # Add statistical lines with annotations
    mean_val = np.mean(final_cr)
    median_val = np.median(final_cr)
    
    # Add vertical lines for statistics
    ax.axvline(mean_val, color=COLORS['warning'], linestyle='--', linewidth=2, 
               alpha=0.9, label=f'Mean: {format_inr(mean_val * 1e7)}')
    ax.axvline(median_val, color=COLORS['primary'], linestyle='--', linewidth=2, 
               alpha=0.9, label=f'Median: {format_inr(median_val * 1e7)}')
    
    # Add goal line if provided
    if 'retirement_goal' in inputs and inputs['retirement_goal'] > 0:
        goal_cr = inputs['retirement_goal'] / 1e7
        if ax.get_xlim()[0] <= goal_cr <= ax.get_xlim()[1]:
            ax.axvline(goal_cr, color=COLORS['success'], linestyle=':', linewidth=2, 
                       alpha=0.9, label=f'Goal: {format_inr(inputs["retirement_goal"])}')
    
    # Add outcome statistics annotation
    ruin_count = np.sum(final_dist <= 0)
    total_sims = len(final_dist)
    success_rate = results.get('success_rate', 0) * 100
    
    stats_text = f"""Simulation Summary:
• Success Rate: {success_rate:.1f}%
• Failures: {ruin_count:,} / {total_sims:,}
• P10: {format_inr(np.percentile(final_dist, 10))}
• P90: {format_inr(np.percentile(final_dist, 90))}"""
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
            fontsize=10, family='monospace')
    
    # Formatting
    ax.set_xlabel('Final Portfolio Value (₹ Crores)', fontweight='bold')
    ax.set_ylabel('Number of Simulations', fontweight='bold')
    ax.set_title('Final Portfolio Distribution: Outcome Probability Analysis', 
                 pad=20, fontweight='bold')
    
    # Professional legend
    legend = ax.legend(loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig


def create_complete_visualization_suite(results, inputs):
    """Create all three enhanced charts and return base64 encoded images."""
    corpus_matrix = results['corpus_matrix']
    
    # Generate all charts
    evolution_fig = create_enhanced_corpus_evolution_chart(corpus_matrix, inputs, results)
    burn_curve_fig = create_enhanced_burn_curve_chart(corpus_matrix, inputs, results)
    histogram_fig = create_enhanced_histogram_chart(results['final_corpus_distribution'], inputs, results)
    
    # Convert to base64
    charts_b64 = {
        'evolution': fig_to_base64(evolution_fig),
        'burn_curve': fig_to_base64(burn_curve_fig),
        'histogram': fig_to_base64(histogram_fig)
    }
    
    return charts_b64, evolution_fig, burn_curve_fig, histogram_fig


def display_enhanced_results(results, inputs):
    """Display enhanced results in Streamlit with professional formatting."""
    # Create all visualizations
    charts_b64, evolution_fig, burn_curve_fig, histogram_fig = create_complete_visualization_suite(results, inputs)
    
    # Display key metrics with enhanced formatting
    st.markdown("### 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    success_rate = results['success_rate'] * 100
    goal_hit_rate = results['goal_hit_rate'] * 100
    
    with col1:
        color = "🟢" if success_rate > 90 else "🟡" if success_rate > 70 else "🔴"
        st.metric(
            label=f"{color} Success Rate",
            value=f"{success_rate:.1f}%",
            help="Probability of never running out of money"
        )
    
    with col2:
        color = "🟢" if goal_hit_rate > 80 else "🟡" if goal_hit_rate > 50 else "🔴"
        st.metric(
            label=f"{color} Goal Achievement",
            value=f"{goal_hit_rate:.1f}%",
            help="Likelihood of reaching retirement target"
        )
    
    with col3:
        st.metric(
            label="💰 Median Final Corpus",
            value=format_inr(results['p50_final']),
            help="Expected portfolio value at end of timeline"
        )
    
    with col4:
        st.metric(
            label="💳 Safe Withdrawal Rate",
            value=format_inr(results.get('swr_annual', 0)),
            help="Annual sustainable withdrawal (Conservative P10 basis)"
        )
    
    # Display charts
    st.markdown("### 📈 Portfolio Evolution Analysis")
    st.pyplot(evolution_fig, use_container_width=True)
    
    st.markdown("### 📉 Risk Profile: Percentile Analysis") 
    st.pyplot(burn_curve_fig, use_container_width=True)
    
    st.markdown("### 📊 Outcome Distribution")
    st.pyplot(histogram_fig, use_container_width=True)
    
    return charts_b64


def generate_comprehensive_html_report(results, inputs, charts_b64=None, version="2.0"):
    """Generate presentation-ready HTML report with comprehensive analysis."""
    # Extract key data
    r = results
    params = r.get("simulation_params", {})
    
    # Calculate additional metrics
    final_dist = r['final_corpus_distribution']
    
    # Professional CSS styling
    css = """
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', 'Arial', sans-serif; 
            line-height: 1.6; 
            color: #333; 
            background: #fafafa;
            padding: 20px;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            border-radius: 10px;
            overflow: hidden;
        }
        .header { 
            background: linear-gradient(135deg, #2E86AB, #A23B72); 
            color: white; 
            padding: 30px; 
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; opacity: 0.9; }
        .content { padding: 30px; }
        .section { margin-bottom: 40px; }
        .section h2 { 
            color: #2E86AB; 
            border-bottom: 3px solid #E3F2FD; 
            padding-bottom: 10px; 
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        .section h3 { 
            color: #424242; 
            margin: 20px 0 10px 0; 
            font-size: 1.3em;
        }
        .metrics-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; 
            margin: 20px 0;
        }
        .metric-card { 
            background: #f8f9fa; 
            padding: 20px; 
            border-radius: 8px; 
            border-left: 4px solid #2E86AB;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .metric-card h4 { color: #2E86AB; margin-bottom: 5px; }
        .metric-card .value { font-size: 1.5em; font-weight: bold; color: #333; }
        .metric-card .description { font-size: 0.9em; color: #666; margin-top: 5px; }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        th, td { 
            padding: 12px 15px; 
            text-align: left; 
            border-bottom: 1px solid #E0E0E0;
        }
        th { 
            background: #E3F2FD; 
            font-weight: bold; 
            color: #2E86AB;
        }
        tr:nth-child(even) { background: #f8f9fa; }
        .chart-container { 
            margin: 20px 0; 
            text-align: center;
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .chart-container img { 
            max-width: 100%; 
            height: auto; 
            border-radius: 5px;
        }
        .alert { 
            padding: 15px; 
            margin: 15px 0; 
            border-radius: 5px; 
            border-left: 4px solid;
        }
        .alert-success { background: #d4edda; border-color: #28a745; color: #155724; }
        .alert-warning { background: #fff3cd; border-color: #ffc107; color: #856404; }
        .alert-danger { background: #f8d7da; border-color: #dc3545; color: #721c24; }
        .recommendations { 
            background: #e8f5e8; 
            padding: 20px; 
            border-radius: 8px; 
            margin: 20px 0;
        }
        .recommendations ul { margin-left: 20px; }
        .recommendations li { margin: 8px 0; }
        .footer { 
            background: #f8f9fa; 
            padding: 20px; 
            text-align: center; 
            font-size: 0.9em; 
            color: #666;
        }
        @media print {
            body { background: white; }
            .container { box-shadow: none; }
            .header { background: #2E86AB; }
        }
    </style>
    """
    
    # Generate content sections
    success_rate = r['success_rate'] * 100
    goal_hit_rate = r['goal_hit_rate'] * 100
    
    executive_summary = f"""
    <div class="section">
        <h2>📊 Executive Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <h4>Success Rate</h4>
                <div class="value" style="color: {'#28a745' if success_rate > 90 else '#ffc107' if success_rate > 70 else '#dc3545'}">{success_rate:.1f}%</div>
                <div class="description">Probability of never running out of money</div>
            </div>
            <div class="metric-card">
                <h4>Goal Achievement</h4>
                <div class="value" style="color: {'#28a745' if goal_hit_rate > 80 else '#ffc107' if goal_hit_rate > 50 else '#dc3545'}">{goal_hit_rate:.1f}%</div>
                <div class="description">Likelihood of reaching retirement goal</div>
            </div>
            <div class="metric-card">
                <h4>Median Final Corpus</h4>
                <div class="value">{format_inr(r['p50_final'])}</div>
                <div class="description">Expected portfolio value at end</div>
            </div>
            <div class="metric-card">
                <h4>Safe Withdrawal Rate</h4>
                <div class="value">{format_inr(r.get('swr_annual', 0))}</div>
                <div class="description">Annual sustainable withdrawal (P10 basis)</div>
            </div>
        </div>
    </div>
    """
    
    # User Inputs Table
    user_inputs = f"""
    <div class="section">
        <h2>📋 Simulation Parameters</h2>
        <table>
            <thead>
                <tr><th>Parameter</th><th>Value</th><th>Parameter</th><th>Value</th></tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Current Age</strong></td>
                    <td>{inputs['current_age']} years</td>
                    <td><strong>Retirement Age</strong></td>
                    <td>{inputs['retirement_age']} years</td>
                </tr>
                <tr>
                    <td><strong>Initial Corpus</strong></td>
                    <td>{format_inr(inputs['initial_corpus'])}</td>
                    <td><strong>Monthly SIP</strong></td>
                    <td>{format_inr(inputs['monthly_contribution'])}</td>
                </tr>
                <tr>
                    <td><strong>Retirement Goal</strong></td>
                    <td>{format_inr(inputs['retirement_goal'])}</td>
                    <td><strong>Monthly Expenses (Year 1)</strong></td>
                    <td>{format_inr(inputs['monthly_expense'])}</td>
                </tr>
                <tr>
                    <td><strong>Equity Allocation</strong></td>
                    <td>{inputs['equity_weight']*100:.0f}%</td>
                    <td><strong>Post-Retirement Adjustment</strong></td>
                    <td>{inputs['dampen_post_ret']*100:.0f}% of expenses</td>
                </tr>
                <tr>
                    <td><strong>Simulation Paths</strong></td>
                    <td>{params.get('n_simulations', 'N/A'):,}</td>
                    <td><strong>Return Window</strong></td>
                    <td>{params.get('return_window', 'N/A')} years</td>
                </tr>
            </tbody>
        </table>
    </div>
    """
    
    # Charts section
    charts_section = ""
    if charts_b64:
        charts_section = f"""
        <div class="section">
            <h2>📊 Visual Analysis</h2>
            
            <div class="chart-container">
                <h3>Portfolio Evolution Over Time</h3>
                <img src="data:image/png;base64,{charts_b64.get('evolution', '')}" alt="Corpus Evolution Chart">
            </div>
            
            <div class="chart-container">
                <h3>Risk Analysis: Percentile Ranges</h3>
                <img src="data:image/png;base64,{charts_b64.get('burn_curve', '')}" alt="Burn Curve Chart">
            </div>
            
            <div class="chart-container">
                <h3>Outcome Distribution</h3>
                <img src="data:image/png;base64,{charts_b64.get('histogram', '')}" alt="Final Distribution">
            </div>
        </div>
        """
    
    # Detailed Results
    detailed_results = f"""
    <div class="section">
        <h2>📈 Detailed Analysis</h2>
        <table>
            <thead>
                <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
            </thead>
            <tbody>
                <tr>
                    <td>P10 Final Corpus (Pessimistic)</td>
                    <td>{format_inr(r['p10_final'])}</td>
                    <td>Bottom 10% outcome</td>
                </tr>
                <tr>
                    <td>P50 Final Corpus (Median)</td>
                    <td>{format_inr(r['p50_final'])}</td>
                    <td>Most likely outcome</td>
                </tr>
                <tr>
                    <td>P90 Final Corpus (Optimistic)</td>
                    <td>{format_inr(r['p90_final'])}</td>
                    <td>Top 10% outcome</td>
                </tr>
                <tr>
                    <td>Median Years to Ruin</td>
                    <td>{r['median_years_to_ruin']:.1f} years</td>
                    <td>When portfolio depletes (if ever)</td>
                </tr>
                <tr>
                    <td>Corpus at Retirement (P50)</td>
                    <td>{format_inr(r['corpus_at_retirement_p50'])}</td>
                    <td>Expected retirement corpus</td>
                </tr>
            </tbody>
        </table>
    </div>
    """
    
    # Risk Assessment and Recommendations
    risk_assessment = f"""
    <div class="section">
        <h2>🎯 Risk Assessment & Recommendations</h2>
        
        {'<div class="alert alert-success"><strong>✅ Excellent Plan:</strong> Your strategy shows strong resilience with high probability of success.</div>' if success_rate > 90 else
         '<div class="alert alert-warning"><strong>⚠️ Moderate Risk:</strong> Consider adjustments to improve success probability.</div>' if success_rate > 70 else
         '<div class="alert alert-danger"><strong>❌ High Risk:</strong> Significant changes needed to avoid financial shortfall.</div>'}
        
        <div class="recommendations">
            <h3>💡 Specific Recommendations</h3>
            <ul>
                <li><strong>Regular Reviews:</strong> Reassess annually for inflation, salary changes, and life events</li>
                <li><strong>Emergency Fund:</strong> Maintain 6-12 months expenses separate from investment corpus</li>
                <li><strong>Professional Advice:</strong> Consult a SEBI-registered financial advisor for personalized strategy</li>
            </ul>
        </div>
    </div>
    """
    
    # Footer
    footer = f"""
    <div class="footer">
        <p><strong>🔥 FIRE Calculator v{version}</strong> | Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p IST')}</p>
        <p>Monte Carlo simulation with {params.get('n_simulations', 'N/A'):,} paths | Indian market data analysis</p>
        <p><em>This comprehensive analysis is for educational purposes only. Please consult with a SEBI-registered financial advisor before making investment decisions.</em></p>
    </div>
    """
    
    # Combine all sections
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🔥 FIRE Planning Report - Monte Carlo Analysis</title>
        {css}
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔥 FIRE Planning Report</h1>
                <p>Comprehensive Monte Carlo Analysis for Financial Independence</p>
            </div>
            <div class="content">
                {executive_summary}
                {user_inputs}
                {charts_section}
                {detailed_results}
                {risk_assessment}
            </div>
            {footer}
        </div>
    </body>
    </html>
    """
    
    return html_content


# -----------------------------------------------------------------------------
# Main Streamlit App (Enhanced)
# -----------------------------------------------------------------------------
def main():
    # ────────────────────────────────────────────────────────────────
    # 1.  PRESET MAP + CALLBACK  (this MUST come first)
    # ────────────────────────────────────────────────────────────────
    presets = get_preset_scenarios()          # e.g. {'2008 Crash': {...}, …}
    presets["Custom"] = {}                    # always allow free-form

    #  Callback – overwrite the slider-keys, no manual rerun needed
    def apply_preset():
        sel = st.session_state["selected_preset"]
        st.session_state["equity_stress_slider"] = int(presets[sel].get("equity_stress", 0) * 100)
        st.session_state["bond_stress_slider"]   = int(presets[sel].get("bond_stress",  0) * 100)

    # ── Single dropdown – defaults to "Custom" ──
    options = ["Custom"] + [k for k in presets.keys() if k != "Custom"]

    selected_preset = st.sidebar.selectbox(
        "🎯 Stress Scenario Preset",
        options=options,          # 'Custom' is first
        key="selected_preset",
        on_change=apply_preset,
        index=0                   # pre-select Custom
    ) 

    # ── Default state for first launch (runs once) ────────────────────────────
    if "initialized" not in st.session_state:
        st.session_state.update(
            {
                # personal
                "current_age": 30,
                "retirement_age": 50,
                "total_years": 35,
                # financial
                "initial_corpus": 500000,
                "monthly_contribution": 50000,
                "include_accumulation": True,
                "retirement_goal": 20000000,
                # expenses
                "monthly_expense": 50000,
                "monthly_expense_growth": 2.0,   # slider shows %; we'll divide by 100 later
                "dampen_post_ret": 80,           # slider shows %
                # portfolio
                "equity_pct": 80,
                "use_glide_path": False,
                "equity_glide_pct": 80,
                # advanced
                "n_simulations": 10000,
                "return_window": 20,
                "equity_ticker": "^NSEI",
                "bond_series": "10-Year G-Sec",
                # stress (Custom starts at 0/0)
                "equity_stress_slider": 0,
                "bond_stress_slider": 0,
                # guard flag
                "initialized": True,
            }
        )

    # ────────────────────────────────────────────────────────────────
    # 2.  REGULAR INPUT WIDGETS
    # ────────────────────────────────────────────────────────────────
    st.title("🔥 FIRE Calculator – Monte Carlo Edition")
    st.markdown("*Comprehensive Financial Independence & Early Retirement Planning for India*")

    st.sidebar.header("📊 Input Parameters")

    # Personal Details
    st.sidebar.subheader("👤 Personal Details")
    current_age = st.sidebar.number_input("Current Age", 20, 65, key="current_age")
    retirement_age = st.sidebar.number_input("Planned Retirement Age",
                                             current_age + 1, 80, key="retirement_age")
    total_years = st.sidebar.number_input("Years in Retirement", 5, 50, key="total_years")

    # Financial Parameters
    st.sidebar.subheader("💰 Financial Parameters")
    initial_corpus = st.sidebar.number_input("Initial Corpus (₹)", 0, 100_000_000,
                                             step=50_000, key="initial_corpus")
    monthly_contribution = st.sidebar.number_input("Monthly SIP (₹)", 0, 1_000_000,
                                                   step=5_000, key="monthly_contribution")
    include_accumulation = st.sidebar.checkbox("Include SIP during accumulation phase",
                                               key="include_accumulation")
    retirement_goal = st.sidebar.number_input("Retirement Goal (₹)", 1_000_000,
                                              1_000_000_000, step=1_000_000,
                                              key="retirement_goal")

    # Expense Parameters
    st.sidebar.subheader("💸 Expense Parameters")
    monthly_expense = st.sidebar.number_input("Monthly Expense (Year 1, ₹)", 10_000, 1_000_000,
                                              step=5_000, key="monthly_expense")
    monthly_expense_growth = st.sidebar.slider("Real Expense Growth % / yr",
                                               0.0, 5.0, step=0.1,
                                               key="monthly_expense_growth") / 100
    dampen_post_ret = st.sidebar.slider("Post-Retirement Expense Reduction %",
                                        50, 100, step=5, key="dampen_post_ret") / 100

    # Portfolio Parameters
    st.sidebar.subheader("📈 Portfolio Parameters")
    equity_pct = st.sidebar.slider("Equity Allocation %", 0, 100, step=5, key="equity_pct")
    equity_weight = equity_pct / 100
    use_glide_path = st.sidebar.checkbox("Use Equity Glide Path", key="use_glide_path")
    if use_glide_path:
        glide_pct = st.sidebar.slider("Final Equity Allocation %", 0, 100, step=5,
                                      key="equity_glide_pct")
        equity_glide_end = glide_pct / 100
    else:
        equity_glide_end = equity_weight

    # Advanced Parameters  (includes the ONLY stress sliders)
    with st.sidebar.expander("🔧 Advanced Parameters", expanded=False):
        n_simulations = st.number_input("Number of Simulations", 1000, 50000,
                                        step=1000, key="n_simulations")
        return_window = st.number_input("Return Window (yrs)", 5, 30,
                                        step=1, key="return_window")
        equity_ticker = st.selectbox("Equity Index", ["^NSEI", "NIFTY50.NS", "^BSESN"],
                                     index=0, key="equity_ticker")
        bond_series = st.selectbox("Bond Series", ["10-Year G-Sec", "Corporate Bond", "Gilt"],
                                   index=0, key="bond_series")

        st.subheader("⚠️ Stress Testing")
        equity_stress = (
            st.slider("Equity Return Stress %", -50, 50, step=5,
                      key="equity_stress_slider") / 100
        )
        bond_stress = (
            st.slider("Bond Return Stress %", -50, 50, step=5,
                      key="bond_stress_slider") / 100
        )

    # ────────────────────────────────────────────────────────────────
    # 3.  VALIDATION AND INPUT PREPARATION
    # ────────────────────────────────────────────────────────────────

    # Basic sanity checks
    if retirement_age <= current_age:
        st.error("Retirement age must be greater than current age.")
        st.stop()

    if monthly_expense * 12 > retirement_goal:
        st.warning("⚠️ Your annual expenses exceed your target retirement corpus.")

    # Consolidated user input dictionary
    inputs = {
        'current_age': current_age,
        'retirement_age': retirement_age,
        'total_years': total_years,
        'initial_corpus': initial_corpus,
        'monthly_contribution': monthly_contribution,
        'include_accumulation': include_accumulation,
        'monthly_expense': monthly_expense,
        'monthly_expense_growth': monthly_expense_growth,
        'equity_weight': equity_weight,
        'equity_glide_end': equity_glide_end,
        'dampen_post_ret': dampen_post_ret,
        'retirement_goal': retirement_goal,
    }

    # Simulation config
    simulation_params = {
        'n_simulations': n_simulations,
        'return_window': return_window,
        'equity_ticker': equity_ticker,
        'bond_series': bond_series,
        'equity_stress': equity_stress,
        'bond_stress': bond_stress,
    }

    # ────────────────────────────────────────────────────────────────
    # 4.  RUN SIMULATION + ENHANCED VISUALIZE RESULTS
    # ────────────────────────────────────────────────────────────────

    if st.button("🚀 Run Monte Carlo Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            try:
                results = simulate_fire(**inputs, **simulation_params)
                st.success("✅ Simulation completed!")

                # Enhanced results display
                charts_b64 = display_enhanced_results(results, inputs)
                
                # Save state for report generation
                st.session_state["last_results"] = results
                st.session_state["last_inputs"] = inputs
                st.session_state["last_charts"] = charts_b64

                # Enhanced detailed results table
                st.markdown("### 📋 Comprehensive Results Analysis")
                try:
                    results_df = pd.DataFrame({
                        'Metric': [
                            'Success Rate (%)',
                            'Goal Hit Rate (%)', 
                            'Ruined Simulations',
                            'P10 Final Corpus (Pessimistic)',
                            'P25 Final Corpus',
                            'P50 Final Corpus (Median)',
                            'P75 Final Corpus',
                            'P90 Final Corpus (Optimistic)',
                            'Mean Final Corpus',
                            'Standard Deviation',
                            'Median Years to Ruin',
                            'Safe Withdrawal Rate (Annual)',
                            'Corpus at Retirement (P50)',
                            'Corpus at Retirement (P10)',
                            'Corpus at Retirement (P90)',
                        ],
                        'Value': [
                            f"{results['success_rate']*100:.2f}%",
                            f"{results['goal_hit_rate']*100:.2f}%",
                            f"{results['ruin_count']:,} / {simulation_params['n_simulations']:,}",
                            format_inr(results['p10_final']),
                            format_inr(np.percentile(results['final_corpus_distribution'], 25)),
                            format_inr(results['p50_final']),
                            format_inr(np.percentile(results['final_corpus_distribution'], 75)),
                            format_inr(results['p90_final']),
                            format_inr(results['mean_final']),
                            format_inr(np.std(results['final_corpus_distribution'])),
                            f"{results['median_years_to_ruin']:.1f} years",
                            format_inr(results.get('swr_annual', 0)),
                            format_inr(results['corpus_at_retirement_p50']),
                            format_inr(results.get('corpus_at_retirement_p10', 0)),
                            format_inr(results.get('corpus_at_retirement_p90', 0)),
                        ],
                        'Interpretation': [
                            'Never ran out of money',
                            'Reached retirement target',
                            'Failed simulations',
                            'Bottom 10% outcome',
                            'Bottom quartile',
                            'Most likely outcome',
                            'Top quartile',
                            'Top 10% outcome', 
                            'Average across all simulations',
                            'Volatility measure',
                            'When portfolio depletes',
                            'Conservative annual withdrawal',
                            'Expected retirement corpus',
                            'Conservative retirement corpus',
                            'Optimistic retirement corpus'
                        ]
                    })
                    st.dataframe(results_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating results table: {e}")

                # Enhanced recommendations
                st.markdown("### 🎯 Professional Analysis & Recommendations")
                
                success_rate = results['success_rate'] * 100
                goal_hit_rate = results['goal_hit_rate'] * 100
                
                # Success rate assessment
                if success_rate > 95:
                    st.success("🎉 **Excellent Plan!** Very high probability of long-term financial security.")
                elif success_rate > 85:
                    st.info("✅ **Strong Plan!** Good probability with minor fine-tuning recommended.")
                elif success_rate > 70:
                    st.warning("⚠️ **Moderate Risk Plan.** Consider strategic adjustments to improve outcomes.")
                else:
                    st.error("❌ **High Risk Plan.** Significant changes required to avoid financial shortfall.")

                # Goal achievement assessment
                if goal_hit_rate > 80:
                    st.success("🎯 **Goal Highly Achievable** - Strong probability of reaching retirement target.")
                elif goal_hit_rate > 60:
                    st.info("🎯 **Goal Likely Achievable** - Reasonable probability with current strategy.")
                elif goal_hit_rate > 40:
                    st.warning("🎯 **Goal Challenging** - Consider adjustments to improve likelihood.")
                else:
                    st.error("🎯 **Goal At Risk** - Target may be unrealistic with current parameters.")

                # Specific recommendations
                st.markdown("### 💡 Specific Action Items")
                
                with st.expander("📈 SIP & Investment Recommendations", expanded=True):
                    recs = []
                    
                    if success_rate < 85:
                        bump = 30 if success_rate < 70 else 20
                        new_sip = monthly_contribution * (1 + bump/100)
                        recs.append(f"**Increase Monthly SIP:** Raise to {format_inr(new_sip)} ({bump}% increase)")
                    
                    if equity_weight < 0.7 and current_age < 40:
                        recs.append(f"**Optimize Asset Allocation:** Consider 70-80% equity allocation (currently {equity_weight*100:.0f}%)")
                    
                    if goal_hit_rate < 60:
                        alternative_goal = retirement_goal * 0.8
                        recs.append(f"**Adjust Expectations:** Consider revised goal of {format_inr(alternative_goal)} or extend working years")
                    
                    if monthly_expense * 12 * 25 > retirement_goal:
                        required_corpus = monthly_expense * 12 * 25
                        recs.append(f"**25x Rule Alert:** Need {format_inr(required_corpus)} for lifetime expenses (vs {format_inr(retirement_goal)} goal)")
                    
                    if not recs:
                        recs.append("**Maintain Course:** Your current strategy appears well-balanced!")
                    
                    for i, rec in enumerate(recs, 1):
                        st.write(f"{i}. {rec}")
                
                with st.expander("🛡️ Risk Management & Planning", expanded=False):
                    risk_recs = [
                        "**Emergency Fund:** Maintain 6-12 months expenses separate from investment corpus",
                        "**Annual Review:** Reassess plan yearly for salary changes, inflation, and life events",
                        "**Tax Optimization:** Maximize ELSS, PPF, NPS contributions for tax efficiency",
                        "**Diversification:** Consider international equity exposure (5-10% allocation)",
                        "**Professional Guidance:** Consult SEBI-registered advisor for personalized strategy"
                    ]
                    
                    for i, rec in enumerate(risk_recs, 1):
                        st.write(f"{i}. {rec}")

            except Exception as e:
                st.error(f"❌ Simulation failed: {e}")
                st.code(traceback.format_exc())
                st.write("Please check your inputs and try again.")

    # --- Enhanced Report Generation Section
    st.markdown("---")
    st.markdown("### 📄 Comprehensive Report Generation")

    if "last_results" in st.session_state and "last_inputs" in st.session_state:
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("""
            **Generate a professional presentation-ready report including:**
            - Executive summary with key performance indicators
            - All interactive charts and visualizations
            - Detailed statistical analysis and percentile data  
            - Risk assessment and personalized recommendations
            - Methodology and assumptions documentation
            """)
        
        with col2:
            if st.button("📊 Generate Report", type="primary", use_container_width=True):
                with st.spinner("Creating comprehensive report..."):
                    try:
                        saved_results = st.session_state["last_results"]
                        saved_inputs = st.session_state["last_inputs"]
                        saved_charts = st.session_state.get("last_charts", {})
                        
                        html_report = generate_comprehensive_html_report(
                            saved_results,
                            saved_inputs, 
                            saved_charts
                        )
                        
                        st.download_button(
                            label="💾 Download HTML Report",
                            data=html_report,
                            file_name=f"FIRE_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html",
                            help="Download complete analysis report with embedded charts",
                            use_container_width=True
                        )
                        
                        st.success("✅ Report generated successfully! Click download button above.")
                        
                        with st.expander("👁️ Report Preview", expanded=False):
                            st.markdown("**Report includes:**")
                            st.markdown("- 📊 Executive dashboard with key metrics")
                            st.markdown("- 📈 All three professional charts embedded")
                            st.markdown("- 📋 Complete statistical analysis")
                            st.markdown("- 🎯 Personalized recommendations")
                            st.markdown("- 📜 Methodology and disclaimers")
                            st.markdown("- 🖨️ Print-ready professional formatting")
                        
                    except Exception as e:
                        st.error(f"❌ Report generation failed: {e}")
                        st.write("Please try running the simulation again.")
    else:
        st.info("💡 **Run a simulation first** to generate a comprehensive report.")
                
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 12px;'>
        <p><strong>🔥 FIRE Calculator v2.0</strong> | Monte Carlo Simulation for Financial Independence</p>
        <p><em>This tool is for educational purposes only. Please consult with a qualified financial advisor before making investment decisions.</em></p>
        <p>Data sources: NSE, BSE, Government Securities | Historical analysis with forward-looking projections</p>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()