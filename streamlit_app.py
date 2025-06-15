"""
🔥 FIRE Calculator – Monte-Carlo Simulation (India Edition)
Complete implementation with all required features
"""

import io
import base64
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from fire_engine import simulate_fire
from data_fetcher import get_preset_scenarios
import traceback


# -----------------------------------------------------------------------------
# App Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="🔥 FIRE Monte-Carlo Calculator",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Matplotlib styling
plt.rcParams.update({
    "axes.facecolor": "#f9f9f9",
    "figure.facecolor": "#ffffff",
    "axes.grid": True,
    "grid.color": "#d8d8d8",
    "grid.alpha": 0.7,
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "figure.autolayout": True,
})

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def format_inr(val):
    """Format currency in Indian notation."""
    if val >= 1e7:
        return f"₹{val/1e7:.2f} Cr"
    elif val >= 1e5:
        return f"₹{val/1e5:.2f} L"
    else:
        return f"₹{val:,.0f}"


def fig_to_b64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160, facecolor='white')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def create_corpus_paths_chart(corpus_matrix, years):
    """Create corpus paths visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot sample of paths
    sample_size = min(300, corpus_matrix.shape[0])
    sample_indices = np.random.choice(corpus_matrix.shape[0], sample_size, replace=False)
    
    colors = plt.cm.Set2(0)
    ax.plot(years, corpus_matrix[sample_indices].T, alpha=0.03, color=colors, linewidth=0.5)
    
    # Add median line
    median_path = np.percentile(corpus_matrix, 50, axis=0)
    ax.plot(years, median_path, color='#d62728', linewidth=2, label='Median Path')
    
    ax.set_title("Simulated Corpus Paths (Sample of 300)", pad=20)
    ax.set_xlabel("Years from Start")
    ax.set_ylabel("Corpus Value (₹)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_inr(x)))
    
    return fig


def create_burn_curve_chart(corpus_matrix, years, retirement_start_year):
    """Create burn curve (percentile) visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set2([0, 1, 2])
    
    p10 = np.percentile(corpus_matrix, 10, axis=0)
    p50 = np.percentile(corpus_matrix, 50, axis=0)
    p90 = np.percentile(corpus_matrix, 90, axis=0)
    
    ax.plot(years, p50, label="P50 (Median)", color=colors[0], linewidth=2)
    ax.plot(years, p10, label="P10 (Pessimistic)", color=colors[1], linestyle="--")
    ax.plot(years, p90, label="P90 (Optimistic)", color=colors[2], linestyle="--")
    
    # Add retirement line
    if retirement_start_year > 0:
        ax.axvline(x=retirement_start_year, color='red', linestyle=':', alpha=0.7, label='Retirement Start')
    
    ax.fill_between(years, p10, p90, alpha=0.1, color=colors[0])
    
    ax.set_title("Burn Curve: P10 / P50 / P90 Outcomes", pad=20)
    ax.set_xlabel("Years from Start")
    ax.set_ylabel("Corpus Value (₹)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_inr(x)))
    
    return fig


def create_histogram_chart(final_dist):
    """Create final corpus distribution histogram."""
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Convert to crores for better readability
    final_cr = final_dist / 1e7
    
    n, bins, patches = ax.hist(final_cr, bins=40, color="#00897b", alpha=0.7, 
                              edgecolor="white", linewidth=0.8)
    
    # Highlight ruin cases (≤ 0)
    ruin_mask = bins[:-1] <= 0
    for i, patch in enumerate(patches):
        if i < len(ruin_mask) and ruin_mask[i]:
            patch.set_color("#d32f2f")
            patch.set_alpha(0.8)
    
    ax.set_title("Final Corpus Distribution", pad=20)
    ax.set_xlabel("Final Corpus (₹ Crores)")
    ax.set_ylabel("Number of Simulations")
    
    # Add statistics
    mean_val = np.mean(final_cr)
    median_val = np.median(final_cr)
    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: ₹{mean_val:.1f}Cr')
    ax.axvline(median_val, color='blue', linestyle='--', alpha=0.7, label=f'Median: ₹{median_val:.1f}Cr')
    ax.legend()
    
    return fig


def build_html_report(results, inputs, version="2.0"):
    """Generate comprehensive HTML report."""
    
    # Extract data
    r = results
    params = r["simulation_params"]
    
    # CSS styling
    css = """
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #fefefe; color: #333; margin: 20px; }
    h1 { color: #1f4e79; border-bottom: 3px solid #ff6b35; padding-bottom: 8px; }
    h2 { color: #2c3e50; border-bottom: 1px solid #bdc3c7; padding-bottom: 4px; margin-top: 30px; }
    h3 { color: #34495e; margin-top: 25px; }
    table { border-collapse: separate; border-spacing: 0; border: 1px solid #bdc3c7; 
            border-radius: 8px; overflow: hidden; margin: 15px 0; width: 100%; }
    th, td { padding: 12px 15px; border-bottom: 1px solid #ecf0f1; text-align: left; }
    th { background: #e8f4ff; font-weight: bold; color: #2c3e50; }
    tr:last-child td { border-bottom: none; }
    tr:nth-child(even) { background: #f8f9fa; }
    img { max-width: 100%; height: auto; margin: 15px 0; border-radius: 8px; 
          box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .metric-box { background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; 
                  border-left: 4px solid #3498db; }
    .warning { background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 10px 0; }
    .footer { font-size: 11px; color: #7f8c8d; margin-top: 40px; padding-top: 20px; 
              border-top: 1px solid #ecf0f1; }
    """
    
    # Build HTML sections
    inputs_html = f"""
    <h2>📊 User Inputs</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Current Age → Retirement Age</td><td>{inputs['current_age']} → {inputs['retirement_age']}</td></tr>
        <tr><td>Years Post-Retirement</td><td>{inputs['total_years']}</td></tr>
        <tr><td>Initial Corpus</td><td>{format_inr(inputs['initial_corpus'])}</td></tr>
        <tr><td>Monthly SIP</td><td>{format_inr(inputs['monthly_contribution'])}</td></tr>
        <tr><td>SIP Accumulation</td><td>{'Enabled' if inputs['include_accumulation'] else 'Disabled'}</td></tr>
        <tr><td>Monthly Expense (Year 1)</td><td>{format_inr(inputs['monthly_expense'])}</td></tr>
        <tr><td>Expense Inflation (Real)</td><td>{inputs['monthly_expense_growth']*100:.1f}% p.a.</td></tr>
        <tr><td>Equity Allocation</td><td>{inputs['equity_weight']*100:.0f}%{' → ' + str(int(inputs.get('equity_glide_end', inputs['equity_weight'])*100)) + '%' if inputs.get('equity_glide_end') != inputs['equity_weight'] else ''}</td></tr>
        <tr><td>Post-Retirement Dampening</td><td>{inputs['dampen_post_ret']*100:.0f}%</td></tr>
        <tr><td>Retirement Goal</td><td>{format_inr(inputs['retirement_goal'])}</td></tr>
        <tr><td>Simulation Paths</td><td>{params['n_simulations']:,}</td></tr>
        <tr><td>Return Window</td><td>{params['return_window']} years</td></tr>
        <tr><td>Equity Ticker</td><td>{params['equity_ticker']}</td></tr>
        <tr><td>Bond Series</td><td>{params['bond_series']}</td></tr>
    </table>
    """
    
    stress_info = ""
    if params['equity_stress'] != 0 or params['bond_stress'] != 0:
        stress_info = f"""
        <div class="warning">
        <strong>⚠️ Stress Test Applied:</strong> Equity {params['equity_stress']*100:+.0f}%, 
        Bond {params['bond_stress']*100:+.0f}%
        </div>
        """
    
    results_html = f"""
    <h2>📈 Simulation Results</h2>
    {stress_info}
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Success Rate (Never Broke)</td><td>{r['success_rate']*100:.2f}%</td></tr>
        <tr><td>Goal Hit Rate (At Retirement)</td><td>{r['goal_hit_rate']*100:.2f}%</td></tr>
        <tr><td>Ruined Simulations</td><td>{r['ruin_count']:,} / {params['n_simulations']:,}</td></tr>
        <tr><td>P10 Final Corpus (Pessimistic)</td><td>{format_inr(r['p10_final'])}</td></tr>
        <tr><td>P50 Final Corpus (Median)</td><td>{format_inr(r['p50_final'])}</td></tr>
        <tr><td>P90 Final Corpus (Optimistic)</td><td>{format_inr(r['p90_final'])}</td></tr>
        <tr><td>Mean Final Corpus</td><td>{format_inr(r['mean_final'])}</td></tr>
        <tr><td>Median Years to Ruin</td><td>{r['median_years_to_ruin']:.1f} years</td></tr>
        <tr><td>Corpus at Retirement (P50)</td><td>{format_inr(r['corpus_at_retirement_p50'])}</td></tr>
    </table>
    """
    
    # Generate charts
    years = np.arange(len(r['corpus_matrix'][0]))
    retirement_start_year = inputs['retirement_age'] - inputs['current_age']
    
    corpus_paths_fig = create_corpus_paths_chart(r['corpus_matrix'], years)
    burn_curve_fig = create_burn_curve_chart(r['corpus_matrix'], years, retirement_start_year)
    histogram_fig = create_histogram_chart(r['final_corpus_distribution'])
    
    # Convert charts to base64
    corpus_paths_b64 = fig_to_b64(corpus_paths_fig)
    burn_curve_b64 = fig_to_b64(burn_curve_fig)
    histogram_b64 = fig_to_b64(histogram_fig)
    
    charts_html = f"""
    <h2>📊 Visualizations</h2>
    
    <h3>Corpus Evolution Paths</h3>
    <img src="data:image/png;base64,{corpus_paths_b64}" alt="Corpus Paths Chart">
    
    <h3>Burn Curve Analysis</h3>
    <img src="data:image/png;base64,{burn_curve_b64}" alt="Burn Curve Chart">
    
    <h3>Final Corpus Distribution</h3>
    <img src="data:image/png;base64,{histogram_b64}" alt="Histogram Chart">
    """
    
    # Analysis section
    success_color = "#28a745" if r['success_rate'] > 0.9 else "#ffc107" if r['success_rate'] > 0.7 else "#dc3545"
    goal_color = "#28a745" if r['goal_hit_rate'] > 0.8 else "#ffc107" if r['goal_hit_rate'] > 0.5 else "#dc3545"
    
    analysis_html = f"""
    <h2>🔍 Analysis & Insights</h2>
    
    <div class="metric-box">
        <h3 style="color: {success_color};">Success Rate: {r['success_rate']*100:.1f}%</h3>
        <p>This represents the percentage of simulations where your corpus never went to zero throughout the entire period.</p>
        {'<p style="color: #28a745;"><strong>✅ Excellent:</strong> Your plan shows strong resilience.</p>' if r['success_rate'] > 0.9 else 
         '<p style="color: #ffc107;"><strong>⚠️ Moderate:</strong> Consider increasing contributions or reducing expenses.</p>' if r['success_rate'] > 0.7 else 
         '<p style="color: #dc3545;"><strong>❌ Poor:</strong> Significant adjustments needed to avoid running out of money.</p>'}
    </div>
    
    <div class="metric-box">
        <h3 style="color: {goal_color};">Goal Achievement: {r['goal_hit_rate']*100:.1f}%</h3>
        <p>Percentage of simulations that reached your retirement goal of {format_inr(inputs['retirement_goal'])} at retirement.</p>
        {'<p style="color: #28a745;"><strong>✅ Excellent:</strong> High probability of meeting your target.</p>' if r['goal_hit_rate'] > 0.8 else 
         '<p style="color: #ffc107;"><strong>⚠️ Moderate:</strong> Consider increasing SIP or extending working years.</p>' if r['goal_hit_rate'] > 0.5 else 
         '<p style="color: #dc3545;"><strong>❌ Poor:</strong> Goal may be too ambitious or contributions too low.</p>'}
    </div>
    
    <h3>Key Observations</h3>
    <ul>
        <li><strong>Risk Assessment:</strong> {r['ruin_count']:,} out of {params['n_simulations']:,} simulations resulted in running out of money.</li>
        <li><strong>Corpus Range:</strong> At the end of the period, 80% of outcomes fall between {format_inr(r['p10_final'])} and {format_inr(r['p90_final'])}.</li>
        <li><strong>Retirement Corpus:</strong> You're likely to have {format_inr(r['corpus_at_retirement_p50'])} when you retire (median scenario).</li>
        <li><strong>Buffer Analysis:</strong> The difference between P90 and P10 outcomes shows the impact of market volatility on your plan.</li>
    </ul>
    
    <h3>Recommendations</h3>
    <ul>
        <li>{'<strong>Maintain Course:</strong> Your current plan shows strong probability of success.' if r['success_rate'] > 0.9 else 
            '<strong>Increase SIP:</strong> Consider raising monthly contributions by 20-30%.' if r['success_rate'] > 0.7 else 
            '<strong>Major Revision Needed:</strong> Either increase contributions significantly, reduce expenses, or extend working years.'}</li>
        <li>{'<strong>Goal on Track:</strong> Your retirement target appears achievable.' if r['goal_hit_rate'] > 0.8 else 
            '<strong>Adjust Expectations:</strong> Consider a more realistic retirement goal or increase savings rate.'}</li>
        <li><strong>Regular Review:</strong> Reassess your plan annually and adjust for inflation, salary changes, and life events.</li>
        <li><strong>Emergency Fund:</strong> Maintain 6-12 months of expenses separate from this corpus for unexpected events.</li>
    </ul>
    """
    
    # Assumptions section
    assumptions_html = f"""
    <h2>📋 Assumptions & Methodology</h2>
    
    <h3>Return Assumptions</h3>
    <ul>
        <li><strong>Equity Returns:</strong> Based on {params['equity_ticker']} historical data with {params['return_window']}-year rolling windows</li>
        <li><strong>Bond Returns:</strong> Based on {params['bond_series']} series data</li>
        <li><strong>Rebalancing:</strong> Annual rebalancing to target allocation</li>
        <li><strong>Tax Impact:</strong> Not explicitly modeled (assumes tax-efficient investing)</li>
    </ul>
    
    <h3>Expense Modeling</h3>
    <ul>
        <li><strong>Inflation:</strong> {inputs['monthly_expense_growth']*100:.1f}% real growth in expenses per year</li>
        <li><strong>Post-Retirement:</strong> {inputs['dampen_post_ret']*100:.0f}% reduction in expenses after retirement</li>
        <li><strong>Timing:</strong> Expenses withdrawn at the beginning of each year</li>
    </ul>
    
    <h3>Simulation Details</h3>
    <ul>
        <li><strong>Monte Carlo Paths:</strong> {params['n_simulations']:,} independent simulations</li>
        <li><strong>Return Sampling:</strong> Bootstrap sampling from historical return distributions</li>
        <li><strong>Sequence Risk:</strong> Modeled through random ordering of historical returns</li>
        <li><strong>Correlation:</strong> Historical equity-bond correlations preserved</li>
    </ul>
    
    <div class="warning">
        <strong>⚠️ Important Disclaimers:</strong>
        <ul>
            <li>Past performance does not guarantee future results</li>
            <li>Real-world returns may differ significantly from historical patterns</li>
            <li>Consider tax implications, transaction costs, and other real-world factors</li>
            <li>This is for educational purposes only and not financial advice</li>
            <li>Consult a qualified financial advisor for personalized guidance</li>
        </ul>
    </div>
    """
    
    # Footer
    footer_html = f"""
    <div class="footer">
        <p><strong>🔥 FIRE Calculator v{version}</strong> | Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        <p>Monte Carlo simulation with {params['n_simulations']:,} paths | Data sources: {params['equity_ticker']}, {params['bond_series']}</p>
        <p><em>This analysis is for educational purposes only. Please consult with a qualified financial advisor before making investment decisions.</em></p>
    </div>
    """
    
    # Combine all sections
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🔥 FIRE Calculator Report</title>
        <style>{css}</style>
    </head>
    <body>
        <h1>🔥 FIRE Calculator - Monte Carlo Analysis Report</h1>
        
        {inputs_html}
        {results_html}
        {charts_html}
        {analysis_html}
        {assumptions_html}
        {footer_html}
    </body>
    </html>
    """
    
    return html_content


# -----------------------------------------------------------------------------
# Main Streamlit App
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
    # st.rerun()          # <- use this if your Streamlit version supports it

    # ── Single dropdown – defaults to “Custom” ──
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
                "monthly_expense_growth": 2.0,   # slider shows %; we’ll divide by 100 later
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
    # 3.  VALIDATION  (after widgets, before simulate)
    # ────────────────────────────────────────────────────────────────
    if retirement_age <= current_age:
        st.error("Retirement age must be greater than current age!")
        st.stop()
    if monthly_expense * 12 > retirement_goal:
        st.warning("Your annual expenses exceed your retirement goal.")

    # … your simulation-button / results / report code continues here …

    
    # Prepare inputs
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
    
    simulation_params = {
        'n_simulations': n_simulations,
        'return_window': return_window,
        'equity_ticker': equity_ticker,
        'bond_series': bond_series,
        'equity_stress': equity_stress,
        'bond_stress': bond_stress,
    }
    
    # Run simulation button
    #st.write("🛠️ Debug inputs:", inputs)
    #st.write("🛠️ Debug sim params:", simulation_params)

    try:
        results = simulate_fire(**inputs, **simulation_params)
    except Exception:
        st.error("❌ Simulation failed; here’s the full traceback:")
        st.code(traceback.format_exc())
        st.stop()
    if st.button("🚀 Run Monte Carlo Simulation", type="primary"):
        with st.spinner("Running simulation... This may take a few moments."):
            try:
                # Run the simulation
                results = simulate_fire(**inputs, **simulation_params)
                
                # Display results
                st.success("✅ Simulation completed successfully!")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Success Rate", f"{results['success_rate']*100:.1f}%")
                with col2:
                    st.metric("Goal Hit Rate", f"{results['goal_hit_rate']*100:.1f}%")
                with col3:
                    st.metric("Median Final Corpus", format_inr(results['p50_final']))
                with col4:
                    st.metric("Retirement Corpus (P50)", format_inr(results['corpus_at_retirement_p50']))
                
                # Charts
                years = np.arange(len(results['corpus_matrix'][0]))
                retirement_start_year = retirement_age - current_age
                
                # Corpus paths chart
                st.subheader("📊 Corpus Evolution Paths")
                corpus_fig = create_corpus_paths_chart(results['corpus_matrix'], years)
                st.pyplot(corpus_fig)
                
                # Burn curve
                st.subheader("📉 Burn Curve Analysis")
                burn_fig = create_burn_curve_chart(results['corpus_matrix'], years, retirement_start_year)
                st.pyplot(burn_fig)
                
                # Histogram
                st.subheader("📊 Final Corpus Distribution")
                hist_fig = create_histogram_chart(results['final_corpus_distribution'])
                st.pyplot(hist_fig)
                
                # Detailed results table
                st.subheader("📋 Detailed Results")
                results_df = pd.DataFrame({
                    'Metric': [
                        'Success Rate (%)',
                        'Goal Hit Rate (%)',
                        'Ruined Simulations',
                        'P10 Final Corpus',
                        'P25 Final Corpus', 
                        'P50 Final Corpus (Median)',
                        'P75 Final Corpus',
                        'P90 Final Corpus',
                        'Mean Final Corpus',
                        'Std Dev Final Corpus',
                        'Median Years to Ruin',
                        'Corpus at Retirement (P50)',
                        'Corpus at Retirement (P10)',
                        'Corpus at Retirement (P90)'
                    ],
                    'Value': [
                        f"{results['success_rate']*100:.2f}%",
                        f"{results['goal_hit_rate']*100:.2f}%",
                        f"{results['ruin_count']:,} / {n_simulations:,}",
                        format_inr(results['p10_final']),
                        format_inr(np.percentile(results['final_corpus_distribution'], 25)),
                        format_inr(results['p50_final']),
                        format_inr(np.percentile(results['final_corpus_distribution'], 75)),
                        format_inr(results['p90_final']),
                        format_inr(results['mean_final']),
                        format_inr(np.std(results['final_corpus_distribution'])),
                        f"{results['median_years_to_ruin']:.1f} years",
                        format_inr(results['corpus_at_retirement_p50']),
                        format_inr(results.get('corpus_at_retirement_p10', 0)),
                        format_inr(results.get('corpus_at_retirement_p90', 0))
                    ]
                })
                st.dataframe(results_df, use_container_width=True)
                
                # Analysis and recommendations
                st.subheader("🔍 Analysis & Recommendations")
                
                # Success rate analysis
                if results['success_rate'] > 0.95:
                    st.success("🎉 **Excellent Plan!** Your current strategy shows very high probability of success.")
                elif results['success_rate'] > 0.85:
                    st.info("✅ **Good Plan!** Your strategy shows strong probability of success with minor room for improvement.")
                elif results['success_rate'] > 0.70:
                    st.warning("⚠️ **Moderate Risk!** Consider increasing contributions or reducing expenses to improve success rate.")
                else:
                    st.error("❌ **High Risk!** Significant adjustments needed - consider increasing SIP, reducing expenses, or extending working years.")
                
                # Goal achievement analysis
                if results['goal_hit_rate'] > 0.80:
                    st.success("🎯 **Goal Achievable!** High probability of reaching your retirement target.")
                elif results['goal_hit_rate'] > 0.60:
                    st.info("🎯 **Goal Likely!** Good chance of reaching your retirement target.")
                elif results['goal_hit_rate'] > 0.40:
                    st.warning("🎯 **Goal Challenging!** Consider increasing SIP or adjusting expectations.")
                else:
                    st.error("🎯 **Goal Unrealistic!** Significant changes needed to reach retirement target.")
                
                # Specific recommendations
                st.subheader("💡 Specific Recommendations")
                
                recommendations = []
                
                if results['success_rate'] < 0.85:
                    current_sip_annual = monthly_contribution * 12
                    recommended_increase = 0.3 if results['success_rate'] < 0.7 else 0.2
                    new_sip = current_sip_annual * (1 + recommended_increase)
                    recommendations.append(f"📈 **Increase SIP:** Consider raising monthly SIP from {format_inr(monthly_contribution)} to {format_inr(new_sip/12)} ({recommended_increase*100:.0f}% increase)")
                
                if results['goal_hit_rate'] < 0.6:
                    recommendations.append("🎯 **Adjust Goal:** Consider reducing retirement target or extending working years by 2-3 years")
                
                if equity_weight < 0.7 and current_age < 40:
                    recommendations.append("📊 **Increase Equity:** Consider higher equity allocation for better long-term growth potential")
                
                if monthly_expense * 12 * 25 > retirement_goal:
                    recommendations.append("💸 **Expense Review:** Your retirement goal may be insufficient for your planned expenses. Consider the 25x rule.")
                
                recommendations.append("📅 **Regular Review:** Reassess your plan annually and adjust for salary changes, inflation, and life events")
                recommendations.append("🆘 **Emergency Fund:** Maintain 6-12 months of expenses in a separate emergency fund")
                recommendations.append("📋 **Professional Advice:** Consult a qualified financial advisor for personalized guidance")
                
                for rec in recommendations:
                    st.write(f"• {rec}")
                
                # Generate and offer HTML report
                
                # Store results in session state for persistence
                st.session_state['last_results'] = results
                st.session_state['last_inputs'] = inputs
                
            except Exception as e:
                st.error(f"❌ Simulation failed: {str(e)}")
                st.write("Please check your inputs and try again.")
    st.subheader("📄 Generate Report")
                
    if st.button("📄 Generate Detailed HTML Report"):
                    with st.spinner("Generating comprehensive report..."):
                        html_report = build_html_report(results, inputs)
                        
                        # Offer download
                        st.download_button(
                            label="💾 Download HTML Report",
                            data=html_report,
                            file_name=f"FIRE_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html"
                        )
                        
                        
                        st.success("✅ Report generated! Click the download button above to save.")
                
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