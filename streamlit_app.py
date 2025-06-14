# ==========================================================
#          FIRE Calculator – Monte Carlo (India Edition)
# ==========================================================

from __future__ import annotations
import io, os, base64, tempfile, shutil
from datetime import datetime

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from weasyprint import HTML

from fire_engine import simulate_fire
from data_fetcher import load_market_data


# ------------------------ Helpers ------------------------

def format_inr(n: float) -> str:
    """Indian-style formatting into lakhs/crores."""
    if n >= 1e7:
        return f"₹{n/1e7:.2f} Cr"
    if n >= 1e5:
        return f"₹{n/1e5:.2f} L"
    return f"₹{n:,.0f}"

def fig_to_base64(fig) -> str:
    """Render a Matplotlib figure to a base64-encoded PNG for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def html_report(inputs_html: str, stats_html: str, chart_html: str) -> str:
    """Assemble full HTML for WeasyPrint."""
    CSS = """
    @page portrait { size:A4 portrait; margin:16mm; }
    @page land     { size:A4 landscape; margin:12mm; }
    body { font-family:Arial, sans-serif; }
    h1   { color:#2c3e50; }
    table,td,th { border:1px solid #ddd; border-collapse:collapse; padding:6px; }
    img.landscape { page: land; max-width: 100%; height: auto; }
    img.portrait  { max-width: 100%; height: auto; page-break-inside: avoid; }
    """
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><style>{CSS}</style></head><body>
<h1>🔥 FIRE Simulation Report – India Edition</h1>
{inputs_html}
{stats_html}
{chart_html}
<hr>
<p style="font-size:11px">
<strong>Assumptions</strong><br>
• Monthly real returns boot-strapped since 2000<br>
• SIPs added end-month; expenses deducted start-month<br>
• Equity % held constant or glide-path applied<br>
• No taxes or fees modeled<br>
<em>Generated on {datetime.now():%d-%b-%Y %H:%M}</em>
</p>
</body></html>"""

def html_to_pdf(html: str) -> str:
    """Write HTML to a temp-file PDF via WeasyPrint, return its path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        HTML(string=html).write_pdf(f.name)
        return f.name


# ------------------------ Page Setup ------------------------

st.set_page_config(page_title="🔥 FIRE Calculator India", layout="centered")
st.title("🔥 FIRE Calculator – Monte Carlo (India Edition)")

with st.expander("📘 How does this work?"):
    st.markdown("""
**Monte-Carlo Simulation**  
- We bootstrap monthly real returns from Nifty 50 & 10Y G-Sec (2000–today)  
- Each simulated path samples past months (with replacement)  
- You specify SIPs during accumulation, expenses post-retirement  
- Equity / bond blend, glide-path, inflation, dampening can all be toggled  
- Thousands of paths reveal best, median, and worst outcomes  
""")


# ------------------------ Sidebar Inputs ------------------------

sb = st.sidebar
sb.header("🧮 Customize Your Scenario")

include_acc = sb.checkbox(
    "📈 Include accumulation phase?", 
    value=True,
    help="Accumulate via SIP until retirement age if checked."
)
use_glide = sb.checkbox(
    "🎢 Use glide-path (equity → bond)?",
    value=False,
    help="Gradually shift from starting to ending equity % over time."
)
use_dampen = sb.checkbox(
    "🔻 Damp returns post-retirement?",
    value=False,
    help="Reduce returns after retirement to model more conservative yields."
)

age_now    = sb.number_input("🎯 Current Age", min_value=18, max_value=90, value=35)
age_ret    = sb.number_input("🏁 Retirement Age", min_value=age_now+1, max_value=100, value=45)
corpus0    = sb.number_input(
    "💼 Initial Corpus (₹)", 
    min_value=0.0, max_value=1e9, value=2e7, step=1e5, format="%.0f",
    help="Your existing retirement savings (in real ₹)."
)
sip        = sb.number_input(
    "💰 Monthly SIP (₹)", 
    min_value=0.0, max_value=1e7, value=30000.0, step=1000.0,
    help="Monthly contribution until retirement."
)
expense0   = sb.number_input(
    "💸 Monthly Expense (₹)", 
    min_value=0.0, max_value=1e7, value=50000.0, step=1000.0,
    help="First-month expense after retirement, before inflation."
)
infl_pct   = sb.slider(
    "📈 Annual Expense Inflation (%)", 
    min_value=0.0, max_value=15.0, value=6.0,
    help="Inflation rate applied to expenses each year."
) / 100.0
draw_years = sb.slider(
    "👵 Retirement Duration (years)", 
    min_value=5, max_value=60, value=40,
    help="How many years post-retirement to simulate."
)
eq_start   = sb.slider(
    "📊 Starting Equity Allocation (%)", 
    min_value=0, max_value=100, value=70,
    help="Portfolio % in equity during accumulation."
) / 100.0
eq_end     = (
    sb.slider("🔁 Ending Equity Allocation (%)", 0, 100, 30, help="Equity % at end of glide-path.") / 100.0
    if use_glide else eq_start
)
paths      = sb.slider(
    "🔁 Monte Carlo Paths", 
    min_value=1000, max_value=20000, value=10000, step=1000,
    help="Number of random scenarios to run."
)
goal       = sb.number_input(
    "🎯 Target Corpus at Retirement (₹)", 
    min_value=0.0, max_value=1e9, value=3e7, step=1e6, format="%.0f",
    help="Goal to measure % of runs that meet or exceed it at retirement."
)


# ------------------------ Run Simulation ------------------------

st.info("Running Monte Carlo simulations…")
res = simulate_fire(
    initial_corpus          = corpus0,
    monthly_contribution    = sip,
    monthly_expense         = expense0,
    equity_weight           = eq_start,
    n_simulations           = paths,
    include_accumulation    = include_acc,
    current_age             = age_now,
    retirement_age          = age_ret,
    total_years             = draw_years,
    monthly_expense_growth  = infl_pct,
    retirement_goal         = goal,
    equity_glide_end        = eq_end if use_glide else None,
    dampen_post_ret         = 0.8 if use_dampen else 1.0,
    start                   = "2000-01-01"
)

# unpack results
corpus_mat     = res["corpus_matrix"]
final_corpus   = res["final_corpus_distribution"]
at_retirement  = res["final_at_retirement"]
success_rate   = res["success_rate"]
goal_hit_rate  = np.mean(at_retirement >= goal)


# ------------------------ Display Metrics ------------------------

st.subheader("📊 Key Metrics")
failures = (final_corpus <= 0).sum()
p10, p50, p90 = np.percentile(final_corpus, [10, 50, 90])

col1, col2, col3 = st.columns(3)
col1.metric("✅ Success Rate", f"{success_rate*100:.2f}%")
col2.metric("🎯 Goal Hit Rate", f"{goal_hit_rate*100:.2f}%")
col3.metric("📉 Ruined Runs", f"{failures} / {paths}")

st.write(f"Median Final Corpus: {format_inr(p50)}")
st.write(f"10th Percentile (P10): {format_inr(p10)}  *(pessimistic)*")
st.write(f"90th Percentile (P90): {format_inr(p90)}  *(optimistic)*")


# ------------------------ Charts ------------------------

years = np.arange(corpus_mat.shape[1]) / 12.0
n_show = min(300, corpus_mat.shape[0])

# 1) Sample paths
fig1, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(years, corpus_mat[:n_show].T, color="blue", alpha=0.03)
ax1.set_title("Monte Carlo Corpus Paths")
ax1.set_xlabel("Years"); ax1.set_ylabel("Corpus (₹)")
st.pyplot(fig1)
st.caption("Each faint line = one simulated path; early crashes highlight sequence risk.")

# 2) Burn-curve percentiles
p10_curve = np.percentile(corpus_mat, 10, axis=0)
p50_curve = np.percentile(corpus_mat, 50, axis=0)
p90_curve = np.percentile(corpus_mat, 90, axis=0)

fig2, ax2 = plt.subplots(figsize=(9, 4))
ax2.plot(years, p50_curve, label="Median (P50)", color="blue")
ax2.plot(years, p10_curve, '--',   label="P10",        color="red")
ax2.plot(years, p90_curve, '--',   label="P90",        color="green")
ax2.set_title("Corpus Burn Curve (P10/P50/P90)")
ax2.set_xlabel("Years"); ax2.set_ylabel("Corpus (₹)")
ax2.legend()
st.pyplot(fig2)
st.caption("Red = worst-10% path; if it hits ₹0, you have a 10% ruin risk by that year.")

# 3) Final corpus histogram
fig3, ax3 = plt.subplots(figsize=(7, 4))
ax3.hist(final_corpus/1e6, bins=40, color="green", alpha=0.7)
ax3.set_title("Final Corpus Distribution")
ax3.set_xlabel("Final Corpus (₹ Crore)")
ax3.set_ylabel("Number of Simulations")
st.pyplot(fig3)
st.caption("Left-most bar at ₹0 shows count of ruined runs.")


# ------------------------ PDF Export ------------------------

# 1) build HTML fragments
inputs_html = "<h2>User Inputs</h2><table>" + "".join(
    f"<tr><td>{key}</td><td>{val}</td></tr>"
    for key, val in {
        "Current Age":             age_now,
        "Retirement Age":          age_ret,
        "Initial Corpus":          format_inr(corpus0),
        "Monthly SIP":             format_inr(sip),
        "First-Month Expense":     format_inr(expense0),
        "Expense Inflation":       f"{infl_pct*100:.1f}%",
        "Equity Start":            f"{eq_start*100:.0f}%",
        "Equity End":              f"{eq_end*100:.0f}%",
        "Include Accumulation":    include_acc,
        "Glide Path":              use_glide,
        "Post-Retirement Dampening": use_dampen,
        "Drawdown Years":          draw_years,
        "Monte Carlo Paths":       paths,
        "Target Corpus":           format_inr(goal),
    }.items()
) + "</table>"

stats_html = f"""
<h2>Key Results</h2>
<table>
<tr><td>Success Rate</td><td>{success_rate*100:.2f}%</td></tr>
<tr><td>Goal Hit Rate</td><td>{goal_hit_rate*100:.2f}%</td></tr>
<tr><td>Failures</td><td>{failures} / {paths}</td></tr>
<tr><td>Median Final Corpus</td><td>{format_inr(p50)}</td></tr>
<tr><td>P10 Final Corpus</td><td>{format_inr(p10)}</td></tr>
<tr><td>P90 Final Corpus</td><td>{format_inr(p90)}</td></tr>
</table>
"""

chart_html = f"""
<h2>Charts & Explanations</h2>
<h3>1. Monte Carlo Paths</h3>
<p>Each thin line is one scenario; early crashes show sequence-of-returns risk.</p>
<img src="data:image/png;base64,{fig_to_base64(fig1)}" class="landscape">
<h3>2. Corpus Burn Curve</h3>
<p>Blue=Median, Red=P10 (pessimistic), Green=P90 (optimistic)</p>
<img src="data:image/png;base64,{fig_to_base64(fig2)}" class="landscape">
<h3>3. Final Corpus Histogram</h3>
<p>Left bar at ₹0 shows count of ruined runs.</p>
<img src="data:image/png;base64,{fig_to_base64(fig3)}" class="portrait">
"""

# if st.button("📄 Generate PDF Report"):
#     html = html_report(inputs_html, stats_html, charts_html)
#     pdf_path = html_to_pdf(html)
#     with open(pdf_path, "rb") as f:
#         st.download_button(
#             label="📥 Download Report",
#             data=f.read(),
#             file_name="FIRE_Report.pdf",
#             mime="application/pdf"
#         )
#     shutil.rmtree(os.path.dirname(pdf_path), ignore_errors=True)

