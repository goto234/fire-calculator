import numpy as np
import pandas as pd
from data_fetcher import load_market_data


def simulate_fire(
    initial_corpus: float,
    monthly_contribution: float,
    monthly_expense: float,
    equity_weight: float,
    n_simulations: int,
    include_accumulation: bool,
    current_age: int,
    retirement_age: int,
    total_years: int,
    monthly_expense_growth: float,
    retirement_goal: float,
    equity_glide_end: float = None,
    dampen_post_ret: float = 1.0,
    equity_ticker: str = "^NSEI",
    bond_series: str = "INDIRLTLT01STQ",
    return_window: int = 15,
    equity_stress: float = 0.0,
    bond_stress: float = 0.0,
    verbose: bool = False
) -> dict:
    """
    Simulate FIRE trajectories using historical bootstrapping with stress testing.
    
    Parameters:
    -----------
    initial_corpus : float
        Starting investment corpus
    monthly_contribution : float
        Monthly SIP amount during accumulation
    monthly_expense : float
        First month expense in retirement
    equity_weight : float
        Starting equity allocation (0-1)
    n_simulations : int
        Number of Monte Carlo paths
    include_accumulation : bool
        Whether to include SIP contributions
    current_age : int
        Current age
    retirement_age : int
        Target retirement age
    total_years : int
        Years to simulate post-retirement
    monthly_expense_growth : float
        Annual real expense inflation rate
    retirement_goal : float
        Target corpus at retirement
    equity_glide_end : float, optional
        Ending equity allocation for glide path
    dampen_post_ret : float
        Post-retirement return dampening factor
    equity_ticker : str
        Yahoo Finance equity ticker
    bond_series : str
        FRED bond series ID
    return_window : int
        Historical data window in years
    equity_stress : float
        Equity return stress adjustment (-1 to 1)
    bond_stress : float
        Bond return stress adjustment (-1 to 1)
    verbose : bool
        Enable verbose logging
        
    Returns:
    --------
    dict with simulation results
    """
    
    # Input validation
    assert 18 <= current_age < retirement_age <= 100, "Invalid age parameters"
    assert 0 <= equity_weight <= 1, "Equity weight must be 0-1"
    assert 0 < total_years <= 100, "Invalid simulation years"
    assert n_simulations >= 100, "Need at least 100 simulations"
    assert monthly_contribution >= 0, "SIP must be non-negative"
    assert monthly_expense >= 0, "Expense must be non-negative"
    assert 0 <= monthly_expense_growth <= 1, "Invalid expense growth"
    assert dampen_post_ret > 0, "Dampening must be positive"
    assert -1 <= equity_stress <= 1, "Equity stress must be -1 to 1"
    assert -1 <= bond_stress <= 1, "Bond stress must be -1 to 1"
    
    # Load market data
    try:
        market_data = load_market_data(
            equity_ticker=equity_ticker,
            bond_series=bond_series,
            return_window=return_window,
            verbose=verbose
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load market data: {e}")
    
    if market_data.empty:
        raise RuntimeError("No market data available")
    
    # Apply stress testing
    equity_returns = market_data["equity_return"].values * (1 + equity_stress)
    bond_returns = market_data["bond_return"].values * (1 + bond_stress)
    
    # Setup simulation parameters
    accumulation_months = 12 * max(0, retirement_age - current_age)
    retirement_months = 12 * total_years
    total_months = accumulation_months + retirement_months
    
    # Initialize arrays
    corpus_matrix = np.zeros((n_simulations, total_months))
    final_at_retirement = np.zeros(n_simulations)
    
    # Setup glide path
    if equity_glide_end is None:
        equity_glide_end = equity_weight
    
    equity_weights = np.linspace(equity_weight, equity_glide_end, total_months)
    
    # Random number generator
    rng = np.random.default_rng(seed=42)
    
    # Run simulations
    for sim in range(n_simulations):
        corpus = initial_corpus
        monthly_expense_current = monthly_expense
        path = []
        
        for month in range(total_months):
            # Sample random market returns
            idx = rng.integers(0, len(equity_returns))
            eq_ret = equity_returns[idx]
            bond_ret = bond_returns[idx]
            
            # Calculate blended return
            w_equity = equity_weights[month]
            blended_return = w_equity * eq_ret + (1 - w_equity) * bond_ret
            
            # Handle accumulation vs retirement phase
            if month >= accumulation_months:
                # Retirement phase
                blended_return *= dampen_post_ret
                corpus -= monthly_expense_current
                monthly_expense_current *= (1 + monthly_expense_growth / 12)
            else:
                # Accumulation phase
                if include_accumulation:
                    corpus += monthly_contribution
            
            # Apply returns
            corpus *= (1 + blended_return)
            corpus = max(corpus, 0)  # Cannot go negative
            path.append(corpus)
        
        corpus_matrix[sim, :] = path
        final_at_retirement[sim] = (
            path[accumulation_months - 1] if accumulation_months > 0 else corpus
        )
    
       # Calculate metrics
    final_corpus     = corpus_matrix[:, -1]
    success_rate     = np.mean(final_corpus > 0)
    goal_hit_rate    = np.mean(final_at_retirement >= retirement_goal)
    ruin_count       = np.sum(final_corpus <= 0)

    # ── Compute median years to ruin ──
    # For each sim, find first month corpus ≤ 0 (or total_months if never ruined)
    months_to_ruin      = np.argmax(corpus_matrix <= 0, axis=1)
    never_ruined        = (corpus_matrix <= 0).sum(axis=1) == 0
    months_to_ruin[never_ruined] = total_months
    median_years_to_ruin = np.median(months_to_ruin) / 12
    # ────────────────────────────────────

    # Calculate Safe Withdrawal Rate (SWR)
    p10_final    = np.percentile(final_corpus, 10)
    swr_annual   = p10_final * 0.04
    swr_monthly  = swr_annual / 12

    mean_final   = np.mean(final_corpus)

    return {
        "corpus_matrix":                 corpus_matrix,
        "final_corpus_distribution":     final_corpus,
        "final_at_retirement":           final_at_retirement,
        "mean_final":                    mean_final,
        "success_rate":                  success_rate,
        "goal_hit_rate":                 goal_hit_rate,
        "median_years_to_ruin":          median_years_to_ruin,
        "ruin_count":                    ruin_count,
        "swr_monthly":                   swr_monthly,
        "swr_annual":                    swr_annual,
        "corpus_at_retirement_p10":      np.percentile(final_corpus, 10),
        "corpus_at_retirement_p50":      np.percentile(final_corpus, 50),
        "corpus_at_retirement_p90":      np.percentile(final_corpus, 90),
        # aliases for existing UI code
        "p10_final":                     np.percentile(final_corpus, 10),
        "p50_final":                     np.percentile(final_corpus, 50),
        "p90_final":                     np.percentile(final_corpus, 90),
        "simulation_params": {
            "n_simulations":     n_simulations,
            "total_months":      total_months,
            "accumulation_months": accumulation_months,
            "retirement_months": retirement_months,
            "equity_stress":     equity_stress,
            "bond_stress":       bond_stress,
            "return_window":     return_window,
            "equity_ticker":     equity_ticker,
            "bond_series":       bond_series
        }
    }
