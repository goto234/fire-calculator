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
    start: str = "2000-01-01"
) -> dict:
    """Simulate FIRE trajectories using historical bootstrapping."""

    # -------- Optional: Rename for clarity --------
    annual_expense_infl = monthly_expense_growth

    # -------- Validate Inputs --------
    assert 18 <= current_age < retirement_age <= 100
    assert 0 <= equity_weight <= 1
    assert 0 < total_years <= 100
    assert n_simulations >= 100
    assert monthly_contribution >= 0
    assert monthly_expense >= 0
    assert 0 <= monthly_expense_growth <= 1
    assert dampen_post_ret > 0

    # -------- Load market data --------
    df = load_market_data(start)
    eq = df["equity_return"].values
    bond = df["bond_return"].values
    rng = np.random.default_rng(seed=42)

    # -------- Setup --------
    acc_months = 12 * max(0, retirement_age - current_age)
    ret_months = 12 * total_years
    total_months = acc_months + ret_months

    corpus_matrix = np.zeros((n_simulations, total_months))
    final_at_retirement = np.zeros(n_simulations)

    if equity_glide_end is None:
        equity_glide_end = equity_weight  # no glide path, use constant weight

    eq_weights = np.linspace(equity_weight, equity_glide_end, total_months)

    # -------- Simulation --------
    for i in range(n_simulations):
        corpus = initial_corpus
        expense = monthly_expense
        path = []

        for m in range(total_months):
            idx = rng.integers(0, len(eq))
            eq_ret = eq[idx]
            bond_ret = bond[idx]

            w_eq = eq_weights[m]
            blended_ret = w_eq * eq_ret + (1 - w_eq) * bond_ret

            if m >= acc_months:
                blended_ret *= dampen_post_ret
                corpus -= expense
                expense *= (1 + annual_expense_infl / 12)
            else:
                if include_accumulation:
                    corpus += monthly_contribution

            corpus *= (1 + blended_ret)
            corpus = max(corpus, 0)
            path.append(corpus)

        corpus_matrix[i, :] = path
        final_at_retirement[i] = path[acc_months-1] if acc_months > 0 else corpus

    # -------- Metrics --------
    final = corpus_matrix[:, -1]
    success_rate = np.mean(final > 0)

    return {
        "corpus_matrix": corpus_matrix,
        "final_corpus_distribution": final,
        "final_at_retirement": final_at_retirement,
        "success_rate": success_rate
    }
