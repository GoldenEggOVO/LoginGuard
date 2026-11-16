import pandas as pd


def detect_rules(features, failure_limit=5, spray_limit=5, success_failure_limit=3):
    """Return heuristic alert reasons, not confirmed attack attribution."""
    names = []
    for r in features.itertuples(index=False):
        fired = []
        if r.account_failures_60s >= failure_limit:
            fired.append('repeated_failures')
        if r.ip_distinct_users_10m >= spray_limit:
            fired.append('password_spraying')
        if r.success and r.prior_consecutive_failures >= success_failure_limit:
            fired.append('success_after_failures')
        names.append(';'.join(fired))
    return pd.Series(names, index=features.index, dtype='str')
