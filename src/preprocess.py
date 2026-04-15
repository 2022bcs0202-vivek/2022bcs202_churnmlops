import pandas as pd

def feature_engineering(df):
    # Example derived features based on available dataset
    
    # Total calls (proxy for ticket frequency)
    df['total_calls'] = (
        df['Total day calls'] +
        df['Total eve calls'] +
        df['Total night calls'] +
        df['Total intl calls']
    )

    # Total minutes
    df['total_minutes'] = (
        df['Total day minutes'] +
        df['Total eve minutes'] +
        df['Total night minutes'] +
        df['Total intl minutes']
    )

    # Charge change proxy (day vs evening)
    df['charge_diff'] = df['Total day charge'] - df['Total eve charge']

    # Avg minutes per call
    df['avg_minutes_per_call'] = df['total_minutes'] / (df['total_calls'] + 1)

    return df
