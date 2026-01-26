import pandas as pd

def extract_latest_records(raw_data, min_year=2015):
    """
    Parses raw API JSON into a cleaned DataFrame of latest country stats.
    Args:
        raw_data (list): Raw data from WHO GHO API.
        min_year (int): Minimum year to consider for latest records.
    Returns:
        pd.DataFrame: DataFrame with latest records per country.
    """
    records = [
        {'Country': r.get('SpatialDim'), 'Year': r.get('TimeDim'), 'Cases': r.get('NumericValue')}
        for r in raw_data if r.get('TimeDim') and r.get('TimeDim') >= min_year
    ]
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = (df.sort_values(['Country', 'Year'], ascending=[True, False])
            .drop_duplicates(subset=['Country'], keep='first'))
    
    return df

def build_triples(df, hetionet_id, threshold):
    """
    Builds surveillance triples for countries exceeding the outbreak threshold.
    Args:
        df (pd.DataFrame): DataFrame with latest country records.
        hetionet_id (str): Hetionet ID for the disease.
        threshold (int): Case count threshold to flag an outbreak.
    Returns:
        list: List of surveillance triples.
    """
    if df.empty:
        return []
    
    active_mask = (df['Cases'] > threshold)
    return [
        (f"Country::{row['Country']}", 'has_active_outbreak', hetionet_id)
        for _, row in df[active_mask].iterrows()
    ]