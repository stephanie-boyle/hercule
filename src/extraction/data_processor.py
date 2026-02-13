import pandas as pd

def parse_data(raw_data):
    """
    Extracts relevant fields from raw JSON and handles initial type casting.
    """
    column_map = {
        'SpatialDim': 'Country',
        'TimeDim': 'Year',
        'NumericValue': 'Cases'
    }
    
    try:
        if not isinstance(raw_data, list):
            raise ValueError("Input data must be a list of records.")

        df = pd.DataFrame(raw_data)
        
        # verify columns exist
        missing = [k for k in column_map.keys() if k not in df.columns]
        if missing:
            raise KeyError(f"Missing expected keys from API: {missing}")

        df = df[list(column_map.keys())].rename(columns=column_map)
        
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Cases'] = pd.to_numeric(df['Cases'], errors='coerce')
        
        return df

    except Exception as e:
        print(f"Parsing Error: {e}")
        return pd.DataFrame()
    

def clean_latest_records(df, min_year=2015):
    """
    Filters data by year and reduces to the most recent record per country.
    """
    try:
        if df.empty:
            return df

        processed_df = (
            df.query("Year >= @min_year")
              .dropna(subset=['Country', 'Year'])
        )

        idx = processed_df.groupby('Country')['Year'].idxmax()
        
        return processed_df.loc[idx].sort_values('Country').reset_index(drop=True)

    except Exception as e:
        print(f"Cleaning Error: {e}")
        return pd.DataFrame()


def build_triples(df, hetionet_id, threshold):
    """
    Builds surveillance triples for countries exceeding the threshold,
    while filtering out aggregate continent/region codes.
    """
    if df.empty:
        return []
    
    # List of aggregate codes to exclude from subject nodes
    continent_codes = {'GLOBAL', 'AFR', 'AMR', 'SEAR', 'EUR', 'EMR', 'WPR'}
    
    # Apply threshold and exclude aggregate regions
    # We use .copy() to avoid SettingWithCopy warnings if df is a slice
    active_mask = (df['Cases'] > threshold) & (~df['Country'].isin(continent_codes))
    
    return [
        (f"Country::{row['Country']}", 'has_active_outbreak', hetionet_id)
        for _, row in df[active_mask].iterrows()
    ]

