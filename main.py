from config_loader import load_disease_configurations
from who_client import fetch_gho_data
import data_processor as dp
import glob

def run_surveillance_pipeline():
    """
    Main pipeline to extract disease surveillance data from WHO GHO.
    Returns:
        list: Extracted surveillance triples.
    """
    json_files = glob.glob("dictionary/*.json")
    disease_configs = load_disease_configurations(json_files)
    
    surveillance_triples = []
    print(f"Scanning WHO GHO for {len(disease_configs)} diseases...")

    for disease in disease_configs:
        raw_data = fetch_gho_data(disease['code'])
        
        df_latest = dp.extract_latest_records(raw_data)
        
        new_triples = dp.build_triples(df_latest, disease['id'], disease['threshold'])
        
        if new_triples:
            surveillance_triples.extend(new_triples)
            print(f" {disease['name']} ({disease['id']}): Found {len(new_triples)} hotspots.")

    print(f"\nTotal Surveillance Links Extracted: {len(surveillance_triples)}")
    return surveillance_triples

if __name__ == "__main__":
    who_triples = run_surveillance_pipeline()