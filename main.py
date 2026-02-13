import logging
import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Project Imports
from src.utils.config_loader import *
from src.utils.serialiser import save_json_records  
from src.extraction.client import fetch_gho_data
from src.extraction.data_processor import *
from src.graph.context_builder import fetch_biomedical_context, create_fused_factory, get_drug_name_mapping
from src.graph.graph_engine import train_knowledge_graph_model
from src.visualisation.dashboard_engine import HerculeDashboard
from src.storage.neo4j_manager import Neo4jManager 

# 1. Environment & Project Setup
load_dotenv()
current_date = datetime.now().strftime("%Y%m%d")

# 1. Load Environment Variables
load_dotenv()

# 2. Setup Logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"{current_date}_surveillance.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(log_filename, mode="a")]
)
logger = logging.getLogger(__name__)

def load_cached_triples(subfolder):
    """Checks for existing triples in local storage to skip API calls."""
    cache_path = Path("data/triples") / subfolder
    all_triples = []
    if cache_path.exists():
        triple_files = list(cache_path.glob("*_triples.json"))
        if triple_files:
            logger.info(f"Found {len(triple_files)} cached triple files for today. Skipping WHO API.")
            for file in triple_files:
                with open(file, 'r') as f:
                    all_triples.extend(json.load(f))
            return all_triples
    return None

def run_surveillance_pipeline():
    """Extracts surveillance data, utilizing cache if available."""
    cached_data = load_cached_triples(current_date)
    if cached_data:
        return cached_data

    try:
        disease_configs = load_disease_config(filename="disease_mapping.json")
    except Exception as e:
        logger.error(f"Failed to load disease configuration: {e}")
        return []
    
    all_triples = []
    for disease in disease_configs:
        name = disease.get('biological_name', 'Unknown')
        code = disease.get('who_code', 'N/A')
        try:
            raw_data = fetch_gho_data(code)
            clean_data = parse_data(raw_data)
            latest_records = clean_latest_records(clean_data, min_year=2015)
            
            save_json_records(data=latest_records, name=name, subfolder=current_date)

            triples = build_triples(
                df=latest_records,
                hetionet_id=disease.get('hetionet_id', 'N/A'),
                threshold=disease.get('outbreak_threshold', 1000)
            )
            all_triples.extend(triples)
            save_json_records(data=triples, name=f"{name}_triples", 
                             subfolder=current_date, base_dir="data/triples")
        except Exception as e:
            logger.warning(f"Failed to fetch data for {name}: {e}")
            continue
    return all_triples

if __name__ == "__main__":
    logging.info("Starting HERCULE surveillance pipeline execution.")
    
    # 1. Data Extraction (Cache-aware)
    who_triples = run_surveillance_pipeline()
    
    if not who_triples:
        logging.error("No surveillance triples available. Aborting.")
    else:
        # 2. Context Building
        active_diseases = list(set([t[2] for t in who_triples]))
        bio_triples = fetch_biomedical_context(active_diseases)
        drug_names = get_drug_name_mapping(bio_triples)

        # 3. Model Training
        tf = create_fused_factory(who_triples, bio_triples)
        training_results = train_knowledge_graph_model(
            triples_factory=tf, epochs=200, embedding_dim=100
        )

        # # 4. Neo4j Export (Factual + Predicted)
        # # Using the new Manager logic to incorporate the Model insights
        # if bio_triples:
        #     logger.info("Syncing data and insights to Neo4j Aura...")
        #     db_manager = Neo4jManager()
        #     try:
        #         db_manager.upload_triples(who_triples + bio_triples)
                
        #         logger.info("Generating model predictions via predict_all_triples...")
                
        #         # This function computes scores for all possible triples 
        #         # (Can be heavy, but we filter by relation)
        #         prediction_df = predict_all_triples(
        #             model=training_results.model,
        #             triples_factory=tf,
        #         ).process(factory=tf).df

        #         # Filter specifically for the 'CtD' relation
        #         # and take the top 100 results
        #         top_predictions = prediction_df[
        #             prediction_df['relation_label'] == 'CtD'
        #         ].sort_values('score', ascending=False).head(100)
                
        #         db_manager.upload_predictions(top_predictions)
        #         logger.info("Neo4j Aura sync complete.")
                
        #     except Exception as e:
        #         logger.error(f"Neo4j sync failed: {e}")
        #         import traceback
        #         logger.error(traceback.format_exc())
        #     finally:
        #         db_manager.close()

        # 5. Dashboard Generation (Final Visualization)
        disease_configs = load_disease_config(filename="disease_mapping.json")
        disease_map = {d['who_code']: d['hetionet_id'] for d in disease_configs}
        readable_names = {d['who_code']: d['biological_name'] for d in disease_configs}

        try:
            dashboard = HerculeDashboard(disease_map, readable_names, drug_names)
            dashboard.build_surveillance_graph(who_triples, who_triples + bio_triples)
            dashboard.render(output_path=f"data/reports/{current_date}_dashboard2.png")
            logging.info("Visual dashboard generated.")
        except Exception as e:
            logger.error(f"Dashboard failed: {e}")

    logging.info("HERCULE pipeline execution complete.")