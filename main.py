from pathlib import Path
from src.utils.config_loader import *
from src.utils.serialiser import save_json_records  
from src.extraction.client import fetch_gho_data
from src.extraction.data_processor import *
from src.graph.context_builder import fetch_biomedical_context, create_fused_factory, get_drug_name_mapping
from src.graph.graph_engine import train_knowledge_graph_model
from src.visualisation.dashboard_engine import HerculeDashboard
from src.storage.neo4j_manager import Neo4jManager 

import logging
import os
from datetime import datetime
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

# 2. Setup Logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
current_date = datetime.now().strftime("%Y%m%d")
log_filename = os.path.join(log_dir, f"{current_date}_surveillance.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, mode="a")
    ]
)
logger = logging.getLogger(__name__)

def run_surveillance_pipeline():
    """Extracts disease surveillance data from WHO GHO."""
    try:
        disease_configs = load_disease_config(filename="disease_mapping.json")
    except Exception as e:
        logger.error(f"Failed to load disease configuration: {e}")
        return []
    
    logger.info(f"Scanning WHO GHO for {len(disease_configs)} diseases...")
    all_triples = []

    for disease in disease_configs:
        name = disease.get('biological_name', 'Unknown')
        code = disease.get('who_code', 'N/A')
        logger.info(f"Processing disease: {name} (WHO Code: {code})")
        
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

            save_json_records(
                data=triples, name=f"{name}_triples", 
                subfolder=current_date, base_dir="data/triples"
            )
        except Exception as e:
            logger.warning(f"Failed to fetch data for {name}: {e}")
            continue

    return all_triples

if __name__ == "__main__":
    logging.info("Starting HERCULE surveillance pipeline execution.")
    
    # 1. Extract Surveillance Data
    who_triples = run_surveillance_pipeline()
    
    if not who_triples:
        logging.error("No surveillance triples were generated. Aborting pipeline.")
    else:
        # 2. Build Biomedical Context
        active_diseases = list(set([t[2] for t in who_triples]))
        bio_triples = fetch_biomedical_context(active_diseases)
        drug_names = get_drug_name_mapping(bio_triples)

        # 3. Train Knowledge Graph Model
        tf = create_fused_factory(who_triples, bio_triples)
        training_results = train_knowledge_graph_model(
            triples_factory=tf,
            epochs=200,
            embedding_dim=100
        )

        # 4. Generate Visual Dashboard
        logging.info("Generating HERCULE Dashboard...")
        disease_configs = load_disease_config(filename="disease_mapping.json")
        disease_map = {d['who_code']: d['hetionet_id'] for d in disease_configs}
        readable_names = {d['who_code']: d['biological_name'] for d in disease_configs}

        try:
            dashboard = HerculeDashboard(
                disease_map=disease_map,
                name_resolver=readable_names,
                drug_map=drug_names
            )
            dashboard.build_surveillance_graph(
                surveillance_triples=who_triples, 
                all_triples=who_triples + bio_triples,
                top_n=3
            )
            report_path = f"data/reports/{current_date}_hercule_dashboard.png"
            dashboard.render(output_path=report_path)
            logging.info(f"Dashboard saved: {report_path}")
        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")

        # 5. Export to Neo4j Aura
        if bio_triples:
            logging.info("Uploading data to Neo4j Aura...")
            db_manager = Neo4jManager()
            try:
                db_manager.upload_triples(who_triples + bio_triples)
                logging.info("Neo4j upload successful.")
            except Exception as e:
                logger.error(f"Neo4j upload failed: {e}")
            finally:
                db_manager.close()

    logging.info("HERCULE pipeline execution complete.")