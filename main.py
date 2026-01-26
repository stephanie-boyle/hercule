from pathlib import Path
from src.utils.config_loader import *
from src.utils.serialiser import save_json_records  
from src.extraction.client import fetch_gho_data
from src.extraction.data_processor import *
from src.graph.context_builder import fetch_biomedical_context, create_fused_factory
from src.graph.graph_engine import train_knowledge_graph_model

import logging
import os
from datetime import datetime

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

current_date = datetime.now().strftime("%Y%m%d")
log_filename = os.path.join(log_dir, f"{current_date}_surveillance.log")

# 3. Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, mode="a") # mode a = append
    ]
)

logger = logging.getLogger(__name__)

def run_surveillance_pipeline():
    """
    Main pipeline to extract disease surveillance data from WHO GHO.
    Returns:
        list: Extracted surveillance triples.
    """

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
            # getting the data from WHO GHO
            raw_data = fetch_gho_data(code)
            logger.info(f"Fetched {len(raw_data)} records from WHO GHO.")

            # processing the raw data
            clean_data = parse_data(raw_data)

            latest_records = clean_latest_records(clean_data, min_year=2015)
            logger.info(f"Filtered to {len(latest_records)} latest records since 2015.")

            # save this data before creating triples
            save_json_records(
                data=latest_records,
                name=name,
                subfolder=current_date
            )

            triples = build_triples(
                df=latest_records,
                hetionet_id=disease.get('hetionet_id', 'N/A'),
                threshold=disease.get('outbreak_threshold', 1000)
            )
            logger.info(f"Built {len(triples)} surveillance triples for {name}.")
            all_triples.extend(triples)

            save_json_records(
                data=triples,
                name=f"{name}_triples",
                subfolder=current_date,
                base_dir="data/triples"
            )
            logger.info(f"Saved triples for {name}.")

        except Exception as e:
            logger.warning(f"Failed to fetch data for {name}: {e}")
            continue

    return all_triples



if __name__ == "__main__":
    logging.info("Starting HERCULE surveillance pipeline execution.")
    who_triples = run_surveillance_pipeline()
    if not who_triples:
        logging.error("No surveillance triples were generated. Aborting pipeline.")
    else:
        # 2. Preparation Phase: Identify unique diseases to find treatments for
        # Extract the Hetionet IDs (tail of the triple) from our WHO findings
        active_diseases = list(set([t[2] for t in who_triples]))
        logging.info(f"Identified {len(active_diseases)} unique diseases from WHO data for context building.")
        
        # 3. Fusion Phase: Use the Knowledge Integrator
        # Pulls CtD/CpD relations from Hetionet for these specific diseases
        bio_triples = fetch_biomedical_context(active_diseases)
        logging.info(f"Retrieved {len(bio_triples)} biomedical triples from Hetionet.")

        # Merge WHO and Hetionet into a PyKEEN TriplesFactory
        tf = create_fused_factory(who_triples, bio_triples)
        logging.info(f"Fused triples factory contains {tf.num_triples} triples total.")

        # 4. Analysis Phase: Use the Model Engine
        # Train the RotatE model on the fused graph
        training_results = train_knowledge_graph_model(
            triples_factory=tf,
            epochs=200,
            embedding_dim=100
        )

        logging.info("HERCULE pipeline execution complete.")