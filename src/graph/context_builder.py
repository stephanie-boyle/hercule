import logging
import torch
import numpy as np
from pykeen.datasets import Hetionet
from pykeen.triples import TriplesFactory

logger = logging.getLogger(__name__)

def fetch_biomedical_context(active_diseases, relations=['CtD', 'CpD']):
    """
    Extracts therapeutic links from Hetionet for specified diseases.
    CtD: Compound treats Disease | CpD: Compound palliates Disease
    """
    logger.info("Initialising Hetionet dataset fusion")
    try:
        dataset = Hetionet()
        hetionet_tf = dataset.training
        
        # map disease labels to internal IDs
        target_ids = [
            hetionet_tf.entity_to_id[d] 
            for d in active_diseases 
            if d in hetionet_tf.entity_to_id
        ]
        
        if not target_ids:
            logger.warning("No matching disease IDs found in Hetionet for provided list")
            return []

        mapped_triples = hetionet_tf.mapped_triples
        target_tensor = torch.tensor(target_ids)

        # mask for triples where the disease is either the head or tail
        mask = torch.isin(mapped_triples[:, 0], target_tensor) | \
               torch.isin(mapped_triples[:, 2], target_tensor)

        relevant_df = hetionet_tf.tensor_to_df(mapped_triples[mask])
        
        # filter for specific therapeutic relations
        filtered_df = relevant_df[relevant_df['relation_label'].isin(relations)]
        
        results = filtered_df[['head_label', 'relation_label', 'tail_label']].values.tolist()
        logger.info(f"Identified {len(results)} therapeutic triples in Hetionet")
        return results

    except Exception as e:
        logger.error(f"Failed to process biomedical context: {e}")
        raise

def create_fused_factory(surveillance_triples, bio_triples):
    """Combines WHO data with Hetionet knowledge into a TriplesFactory."""

    logger.info("Fusing surveillance and biomedical triples into TriplesFactory")

    all_triples = surveillance_triples + bio_triples
    
    return TriplesFactory.from_labeled_triples(
        np.array(all_triples), 
        create_inverse_triples=True
    )