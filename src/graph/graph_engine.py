import logging
import torch
from pykeen.pipeline import pipeline

logger = logging.getLogger(__name__)

def train_knowledge_graph_model(triples_factory, epochs=200, embedding_dim=100):
    """
    Trains a RotatE model using the provided TriplesFactory.
    Optimises for GPU if available.
    Args:
        triples_factory (TriplesFactory): The triples factory containing training data.
        epochs (int): Number of training epochs.
        embedding_dim (int): Dimension of the embeddings.
    Returns:
        Result: The result of the training pipeline.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Starting model training on {device} for {epochs} epochs")
    
    try:
        result = pipeline(
            training=triples_factory,
            testing=triples_factory,
            model='RotatE',
            model_kwargs={'embedding_dim': embedding_dim},
            epochs=epochs,
            random_seed=42,
            device=device
        )
        logger.info("Model training completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during model training pipeline: {e}")
        raise