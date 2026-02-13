import os
import logging
import requests
import json
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class Neo4jManager:
    def __init__(self):
        """Initializes the Neo4j Manager using HTTPS Query API for Aura compatibility."""
        load_dotenv()
        
        raw_uri = os.getenv("NEO4J_URI")
        self.host = raw_uri.split("//")[-1] if raw_uri else None
        self.user = os.getenv("NEO4J_USER")
        self.password = os.getenv("NEO4J_PASSWORD")
        
        self.url = f"https://{self.host}/db/neo4j/query/v2"

        if not all([self.host, self.user, self.password]):
            logger.error("HTTPS Neo4j credentials missing in .env.")
            raise ValueError("Check NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD.")

    def _run_query(self, cypher, parameters=None):
        """Sends a Cypher statement to the Aura HTTPS Query API."""
        payload = {
            "statement": cypher,
            "parameters": parameters or {}
        }
        
        response = requests.post(
            self.url,
            auth=(self.user, self.password),
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        if response.status_code != 200:
            logger.error(f"Cypher Query Failed: {response.text}")
            response.raise_for_status()
            
        return response.json()

    def upload_triples(self, triples, clear_first=True):
        """
        Uploads the factual knowledge (WHO + Hetionet) to Neo4j.
        Labels nodes based on their ID prefixes (e.g., 'Country::', 'Compound::').
        """
        if clear_first:
            logger.info("Clearing existing graph...")
            self._run_query("MATCH (n) DETACH DELETE n")

        batch_data = []
        for s, p, o in triples:
            batch_data.append({
                "s_id": s, "s_label": s.split("::")[0] if "::" in s else "Entity",
                "rel": p.upper().replace(" ", "_"),
                "o_id": o, "o_label": o.split("::")[0] if "::" in o else "Entity"
            })

        cypher = """
        UNWIND $batches AS item
        MERGE (s:Resource {id: item.s_id})
        WITH s, item
        CALL apoc.create.addLabels(s, [item.s_label]) YIELD node AS s_node
        
        MERGE (o:Resource {id: item.o_id})
        WITH s_node, o, item
        CALL apoc.create.addLabels(o, [item.o_label]) YIELD node AS o_node
        
        CALL apoc.merge.relationship(s_node, item.rel, {}, {}, o_node, {}) YIELD rel
        RETURN count(*)
        """
        self._run_query(cypher, {"batches": batch_data})
        logger.info(f"Factual triples ({len(triples)}) uploaded.")

    def upload_predictions(self, predictions_df, rel_type="PREDICTED_TREATMENT"):
        """
        Uploads scores from your PyKEEN model.
        Expects a DataFrame with columns: ['head_label', 'tail_label', 'score']
        """
        logger.info(f"Integrating {len(predictions_df)} model predictions into Neo4j...")
        
        batch_data = []
        for _, row in predictions_df.iterrows():
            batch_data.append({
                "s_id": str(row['head_label']),
                "o_id": str(row['tail_label']),
                "score": round(float(row['score']), 4)
            })

        # We use MATCH here instead of MERGE because nodes should already exist 
        # from the factual triples upload.
        cypher = f"""
        UNWIND $batches AS item
        MATCH (s:Resource {{id: item.s_id}})
        MATCH (o:Resource {{id: item.o_id}})
        CALL apoc.merge.relationship(s, "{rel_type}", {{confidence: item.score}}, {{}}, o, {{}}) YIELD rel
        RETURN count(*)
        """
        
        self._run_query(cypher, {"batches": batch_data})
        logger.info("Model predictions integrated successfully.")

    def close(self):
        """Stateless HTTPS doesn't require a close, but kept for interface consistency."""
        pass