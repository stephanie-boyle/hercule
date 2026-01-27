import os
import logging
import requests
import json
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class Neo4jManager:
    def __init__(self):
        """
        Initializes the Neo4j Manager using the HTTPS Query API.
        Hostname should be extracted from the URI in the .env file.
        """
        load_dotenv()
        
        # Extract host from URI: neo4j+s://xxxxxxxx.databases.neo4j.io -> xxxxxxxx.databases.neo4j.io
        raw_uri = os.getenv("NEO4J_URI")
        self.host = raw_uri.split("//")[-1] if raw_uri else None
        self.user = os.getenv("NEO4J_USER")
        self.password = os.getenv("NEO4J_PASSWORD")
        
        # Standard Aura HTTPS Query API Endpoint
        self.url = f"https://{self.host}/db/neo4j/query/v2"

        if not all([self.host, self.user, self.password]):
            logger.error("HTTPS Neo4j credentials missing in .env.")
            raise ValueError("Ensure NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD are set.")

    def _run_query(self, cypher, parameters=None):
        """Internal helper to send POST requests to the Query API."""
        payload = {
            "statement": cypher,
            "parameters": parameters or {}
        }
        
        response = requests.post(
            self.url,
            auth=(self.user, self.password),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            data=json.dumps(payload)
        )
        
        if response.status_code != 200:
            logger.error(f"Cypher Query Failed: {response.status_code} - {response.text}")
            response.raise_for_status()
            
        return response.json()

    def upload_triples(self, triples):
        """
        Clears the database and uploads triples via HTTPS.
        """
        logger.info("Cleaning existing graph data via HTTPS...")
        self._run_query("MATCH (n) DETACH DELETE n")

        batch_data = []
        for s, p, o in triples:
            s_label = s.split("::")[0] if "::" in s else "Entity"
            o_label = o.split("::")[0] if "::" in o else "Entity"
            
            batch_data.append({
                "s_id": s, "s_label": s_label,
                "rel": p.upper().replace(" ", "_"),
                "o_id": o, "o_label": o_label
            })

        # Since Query API v2 handles one statement per request, 
        # we use UNWIND for high-performance batching.
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

        logger.info(f"Uploading {len(triples)} triples to Neo4j Aura via HTTPS...")
        self._run_query(cypher, {"batches": batch_data})
        logger.info("HTTPS Graph creation complete.")

    def close(self):
        # Requests is stateless, so no persistent connection to close
        pass