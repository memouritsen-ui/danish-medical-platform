import os
import nest_asyncio
nest_asyncio.apply() # Apply globally to fix nested loop issues in Docker

import chromadb
from chromadb.config import Settings
import networkx as nx
import pickle
from neo4j import GraphDatabase
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

# Chroma Setup
CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")

try:
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=int(CHROMA_PORT))
    collection = chroma_client.get_or_create_collection(name="medical_evidence")
except Exception as e:
    logger.warning(f"Could not connect to ChromaDB: {e}. Using local fallback if possible or failing.")
    # Fallback for local dev without docker immediately
    chroma_client = chromadb.PersistentClient(path="./chroma_data")
    collection = chroma_client.get_or_create_collection(name="medical_evidence")

# Graph Setup
class GraphManager:
    def __init__(self):
        self.nx_graph = nx.DiGraph()
        self.neo4j_driver = None
        self.graph_file = "graph_data.pkl"
        
        # Load NetworkX
        if os.path.exists(self.graph_file):
            try:
                with open(self.graph_file, 'rb') as f:
                    self.nx_graph = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load graph pickle: {e}")

        # Connect Neo4j
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_pass = os.getenv("NEO4J_PASSWORD", "password")
        
        try:
            self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
            self.neo4j_driver.verify_connectivity()
            logger.info("Connected to Neo4j")
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j: {e}. Using NetworkX only.")

    def add_claim(self, source: str, claim: str, evidence_strength: float, verification: str):
        # NetworkX
        self.nx_graph.add_node(source, type="source")
        self.nx_graph.add_node(claim, type="claim", verification=verification)
        self.nx_graph.add_edge(source, claim, weight=evidence_strength, relation="asserts")
        self._save_nx()

        # Neo4j
        if self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    session.run(
                        """
                        MERGE (s:Source {url: $source})
                        MERGE (c:Claim {text: $claim})
                        SET c.verification = $verification
                        MERGE (s)-[r:ASSERTS {weight: $weight}]->(c)
                        """,
                        source=source, claim=claim, verification=verification, weight=evidence_strength
                    )
            except Exception as e:
                logger.error(f"Neo4j write failed: {e}")

    def add_contradiction(self, claim1: str, claim2: str):
        self.nx_graph.add_edge(claim1, claim2, relation="contradicts")
        self._save_nx()
        
        if self.neo4j_driver:
             with self.neo4j_driver.session() as session:
                session.run(
                    """
                    MATCH (c1:Claim {text: $c1}), (c2:Claim {text: $c2})
                    MERGE (c1)-[:CONTRADICTS]-(c2)
                    """,
                    c1=claim1, c2=claim2
                )

    def get_graph_data(self):
        return nx.node_link_data(self.nx_graph)

    def _save_nx(self):
        with open(self.graph_file, 'wb') as f:
            pickle.dump(self.nx_graph, f)

    def close(self):
        if self.neo4j_driver:
            self.neo4j_driver.close()

graph_db = GraphManager()
