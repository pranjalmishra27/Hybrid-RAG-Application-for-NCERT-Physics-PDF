#!/usr/bin/env python3
"""
Neo4j Graph Setup Script
Verifies connection, creates constraints, and optionally runs a test query.

Usage:
    python scripts/setup_neo4j.py               # verify connection
    python scripts/setup_neo4j.py --test-query  # run sample Cypher
    python scripts/setup_neo4j.py --clear       # clear all data
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from loguru import logger

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "physics_rag_2024")


def verify_connection():
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        logger.success(f"Neo4j connected at {NEO4J_URI}")
        return driver
    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}")
        logger.info("Start Neo4j with:")
        logger.info("  docker run -d --name neo4j \\")
        logger.info("    -p 7474:7474 -p 7687:7687 \\")
        logger.info("    -e NEO4J_AUTH=neo4j/physics_rag_2024 \\")
        logger.info("    neo4j:5.18-community")
        return None


def print_graph_stats(driver):
    with driver.session() as session:
        result = session.run("""
            MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count
            ORDER BY count DESC
        """)
        rows = list(result)
        if not rows:
            logger.info("Graph is empty — run ingestion first")
            return
        logger.info("Graph node counts:")
        for row in rows:
            logger.info(f"  {row['label']:15s}: {row['count']}")

        rel_result = session.run("""
            MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS count
            ORDER BY count DESC
        """)
        logger.info("Relationship counts:")
        for row in rel_result:
            logger.info(f"  {row['rel']:20s}: {row['count']}")


def run_test_queries(driver):
    test_queries = [
        ("Find laws in Electric Charges chapter",
         "MATCH (l:Law)-[:BELONGS_TO]->(c:Chapter {name: 'Electric Charges and Fields'}) RETURN l.name LIMIT 5"),
        ("Find formulas for Coulomb",
         "MATCH (f:Formula)-[:BELONGS_TO]->(c:Chapter) WHERE c.name CONTAINS 'Electric' RETURN f.expression LIMIT 5"),
        ("Multi-hop: Topic → Chapter → Chunk",
         "MATCH (t:Topic)-[:BELONGS_TO]->(c:Chapter)<-[:BELONGS_TO]-(chunk:Chunk) WHERE t.name CONTAINS 'Electric Field' RETURN t.name, c.name, chunk.page LIMIT 3"),
        ("Find related scientists",
         "MATCH (s:Scientist)-[:CONTRIBUTED_TO]->(c:Chapter) RETURN s.name, c.name LIMIT 5"),
    ]

    logger.info("\nRunning test queries:")
    with driver.session() as session:
        for name, cypher in test_queries:
            try:
                result = list(session.run(cypher))
                logger.info(f"\n  [{name}]")
                for row in result:
                    logger.info(f"    {dict(row)}")
            except Exception as e:
                logger.warning(f"  Query failed: {e}")


def create_constraints(driver):
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chapter) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Scientist) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Law) REQUIRE l.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Definition) REQUIRE d.term IS UNIQUE",
    ]
    with driver.session() as session:
        for c in constraints:
            try:
                session.run(c)
            except Exception:
                pass
    logger.info("Constraints created")


def clear_graph(driver):
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    logger.info("Graph cleared")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-query", action="store_true")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--stats", action="store_true", default=True)
    args = parser.parse_args()

    driver = verify_connection()
    if driver:
        if args.clear:
            clear_graph(driver)
        create_constraints(driver)
        if args.stats:
            print_graph_stats(driver)
        if args.test_query:
            run_test_queries(driver)
        driver.close()
