"""
Neo4j Knowledge Graph Module
Builds and queries a physics knowledge graph from NCERT content.

Nodes: Chapter, Topic, Formula, Definition, Concept, Law, Scientist
Relationships: BELONGS_TO, DERIVED_FROM, RELATED_TO, USES, EXPLAINS
"""

import re
from typing import List, Dict, Optional, Tuple
from loguru import logger

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

from app.ingestion.pdf_parser import PhysicsChunk


# Physics knowledge seed: laws, scientists, key concepts per chapter
PHYSICS_KNOWLEDGE_SEED = {
    "Electric Charges and Fields": {
        "laws": ["Coulomb's Law", "Gauss's Law", "Superposition Principle"],
        "concepts": ["Electric Charge", "Electric Field", "Electric Field Lines",
                     "Electric Flux", "Point Charge", "Continuous Charge Distribution"],
        "formulas": ["F = kq1q2/r²", "E = F/q", "Φ = E·A", "E = σ/ε₀"],
        "scientists": ["Coulomb", "Gauss", "Faraday"],
        "definitions": ["Electric charge", "Field intensity", "Permittivity"],
    },
    "Electrostatic Potential and Capacitance": {
        "laws": ["Work-Energy Theorem", "Gauss's Law"],
        "concepts": ["Electric Potential", "Equipotential Surface", "Capacitance",
                     "Dielectric", "Energy Stored in Capacitor", "Capacitors in Series/Parallel"],
        "formulas": ["V = kq/r", "C = Q/V", "U = Q²/2C", "C = ε₀A/d"],
        "scientists": ["Faraday", "Van de Graaff"],
        "definitions": ["Potential", "Capacitance", "Dielectric constant"],
    },
    "Current Electricity": {
        "laws": ["Ohm's Law", "Kirchhoff's Voltage Law", "Kirchhoff's Current Law",
                 "Joule's Law of Heating"],
        "concepts": ["Electric Current", "Resistance", "Resistivity", "EMF",
                     "Drift Velocity", "Mobility", "Wheatstone Bridge"],
        "formulas": ["V = IR", "R = ρL/A", "P = VI", "I = nAevd"],
        "scientists": ["Ohm", "Kirchhoff", "Joule", "Wheatstone"],
        "definitions": ["Current", "Resistance", "Conductivity"],
    },
    "Moving Charges and Magnetism": {
        "laws": ["Biot-Savart Law", "Ampere's Circuital Law", "Lorentz Force Law"],
        "concepts": ["Magnetic Field", "Magnetic Force", "Cyclotron", "Solenoid",
                     "Toroid", "Moving Coil Galvanometer"],
        "formulas": ["F = qv×B", "dB = μ₀Idl×r/4πr³", "B = μ₀nI"],
        "scientists": ["Biot", "Savart", "Ampere", "Lorentz"],
        "definitions": ["Magnetic field", "Magnetic flux density"],
    },
    "Magnetism and Matter": {
        "laws": ["Gauss's Law for Magnetism", "Curie's Law"],
        "concepts": ["Diamagnetism", "Paramagnetism", "Ferromagnetism",
                     "Magnetic Susceptibility", "Hysteresis", "Curie Temperature"],
        "formulas": ["M = χH", "μr = 1 + χ"],
        "scientists": ["Curie", "Weiss"],
        "definitions": ["Magnetization", "Susceptibility", "Permeability"],
    },
    "Electromagnetic Induction": {
        "laws": ["Faraday's Law of Induction", "Lenz's Law"],
        "concepts": ["Magnetic Flux", "EMF Induction", "Self Inductance",
                     "Mutual Inductance", "Eddy Currents", "AC Generator"],
        "formulas": ["ε = -dΦ/dt", "Φ = NBA", "M = μ₀N₁N₂A/l"],
        "scientists": ["Faraday", "Lenz", "Henry"],
        "definitions": ["Inductance", "Mutual inductance", "Self inductance"],
    },
    "Alternating Current": {
        "laws": ["Ohm's Law for AC"],
        "concepts": ["RMS Value", "Impedance", "Resonance", "Power Factor",
                     "LC Oscillations", "Transformer", "Q-factor"],
        "formulas": ["Irms = I₀/√2", "Z = √(R²+(XL-XC)²)", "P = Vrms·Irms·cosφ"],
        "scientists": ["Tesla", "Edison"],
        "definitions": ["Impedance", "Reactance", "Power factor"],
    },
    "Electromagnetic Waves": {
        "laws": ["Maxwell's Equations"],
        "concepts": ["Displacement Current", "EM Spectrum", "Speed of Light",
                     "Radio Waves", "Microwaves", "X-rays", "Gamma Rays"],
        "formulas": ["c = 1/√(μ₀ε₀)", "c = 3×10⁸ m/s"],
        "scientists": ["Maxwell", "Hertz"],
        "definitions": ["Displacement current", "Electromagnetic wave"],
    },
}


class GraphRetriever:
    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None
        self.available = False

        if not NEO4J_AVAILABLE:
            logger.warning("neo4j package not installed. Graph retrieval disabled.")
            return

        try:
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            self._driver.verify_connectivity()
            self.available = True
            logger.success("Neo4j connection established")
        except Exception as e:
            logger.warning(f"Neo4j not available: {e}. Graph retrieval will be skipped.")

    def close(self):
        if self._driver:
            self._driver.close()

    def build_graph(self, chunks: List[PhysicsChunk]):
        """Build knowledge graph from physics chunks + seed knowledge."""
        if not self.available:
            return

        logger.info("Building Neo4j knowledge graph...")

        with self._driver.session() as session:
            # Clear existing graph
            session.run("MATCH (n) DETACH DELETE n")

            # Create constraints
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chapter) REQUIRE c.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (f:Formula) REQUIRE f.expression IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Scientist) REQUIRE s.name IS UNIQUE")

            # Create chapter nodes
            chapters_seen = set()
            for chunk in chunks:
                if chunk.chapter and chunk.chapter not in chapters_seen:
                    session.run(
                        "MERGE (c:Chapter {name: $name, num: $num})",
                        name=chunk.chapter, num=chunk.chapter_num
                    )
                    chapters_seen.add(chunk.chapter)

            # Create chunk nodes and link to chapters
            for chunk in chunks:
                session.run("""
                    MERGE (ch:Chunk {id: $chunk_id})
                    SET ch.content = $content,
                        ch.page = $page,
                        ch.subheading = $subheading,
                        ch.chunk_type = $chunk_type
                    WITH ch
                    MATCH (c:Chapter {name: $chapter})
                    MERGE (ch)-[:BELONGS_TO]->(c)
                """,
                    chunk_id=chunk.chunk_id,
                    content=chunk.content[:500],  # truncate for graph
                    page=chunk.page,
                    subheading=chunk.subheading,
                    chunk_type=chunk.chunk_type,
                    chapter=chunk.chapter,
                )

                # Create formula nodes
                for formula in chunk.formulas[:3]:  # limit per chunk
                    if len(formula) > 5:
                        session.run("""
                            MERGE (f:Formula {expression: $expr})
                            WITH f
                            MATCH (c:Chapter {name: $chapter})
                            MERGE (f)-[:BELONGS_TO]->(c)
                        """, expr=formula[:200], chapter=chunk.chapter)

            # Inject seed knowledge
            for chapter_name, knowledge in PHYSICS_KNOWLEDGE_SEED.items():
                # Laws
                for law in knowledge.get("laws", []):
                    session.run("""
                        MERGE (l:Law {name: $name})
                        WITH l
                        MATCH (c:Chapter {name: $chapter})
                        MERGE (l)-[:BELONGS_TO]->(c)
                    """, name=law, chapter=chapter_name)

                # Concepts / Topics
                for concept in knowledge.get("concepts", []):
                    session.run("""
                        MERGE (t:Topic {name: $name})
                        WITH t
                        MATCH (c:Chapter {name: $chapter})
                        MERGE (t)-[:BELONGS_TO]->(c)
                    """, name=concept, chapter=chapter_name)

                # Formulas
                for formula in knowledge.get("formulas", []):
                    session.run("""
                        MERGE (f:Formula {expression: $expr})
                        WITH f
                        MATCH (c:Chapter {name: $chapter})
                        MERGE (f)-[:BELONGS_TO]->(c)
                    """, expr=formula, chapter=chapter_name)

                # Scientists
                for scientist in knowledge.get("scientists", []):
                    session.run("""
                        MERGE (s:Scientist {name: $name})
                        WITH s
                        MATCH (c:Chapter {name: $chapter})
                        MERGE (s)-[:CONTRIBUTED_TO]->(c)
                    """, name=scientist, chapter=chapter_name)

                # Definitions
                for defn in knowledge.get("definitions", []):
                    session.run("""
                        MERGE (d:Definition {term: $term})
                        WITH d
                        MATCH (c:Chapter {name: $chapter})
                        MERGE (d)-[:DEFINED_IN]->(c)
                    """, term=defn, chapter=chapter_name)

            # Create RELATED_TO relationships between concepts in adjacent chapters
            session.run("""
                MATCH (t1:Topic)-[:BELONGS_TO]->(c1:Chapter),
                      (t2:Topic)-[:BELONGS_TO]->(c2:Chapter)
                WHERE c1.num = c2.num - 1
                  AND (t1.name CONTAINS 'Electric' OR t1.name CONTAINS 'Magnetic')
                  AND (t2.name CONTAINS 'Electric' OR t2.name CONTAINS 'Magnetic')
                MERGE (t1)-[:RELATED_TO]->(t2)
            """)

            # Law USES Formula
            session.run("""
                MATCH (l:Law)-[:BELONGS_TO]->(c:Chapter),
                      (f:Formula)-[:BELONGS_TO]->(c)
                MERGE (l)-[:USES]->(f)
            """)

        logger.success("Knowledge graph built successfully")
        self._print_stats()

    def _print_stats(self):
        with self._driver.session() as session:
            result = session.run("""
                MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count
                ORDER BY count DESC
            """)
            stats = {r["label"]: r["count"] for r in result}
            logger.info(f"Graph stats: {stats}")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Graph-based retrieval using keyword matching on nodes.
        Returns chunks linked to matching concepts/laws/formulas.
        """
        if not self.available:
            return []

        query_lower = query.lower()
        query_words = [w for w in re.findall(r'\b\w{3,}\b', query_lower) if w not in
                       {'what', 'how', 'why', 'when', 'where', 'which', 'explain', 'define',
                        'give', 'state', 'derive', 'find', 'calculate', 'the', 'and', 'for'}]

        results = []

        with self._driver.session() as session:
            for word in query_words[:5]:  # limit query words
                # Search concepts and laws
                cypher_result = session.run("""
                    MATCH (n)-[:BELONGS_TO]->(c:Chapter)
                    WHERE toLower(n.name) CONTAINS $word
                       OR toLower(n.expression) CONTAINS $word
                       OR toLower(n.term) CONTAINS $word
                    WITH n, c
                    OPTIONAL MATCH (chunk:Chunk)-[:BELONGS_TO]->(c)
                    RETURN 
                        labels(n)[0] AS node_type,
                        COALESCE(n.name, n.expression, n.term) AS node_name,
                        c.name AS chapter,
                        chunk.id AS chunk_id,
                        chunk.page AS page,
                        chunk.content AS content
                    LIMIT $limit
                """, word=word, limit=top_k)

                for record in cypher_result:
                    if record["chunk_id"]:
                        results.append({
                            "chunk_id": record["chunk_id"],
                            "content": record["content"] or "",
                            "metadata": {
                                "page": record["page"] or 0,
                                "chapter": record["chapter"],
                                "node_type": record["node_type"],
                                "node_name": record["node_name"],
                            },
                            "graph_path": f"{record['node_type']}: {record['node_name']} → {record['chapter']}",
                            "graph_score": 0.8,
                            "retrieval_source": "graph",
                        })

            # Also do a path-based multi-hop search
            hop_result = session.run("""
                MATCH path = (n1)-[r1]->(c:Chapter)<-[r2]-(n2)
                WHERE toLower(n1.name) CONTAINS $query
                   OR toLower(n2.name) CONTAINS $query
                WITH n1, n2, c, type(r1) AS rel1, type(r2) AS rel2
                OPTIONAL MATCH (chunk:Chunk)-[:BELONGS_TO]->(c)
                RETURN 
                    n1.name AS from_node,
                    rel1 AS relation,
                    c.name AS chapter,
                    n2.name AS to_node,
                    chunk.id AS chunk_id,
                    chunk.page AS page,
                    chunk.content AS content
                LIMIT $limit
            """, query=query_lower[:30], limit=top_k)

            for record in hop_result:
                if record["chunk_id"] and record["chunk_id"] not in [r["chunk_id"] for r in results]:
                    results.append({
                        "chunk_id": record["chunk_id"],
                        "content": record["content"] or "",
                        "metadata": {
                            "page": record["page"] or 0,
                            "chapter": record["chapter"],
                        },
                        "graph_path": f"{record['from_node']} --{record['relation']}--> {record['chapter']} <--{record['to_node']}",
                        "graph_score": 0.7,
                        "retrieval_source": "graph",
                    })

        # Deduplicate
        seen = set()
        unique_results = []
        for r in results:
            if r["chunk_id"] not in seen:
                seen.add(r["chunk_id"])
                unique_results.append(r)

        return unique_results[:top_k]

    def get_chapter_graph(self, chapter_name: str) -> Dict:
        """Get graph visualization data for a chapter."""
        if not self.available:
            return {"nodes": [], "edges": []}

        with self._driver.session() as session:
            result = session.run("""
                MATCH (n)-[r]->(c:Chapter {name: $chapter})
                RETURN labels(n)[0] AS type, 
                       COALESCE(n.name, n.expression, n.term) AS name,
                       type(r) AS relation
                LIMIT 30
            """, chapter=chapter_name)

            nodes = [{"id": chapter_name, "label": chapter_name, "type": "Chapter"}]
            edges = []
            seen_nodes = {chapter_name}

            for record in result:
                node_name = record["name"]
                if node_name and node_name not in seen_nodes:
                    nodes.append({"id": node_name, "label": node_name, "type": record["type"]})
                    seen_nodes.add(node_name)
                if node_name:
                    edges.append({"from": node_name, "to": chapter_name, "label": record["relation"]})

        return {"nodes": nodes, "edges": edges}
