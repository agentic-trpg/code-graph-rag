"""Namespace-scoped Neo4j snapshots with transactional staging and compare/swap publish."""

import json
import secrets
import uuid

from neo4j import GraphDatabase, Query, unit_of_work

from .model import RELATIONS, canonical, digest, validate, verify_sources


class Neo4jGraph:
    """No database-wide deletion, external-service writes or multi-database requirement."""

    def __init__(self, uri, namespace, auth=None, database="neo4j", seconds=5):
        if not namespace or len(namespace) > 100:
            raise ValueError("Explicit namespace required")
        self.namespace = namespace
        self.database = database
        self.seconds = seconds
        self.driver = GraphDatabase.driver(
            uri, auth=auth, max_connection_pool_size=4, connection_timeout=5
        )
        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

    def read(self, query, **params):
        """Bound even fixed-length adjacency retrieval by a server transaction timeout."""
        with self.driver.session(database=self.database) as session:
            return session.run(
                Query(query, timeout=self.seconds), ns=self.namespace, **params
            ).data()

    def transaction(self, callback):
        """Managed retry is safe because every writer is idempotent and scoped."""
        with self.driver.session(database=self.database) as session:
            return session.execute_write(unit_of_work(timeout=self.seconds)(callback))

    def setup(self):
        """Create only CG-specific constraints/indexes and our namespace's pointer record."""
        for label in ["CGScope", "CGSnapshot", "CGNode", "CGObservation"]:
            self.read(
                f"CREATE CONSTRAINT {label.lower()}_key IF NOT EXISTS FOR (n:{label}) REQUIRE n.key IS UNIQUE"
            )
        self.read(
            "CREATE INDEX cgnode_scope_kind IF NOT EXISTS FOR (n:CGNode) ON (n.namespace,n.snapshot_id,n.kind)"
        )
        self.read(
            "MERGE (s:CGScope {key:$ns}) ON CREATE SET s.active='',s.cursor_secret=$secret,s.epoch=0",
            secret=secrets.token_hex(32),
        )

    def active(self):
        rows = self.read("MATCH (s:CGScope {key:$ns}) RETURN s.active AS active")
        return (rows[0]["active"] if rows else "") or None

    def complete(self, snapshot_id=None):
        """Pin an explicitly complete immutable snapshot; READY/BUILDING are invisible."""
        ident = snapshot_id or self.active()
        if not ident:
            raise ValueError("No complete active graph snapshot")
        rows = self.read(
            "MATCH (s:CGSnapshot {key:$key}) WHERE s.namespace=$ns AND s.state='COMPLETE' RETURN s.snapshot_id AS id",
            key=self.namespace + ":" + ident,
        )
        if not rows:
            raise ValueError("Snapshot incomplete, missing, or wrong namespace")
        return ident

    def build(self, bundle, fail_at=None, batch_size=200, on_staged=None):
        """One transaction writes/validates/publishes, so official read-cypher sees no staging nodes."""
        validate(bundle)
        verify_sources(bundle)
        snapshot = bundle["snapshot_id"]
        key = self.namespace + ":" + snapshot
        writer = uuid.uuid4().hex
        content = digest(
            canonical(
                {
                    "nodes": sorted(bundle["nodes"], key=lambda node: node["node_id"]),
                    "edges": sorted(bundle["edges"], key=lambda edge: edge["edge_id"]),
                    "observations": sorted(bundle["observations"], key=canonical),
                }
            )
        )
        previous = self.active()

        def prepare(tx):
            row = tx.run(
                "MERGE (s:CGSnapshot {key:$key}) ON CREATE SET s.namespace=$ns,s.snapshot_id=$sid,s.content=$content,s.state='BUILDING',s.lock=0 SET s.lock=s.lock+1 RETURN s.state AS state,s.content AS content",
                key=key,
                ns=self.namespace,
                sid=snapshot,
                content=content,
            ).single()
            if row["content"] != content:
                raise ValueError(
                    "Snapshot content conflict; source/adapter identity insufficient"
                )
            if row["state"] == "COMPLETE" and previous == snapshot:
                return "COMPLETE"
            tx.run(
                "MATCH(s:CGSnapshot {key:$key}) SET s.state='BUILDING',s.writer=$writer",
                key=key,
                writer=writer,
            ).consume()
            return "BUILDING"

        if self.transaction(prepare) == "COMPLETE":
            return dict(
                snapshot_id=snapshot, previous=previous, state="COMPLETE", reused=True
            )
        node_rows = [
            dict(
                key=self.namespace + ":" + n["node_id"],
                node_id=n["node_id"],
                snapshot_id=snapshot,
                namespace=self.namespace,
                kind=n["kind"],
                name=n["qualname"],
                name_lower=n["qualname"].lower(),
                repository=n["repository"],
                path=n["path"],
                parent=n["parent"],
                resolution_status=n.get("resolution_status", ""),
                execution_owner=n.get("execution_owner", ""),
                active_call=n.get("active_call", False),
                payload=canonical(n),
            )
            for n in bundle["nodes"]
        ]

        def build(tx):
            row = tx.run(
                "MATCH(s:CGSnapshot {key:$key}) SET s.lock=s.lock+1 RETURN s.writer AS writer,s.state AS state",
                key=key,
            ).single()
            if row["writer"] != writer or row["state"] != "BUILDING":
                raise ValueError("Build ownership changed")
            tx.run(
                "MATCH(n:CGNode {namespace:$ns}) WHERE n.snapshot_id<>$sid DETACH DELETE n",
                ns=self.namespace,
                sid=snapshot,
            ).consume()
            tx.run(
                "MATCH(n:CGObservation {namespace:$ns}) WHERE n.snapshot_id<>$sid DELETE n",
                ns=self.namespace,
                sid=snapshot,
            ).consume()
            for offset in range(0, len(node_rows), batch_size):
                tx.run(
                    "UNWIND $rows AS row MERGE(n:CGNode {key:row.key}) SET n += row",
                    rows=node_rows[offset : offset + batch_size],
                ).consume()
                if fail_at == "batch":
                    raise RuntimeError("Injected intermediate build failure")
            for kind in sorted(RELATIONS):
                edges = [
                    dict(
                        key=self.namespace + ":" + e["edge_id"],
                        source=self.namespace + ":" + e["source"],
                        target=self.namespace + ":" + e["target"],
                        caller_id=e.get("caller_id", ""),
                        status=e["status"],
                        payload=canonical(e),
                    )
                    for e in bundle["edges"]
                    if e["kind"] == kind
                ]
                for offset in range(0, len(edges), batch_size):
                    tx.run(
                        f"UNWIND $rows AS row MATCH(a:CGNode {{key:row.source}}),(b:CGNode {{key:row.target}}) MERGE(a)-[r:{kind} {{key:row.key}}]->(b) SET r += row",
                        rows=edges[offset : offset + batch_size],
                    ).consume()
            observations = [
                dict(
                    key=key + ":" + str(i),
                    namespace=self.namespace,
                    snapshot_id=snapshot,
                    payload=canonical(o),
                )
                for i, o in enumerate(bundle["observations"])
            ]
            if observations:
                tx.run(
                    "UNWIND $rows AS row MERGE(o:CGObservation {key:row.key}) SET o += row",
                    rows=observations,
                ).consume()
            counts = tx.run(
                "MATCH(n:CGNode {namespace:$ns,snapshot_id:$sid}) RETURN count(n) AS n",
                ns=self.namespace,
                sid=snapshot,
            ).single()["n"]
            edge_count = tx.run(
                "MATCH(n:CGNode {namespace:$ns,snapshot_id:$sid})-[r]->(m:CGNode {namespace:$ns,snapshot_id:$sid}) RETURN count(r) AS n",
                ns=self.namespace,
                sid=snapshot,
            ).single()["n"]
            if counts != len(node_rows) or edge_count != len(bundle["edges"]):
                raise ValueError("Stored count validation failed")
            tx.run(
                "MATCH(s:CGSnapshot {key:$key}) SET s.state='READY',s.node_count=$n,s.edge_count=$e,s.identity=$identity",
                key=key,
                n=counts,
                e=edge_count,
                identity=canonical(bundle["identity"]),
            ).consume()
            if on_staged:
                on_staged(snapshot)
            verify_sources(bundle)
            scope = tx.run(
                "MATCH(s:CGScope {key:$ns}) SET s.epoch=s.epoch+1 RETURN s.active AS active",
                ns=self.namespace,
            ).single()
            if (scope["active"] or None) != previous:
                raise ValueError("Active graph changed; publication CAS refused")
            tx.run(
                "MATCH(s:CGScope {key:$ns}) SET s.active=$sid",
                ns=self.namespace,
                sid=snapshot,
            ).consume()
            if fail_at == "publish":
                raise RuntimeError("Injected publication failure")
            tx.run(
                "MATCH(s:CGSnapshot {key:$key}) SET s.state='COMPLETE'", key=key
            ).consume()
            return dict(
                snapshot_id=snapshot, previous=previous, state="COMPLETE", reused=False
            )

        try:
            return self.transaction(build)
        except BaseException:
            self.read(
                "MATCH(s:CGSnapshot {key:$key}) WHERE s.writer=$writer AND s.state='BUILDING' SET s.state='FAILED'",
                key=key,
                writer=writer,
            )
            raise

    def node(self, snapshot, node_id):
        rows = self.read(
            "MATCH(n:CGNode {key:$key,namespace:$ns,snapshot_id:$sid}) RETURN n.payload AS payload",
            key=self.namespace + ":" + node_id,
            sid=snapshot,
        )
        return json.loads(rows[0]["payload"]) if rows else None

    def cursor_secret(self):
        rows = self.read("MATCH(s:CGScope {key:$ns}) RETURN s.cursor_secret AS secret")
        return rows[0]["secret"].encode()
