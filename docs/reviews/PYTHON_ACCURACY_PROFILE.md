# Python accuracy profile and integration API v1

Base: 4686c82b53b6745acd3a760599cb94d709a9ba79 (0.3.18). No business repository was indexed.

`codebase_rag.integration.api.build_snapshot` accepts explicit Python file lists and returns captured source provenance, canonical Repository/Module/Class/Function/CallSite nodes, evidence-bearing relationships, observations, file deltas, source identity, graph snapshot identity and the existing upstream parser fingerprint. `analyze_sources` is the lower-level fixed-byte interface. Consumers must not rely on GraphUpdater internals. This is a Python-only, bounded initial API; schema/API changes require an explicit version change.

`integration.neo4j.Neo4jGraph` uses the official driver and an explicit namespace. One transaction replaces nodes/observations belonging to that namespace, validates counts and captured sources and switches CGScope.active. Other namespaces and labels are retained. Historical COMPLETE metadata does not promise retained historical nodes. The caller must close independent read entrances during coordinated graph/retrieval publication; online Protobuf manifest checks are not a publication boundary.

The existing native GraphUpdater now consumes this Python profile for CALLS/REFERENCES/INSTANTIATES, while other languages and other passes retain their previous processing. File delta detection/provenance and parser fingerprint remain upstream facilities. The public snapshot API currently reparses and revalidates the whole explicit scope on each update; its delta describes changes, not optimized incremental parsing. Full-vs-update equivalence must be assessed independently.

## Resolution semantics

CALLS originates at a positioned CallSite, with its execution owner recorded separately. Function references, imports, inheritance and containment are distinct. All callable targets are candidates, `exhaustive=false`; a single candidate is never proof of certainty. Unknown and rejected origins remain on the call site. Module/default/decorator execution is separate from a deferred function or lambda body. Anonymous functions have per-owner source-order ordinals, including same-line lambdas. Snapshot node IDs include captured content identity and a logical key; moves/redefinitions may change identity. Never use Neo4j numeric IDs as persistent source identity.

Default resolution uses bounded value propagation, branch joins, sequential overwrite kills and LEGB scope. Jedi mode adds real Jedi 0.20.1 inference against a mirror of captured source, and declaration goto only for repository-anchored classmethod descriptor artifacts. Real external results remain boundaries. Neither mode is whole-program sound Python analysis. Dynamic imports, arbitrary decorators/metaclasses/monkey patching, exception control flow and reflection can remain unknown or yield over-approximations. No exhaustive classification is inferred from cardinality.

The frozen 39-site corpus was copied unchanged from local-code-tools/experiments/python_graph; it is independent manual truth. `python-accuracy-before.json` and `python-accuracy-after.json` record actual native GraphUpdater output for 11 minimal reproductions in both frontends (sink mocked, parsing real). Test-program success does not certify every frozen positive or whole-repository precision/recall. No real retrieval quality or paid provider was tested.

## Reproduce

Use Python 3.12 and an isolated environment. Install this fork with `pip install . --no-deps` after resolving the locked integration environment described by the consuming local-code-tools repository. In a source checkout, uv sources build the Rust extension; the integration environment deliberately uses the published cgr-graph-rs wheel and Python grammar only, and does not require Docker, Rust or Memgraph. Install `jedi==0.20.1`, `neo4j==6.4.1`, test dependencies and tree-sitter-python.

```powershell
python -m pytest -o addopts='' -q codebase_rag/tests/test_python_accuracy_profile.py codebase_rag/tests/test_python_frontend.py codebase_rag/tests/test_python_nested_functions.py codebase_rag/tests/test_python_scope_bound_names.py codebase_rag/tests/test_python_inheritance.py
```

No LLM or embedding endpoint is invoked by these tests. Native Neo4j and two actual HTTP MCP calls are verified in the consuming repository, using isolated fixtures.

## Executed validation (2026-10-27)

The selected real updater regression run completed: 49 passed, 3 skipped. The three skips require the absent JavaScript grammar (not Python capability passes). It included this profile, existing Jedi frontend, nested functions, scope, inheritance and factory-return-chain tests; no full upstream suite or CI claim is made. Declared factory return types are retained as candidates with unknown runtime origins, rather than losing previously supported annotated receiver links.

The frozen evaluator records 39 manually annotated sites, 37 closed-truth sites, in each mode: TP 37, FP 0, FN 1, precision 1.0, recall 0.973684. The missing edge is `json.loads` outside the captured repositories; its positioned external boundary is retained instead of fabricating a local definition node. These are narrow labeled-set metrics (including that strict external edge expectation), not repository-wide accuracy. All marked call attempts and owners were retained; no exhaustive/definite target claim was made. The full rows retain expectations, candidates, rejected inference and reasons, so unsupported behavior is visible.

`ty` uses the repository's locked ty 0.0.84 and pydantic-ai-slim 2.53.0 in an isolated optional-dependency environment. An initial drifted environment produced deprecation diagnostics and was corrected to the locked version. The dynamically imported pinned Julia grammar returns PyCapsule; a type-only cast selects that actual Language overload rather than the deprecated integer overload (native Julia parser construction smoke-tested). No runtime Julia resolution behavior changed. The Windows benchmark RSS helper now makes its pre-existing Windows `None` fallback explicit before importing POSIX `resource`; runtime behavior is preserved. Frozen intentionally invalid fixtures have precise lint and formatter exclusions so their source bytes stay frozen.
