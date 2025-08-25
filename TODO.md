**Big picture:** You’re building a local-first, single-process meta MCP server that discovers many upstream MCPs, normalizes their tools into “capability cards,” and returns a session-scoped toolset via hybrid IR → cross-encoder rerank → optional LLM planning, then proxies calls under RBAC/ACL. Postgres+pgvector store the catalog and vectors; Tantivy powers BM25. Models are pluggable (ONNX/API) and latency is mode-tunable (fast/balanced/thorough). The monolith runs great on a laptop and upgrades cleanly to cloud.

**Tips, tricks, gotchas:** Persist BM25 and dense indexes; don’t rebuild at boot. Prefer Tantivy over pure-Python BM25. Enforce planner output with a JSON grammar; add auto-repair then fallback to “no-planner” exposure. Collect per-tool success/latency from the proxy; utility scoring beats cosine alone. Put a thin auth layer on the Admin UI even on localhost. Secrets: Keychain (macOS), Secret Service (Linux), with an encrypted file vault fallback; avoid interactive prompts in server mode. Ingest with producer/consumer and backoff; snapshot catalogs; treat upstream metadata as untrusted; run example probes to downrank dishonest tools. Containerize early; version internal APIs (`/v1/...`) from day one.

```xml
<project name="Meta MCP Monolith" version="v1">
  <assumptions>
    <runtime>Python 3.11+, FastAPI/uvicorn, httpx (async), uvloop</runtime>
    <hardware>Local dev on Apple M3/M4 or NVIDIA 40xx; CPU fallback</hardware>
    <storage>Postgres 14+ with pgvector; BM25 via Tantivy (Rust bindings)</storage>
    <clients>Claude Code + custom agent via standard MCP</clients>
    <catalogScale>10k–100k tools over time; refresh hours–days</catalogScale>
  </assumptions>

  <nonGoals>
    <item>No multi-tenant SaaS control plane in v1</item>
    <item>No quota/rate limits beyond basic circuit breakers</item>
  </nonGoals>

  <phase id="0" title="Repo bootstrap & foundations">
    <steps>
      <step>Create mono-repo structure: /app, /admin_ui, /migrations, /scripts, /docs, /tests.</step>
      <step>Add pyproject.toml with dependencies: fastapi, uvicorn[standard], httpx, pydantic, sqlalchemy, asyncpg, alembic, opentelemetry-sdk, opentelemetry-instrumentation-fastapi, keyring, cryptography, orjson, numpy, onnxruntime, uvloop.</step>
      <step>Add Rust toolchain (for tantivy bindings) and bindgen in build instructions.</step>
      <step>Define internal API prefix /v1 for all endpoints (public MCP under /mcp).</step>
      <step>Add Dockerfile (CPU baseline) and docker-compose.yml with Postgres; ensure pgvector extension enabled.</step>
      <step>Provision Postgres: CREATE EXTENSION IF NOT EXISTS vector; set shared_buffers/work_mem sensibly for laptop.</step>
      <step>Wire config loader (YAML + env override) with a strict schema; provide default profiles (fast/balanced/thorough).</step>
    </steps>
    <acceptance>Server boots with /healthz and config validation; Postgres reachable; pgvector available.</acceptance>
  </phase>

  <phase id="1" title="Schema, migrations, and indexing surfaces">
    <steps>
      <step>Implement Alembic migrations for tables: origins, tools, tool_embeddings, tool_stats, policies, sessions, traces, secrets.</step>
      <step>Table highlights:
        <detail>origins(url UNIQUE, auth_type, status, meta JSONB, tls_pinning optional)</detail>
        <detail>tools(origin_id, name UNIQUE per origin, version, brief, description, args_schema JSONB, returns_schema JSONB, categories[], examples JSONB, last_seen_at, deprecated)</detail>
        <detail>tool_embeddings(tool_id PK, embedding vector(768) default, model, updated_at)</detail>
        <detail>tool_stats(tool_id, window_start, window_len, p50_ms, p95_ms, success_rate, error_taxonomy JSONB)</detail>
        <detail>policies(org, role, allow_list[], deny_list[], meta JSONB)</detail>
      </step>
      <step>Create pgvector HNSW index with vector_l2_ops; set (m=16, ef_construction=200). Tune ef_search per latency mode.</step>
      <step>Implement Tantivy index builder:
        <detail>Fields: name (text), brief (text), description (text), origin (keyword), categories (facet), version (keyword), tool_id (u64)</detail>
        <detail>Analyzer: standard + lowercase + stopwords; store doc mapping tool_id→DB row</detail>
        <detail>Persistence: index directory under $DATA_DIR/bm25; mmap on open; atomic snapshot/replace on rebuild</detail>
      </step>
      <step>Add incremental “delta” updates to Tantivy on tool insert/update/delete.</step>
    </steps>
    <acceptance>DDL applied; pgvector and Tantivy indexes build from seed data; basic search returns results.</acceptance>
  </phase>

  <phase id="2" title="Model runtime (pluggable, local-first)">
    <steps>
      <step>Implement ModelRegistry with providers: onnx_local, api_remote.</step>
      <step>Embeddings:
        <detail>Default: small 768-d sentence embedding ONNX; device selection (CUDA, MPS, CPU) via onnxruntime providers list</detail>
        <detail>Batching: micro-batch 32–128 tokens; normalize vectors; enforce dim at write-time</detail>
      </step>
      <step>Reranker:
        <detail>Cross-encoder ONNX (100–150M params). Interface: score(query_text, tool_texts[]) → float[]</detail>
        <detail>Dynamic batching; early-exit on margin; int8 option via onnxruntime EP if available</detail>
      </step>
      <step>Planner:
        <detail>Provider off by default; when on: API LLM with JSON grammar (e.g., regex/EBNF enforced via a streaming parser)</detail>
        <detail>Retry policy: parse→repair (few-shot w/ last error)→fallback to trivial plan</detail>
      </step>
      <step>Warmup hooks on server start to compile sessions and JIT kernels; time-box to avoid long cold starts.</step>
    </steps>
    <acceptance>Embedding/rerank calls return within configured budgets; planner optional path returns valid JSON or falls back.</acceptance>
  </phase>

  <phase id="3" title="Ingestion: discovery, validation, snapshots">
    <steps>
      <step>Implement /v1/admin/origins (add/list/delete). Store auth and TLS settings in meta; secrets referenced by locator keys.</step>
      <step>Build crawler:
        <detail>Async fan-out with bounded concurrency; per-origin connection pool</detail>
        <detail>Retry/backoff (jittered exponential); classify failures (auth, network, schema)</detail>
        <detail>Fetch /tools and metadata; validate JSON Schema; normalize into capability cards</detail>
      </step>
      <step>Optional health probes:
        <detail>For each tool with examples, perform dry-run (side-effect-free if declared) in a sandbox and record latency/validity</detail>
        <detail>Mark dishonest/incoherent descriptions for downranking</detail>
      </step>
      <step>Write DB rows (origins/tools). Upsert embeddings asynchronously. Update Tantivy incrementally.</step>
      <step>Create signed catalog snapshots (versioned) for rollback; store manifest in DB and files under $DATA_DIR/snapshots.</step>
      <step>Schedule periodic refresh (default 24h) with a manual /v1/admin/refresh endpoint.</step>
    </steps>
    <acceptance>Given N origins, tools materialize with embeddings and BM25 docs; snapshot is created; refresh updates last_seen_at & deprecates missing tools.</acceptance>
  </phase>

  <phase id="4" title="Selection: hybrid IR → rerank → MMR → policy → modes">
    <steps>
      <step>Implement BM25 search (top Kb) and dense ANN query (top Kd). Merge candidates by union with source tags.</step>
      <step>Rerank union with cross-encoder on (goal_text || recent_context, card_text).</step>
      <step>MMR diversify to M candidates using cosine on embeddings; λ≈0.3 configurable.</step>
      <step>Optional utility score: U = α*s + β*success_rate − γ*p95_ms − ρ*risk_penalty; coefficients per-mode in config.</step>
      <step>Policy/ACL filter applied both pre- and post-ranking; deny_list wins over allow_list.</step>
      <step>Latency modes:
        <detail>fast: Kb=24,Kd=16, rerank 16, expose 5</detail>
        <detail>balanced: 48,32,32,8</detail>
        <detail>thorough: 96,64,64,10 + planner</detail>
      </step>
      <step>Implement cache: (goal_hash, policy_version, catalog_snapshot) → selected tool_ids (short TTL).</step>
      <step>Expose selection via /v1/debug/select for admin testing with scores and rationale (top features, ties).</step>
    </steps>
    <acceptance>Selection returns M tools within mode budget; debug endpoint shows BM25/dense/rerank scores and MMR choices.</acceptance>
  </phase>

  <phase id="5" title="Planner: constrained minimal DAG (optional)">
    <steps>
      <step>Define JSON Schema for DAG: nodes (tool_id, args), edges (from.node.output → to.node.input), invariants (acyclic, bound args).</step>
      <step>Prompt template: goal + compact cards (brief + args schema) + constraints; decode with grammar to enforce valid JSON.</step>
      <step>Implement repair loop: if schema invalid, include validator errors back to the model; cap at 2 attempts.</step>
      <step>Trivial plan fallback: expose top-M tools with empty DAG; add a universal “search tool”.</step>
      <step>Validator checks: cycles, missing bindings, illegal tools per policy; strip unsafe nodes.</step>
    </steps>
    <acceptance>Planner produces valid DAG ≥95% when enabled; otherwise safe fallback; invalid nodes never reach proxy.</acceptance>
  </phase>

  <phase id="6" title="Proxy/executor: safe, observable calls">
    <steps>
      <step>Implement upstream client with timeouts (connect/read), per-origin concurrency caps, and circuit breakers (open on p95 spikes & 5xx bursts).</step>
      <step>Args hardening: JSON Schema validation; size caps; enum allowlists; redact marked fields from logs.</step>
      <step>Idempotency: attach x-meta-request-id; retry only idempotent methods; exponential backoff with jitter.</step>
      <step>Response streaming: pass-through chunked responses where supported; record byte counts and duration.</step>
      <step>Metrics emission per call: success, latency, payload sizes, error taxonomy; write rollup into tool_stats.</step>
      <step>Provenance headers: origin_id, tool_id, catalog_version; expose to client for audit.</step>
    </steps>
    <acceptance>Proxy enforces budgets; failures classified; stats table updates; traces show end-to-end spans.</acceptance>
  </phase>

  <phase id="7" title="Policy, RBAC, ACL">
    <steps>
      <step>Define roles: admin/operator/user. Policy resolution: deny_list &gt; allow_list &gt; default.</step>
      <step>Enforcement points: selection (filter candidates), planner (strip nodes), proxy (block execution).</step>
      <step>Policy versioning and audit logs for every allow/deny decision with reasons.</step>
    </steps>
    <acceptance>Admin can restrict an origin/tool and see immediate effect in selection and proxy; audits visible in UI.</acceptance>
  </phase>

  <phase id="8" title="Secrets and auth">
    <steps>
      <step>Implement SecretStore with providers: macos_keychain, linux_secret_service, file_vault (AES-GCM with scrypt key derivation; passphrase CLI or env).</step>
      <step>Non-interactive mode for server: read vault key from env/OS store; fail closed if unavailable.</step>
      <step>Never forward raw user tokens; where possible mint short-lived per-origin proxy tokens (signed JWT) with narrow scopes.</step>
    </steps>
    <acceptance>Origins can be configured with credentials; secrets not logged; restart does not prompt in server mode.</acceptance>
  </phase>

  <phase id="9" title="Admin UI (local-first)">
    <steps>
      <step>Serve under /admin with simple auth (username/password in config, HTTPS recommended even on LAN); optional GitLab OAuth later.</step>
      <step>Views:
        <detail>Tool Search: query, filters (origin, category, deprecated), scores (BM25/dense/rerank), quick “dry-run select”</detail>
        <detail>ACL Editor: allow/deny per origin/tool; policy preview; version diff</detail>
        <detail>Logs/Traces: recent sessions, selection decisions, proxy outcomes; filter by origin/tool/session</detail>
      </step>
      <step>Safety UI: provenance badges; one-click deprecate or downrank suspicious tools.</step>
    </steps>
    <acceptance>Admin can search tools, modify ACLs, and observe selections/calls in near real-time.</acceptance>
  </phase>

  <phase id="10" title="Observability & reliability">
    <steps>
      <step>OpenTelemetry for traces (FastAPI, httpx); metrics via Prometheus exporter; structured logs (JSON) with request-id.</step>
      <step>Define SLOs: selection p95 budgets per mode; availability 99.9%; proxy error budget &lt;1%.</step>
      <step>Degradation modes: reranker→off; dense→BM25-only; stale catalog→last-good snapshot; Postgres read-only→serve from snapshot.</step>
      <step>Health endpoints: /healthz (liveness), /readyz (DB/index ready), /status (catalog snapshot info).</step>
    </steps>
    <acceptance>Dashboards show p50/p95, error taxonomy; breaker activates on upstream incidents; health endpoints reflect state.</acceptance>
  </phase>

  <phase id="11" title="Evaluation scaffold & tuning">
    <steps>
      <step>Define YAML task format: goal, gold_tools[], mode, notes.</step>
      <step>Implement offline runner: compute Recall@M, selection latency, unnecessary exposure %, and success via dry-run proxy.</step>
      <step>Grid search over {Kb,Kd,top_rerank,M,λ,α,β,γ,ρ} to produce a “profile” per mode; export to config.</step>
      <step>Curate and expand golden set continuously; tag by domain (code/docs/search/cloud-ops).</step>
    </steps>
    <acceptance>Runner outputs a profile that improves Recall@M without breaking latency budgets; profiles load via config.</acceptance>
  </phase>

  <phase id="12" title="Packaging & distribution">
    <steps>
      <step>CPU Docker image; optional GPU image with CUDA base and onnxruntime-gpu; multi-arch for Apple Silicon.</step>
      <step>Entrypoint performs migrations, index warmup (bounded), and starts server.</step>
      <step>Versioning scheme (semver) for internal APIs and catalog snapshots; embed git SHA in /status.</step>
      <step>Provide make targets: make dev, make test, make run, make docker-build, make migrate.</step>
    </steps>
    <acceptance>`docker compose up` yields a working local instance with Admin UI and selection endpoint.</acceptance>
  </phase>

  <phase id="13" title="Cloud-ready upgrades (no rewrite)">
    <steps>
      <step>Configurable Postgres DSN → managed Postgres; object storage for snapshot artifacts.</step>
      <step>External model providers via API keys; small GPU node for reranker if desired.</step>
      <step>Add TLS/mTLS to upstreams; percent-based canary for selection parameters; API rate limiting.</step>
    </steps>
    <acceptance>Same binary runs in cloud with minimal config changes; observability exports to hosted backends.</acceptance>
  </phase>

  <phase id="14" title="Milestones & acceptance gates">
    <steps>
      <step>M1 Skeleton: schema + ingest + BM25/dense + fast mode + proxy + logs.</step>
      <step>M2 Rerank + Admin UI: cross-encoder + modes + ACL editor + traces.</step>
      <step>M3 Planner optional + policies: DAG with grammar; session-scoped exposure; audits.</step>
      <step>M4 Hardening: secrets facade; circuit breakers; snapshots/rollbacks; error taxonomy.</step>
      <step>M5 Scale polish: re-embed pipeline; index compaction; golden-set tuning for 50k+ tools.</step>
    </steps>
    <acceptance>Each milestone ships with reproducible docker-compose, test fixtures, and a short runbook.</acceptance>
  </phase>

  <phase id="15" title="Runbooks, risks, and rollbacks">
    <steps>
      <step>Runbooks: Postgres issues, index rebuilds, snapshot rollback, upstream outage response.</step>
      <step>Risks:
        <detail>Planner instability → keep off by default; enable per-session</detail>
        <detail>Index bloat → compaction jobs; TTL for deprecated tools</detail>
        <detail>Catalog poisoning → probe tests; provenance surfacing; downranking</detail>
      </step>
      <step>Rollback: single command to activate previous snapshot; policy pinning by version.</step>
    </steps>
    <acceptance>On-call can mitigate incidents using documented procedures in &lt;15 minutes.</acceptance>
  </phase>

  <artifacts>
    <item>Postgres DDL + Alembic migrations</item>
    <item>Config schema (YAML) with mode presets</item>
    <item>Model registry with ONNX/API providers</item>
    <item>Tantivy BM25 indexer with persistence and delta updates</item>
    <item>Hybrid select → rerank → MMR pipeline with utility scoring</item>
    <item>Planner (optional) with JSON grammar and repair loop</item>
    <item>Proxy with circuit breakers, idempotency, and provenance</item>
    <item>RBAC/ACL policy engine + audits</item>
    <item>SecretStore providers (Keychain/Secret Service/FileVault)</item>
    <item>Admin UI (tool search, ACL, logs/traces, dry-run select)</item>
    <item>Observability dashboards (OTel/Prometheus), health endpoints</item>
    <item>Evaluation runner and profiles</item>
    <item>Dockerized distribution and runbooks</item>
  </artifacts>
</project>
```
