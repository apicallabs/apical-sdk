<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/apical-plexus-dark.svg" />
    <source media="(prefers-color-scheme: light)" srcset="assets/logo/apical-plexus.svg" />
    <img src="assets/logo/apical-plexus-dark.svg" alt="Apical Labs — Apical Arbor & Apical Plexus" width="380" />
  </picture>
</p>

# `apical` — Official Python SDK & Universal MCP Router (`v0.5.5`)

[![PyPI version](https://img.shields.io/pypi/v/apical.svg)](https://pypi.org/project/apical/)
[![Python Versions](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://pypi.org/project/apical/)
[![GitHub](https://img.shields.io/badge/GitHub-apicallabs%2Fapical--sdk-blue?logo=github)](https://github.com/apicallabs/apical-sdk)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Official Python thin client and Universal Model Context Protocol (`MCP`) server for the **Apical Labs** platform, providing **Apical Arbor** (continual linear readout on frozen embeddings) and **Apical Plexus** (multi-layer non-linear continual adaptation, semantic MCP tool routing, and bare-metal edge microcircuits).

* **Homepage:** [apical-labs.com](https://apical-labs.com)
* **Documentation & Specs:** [apical-labs.com/docs](https://apical-labs.com/docs)
* **Developer Console:** [apical-labs.com/console](https://apical-labs.com/console)
* **GitHub Repository:** [github.com/apicallabs/apical-sdk](https://github.com/apicallabs/apical-sdk)

---

## Core Architectural Invariants

- **Strict Zero Data Retention (`persisted_bytes_delta == 0`):** Every online update (`learn`, `reinforce`, `unlearn`, `bridge`) operates in **constant $O(1)$ memory** on a pre-allocated fixed-byte synaptic tensor container. No replay buffers, no historical vectors, and no raw text are ever written to disk, logs, or databases.
- **Sub-Millisecond CPU Plasticity:** Single-pass directional synaptic realignment executes in **`< 0.10 ms`** (Arbor) to **`< 0.22 ms`** (Plexus) on standard CPU hardware without backpropagation or GPU fine-tuning queues.
- **Client-Side Keyed Privacy Blindness:** Optionally blind input vectors on your own infrastructure via a deterministic client-held secret key (`OrthogonalPrivacyProjector`) before they ever leave your servers.
- **Offline Cryptographic Attestation (`Ed25519`):** Every API and enclave operation returns a canonical JSON receipt signed by the enclave's hardware-protected `Ed25519` keypair, verifiable offline via `verify_zero_retention_receipt()`.
- **Deterministic Machine Unlearning (`< 1 ms`):** Purge any class or routing coordinate back to quiescent baseline in a single step to satisfy **GDPR / UK GDPR Article 17** right-to-be-forgotten mandates without retraining.
- **Zero-Downtime Cross-Backbone Migration (`< 20 ms`):** Upgrade foundation embedding encoders (e.g., `MiniLM-384D` $\rightarrow$ `BGE-512D`) using public anchor pairs with zero stored customer training data.

---

## Installation

```bash
pip install apical
```

**Minimal Dependencies:** `numpy>=1.24.0`, `httpx>=0.25.0`, `cryptography>=41.0.0`.
*Requires **ZERO PyTorch** and **ZERO heavy ML framework dependencies**.*

**CLI Entrypoints Installed Automatically:**
- `apical` — Top-level CLI (`apical mcp setup`, `apical mcp --dashboard`, `apical mcp --stats`, `apical mcp --wipe`)
- `apical-mcp` — Native JSON-RPC 2.0 `stdio` MCP server for Cursor, Claude Desktop, Windsurf, Cline, and Antigravity
- `apical-proxy` — Local HTTP zero-retention proxy server

---

## Architectural Selection & Production Use Cases (`Arbor` vs. `Plexus`)

To determine whether a production workload belongs on **Apical Arbor** or **Apical Plexus**, evaluate three foundational engineering criteria across feature geometry, execution caller, and lifecycle primitives:

### 1. Three Engineering Decision Criteria

| Criterion | Apical Arbor (`client.arbor`) | Apical Plexus (`client.plexus` / `apical-mcp`) |
| :--- | :--- | :--- |
| **Feature Geometry**<br>*(Separated Embeddings vs. Entangled States)* | Contrastive text and vision encoders (`BGE`, `MiniLM`, `OpenAI`, `DINOv2`, `CLIP`) already separate classes metrically. Use a single-layer directional readout in **`< 0.10 ms`**. | Raw causal LLM hidden states (`Qwen-2.5`, `Llama-3`), multi-hop agent intents, and raw unencoded sensor arrays are non-linearly entangled. Use multi-layer continual adaptation in **`< 0.22 ms`**. |
| **Execution Caller**<br>*(Backend Pipelines vs. Autonomous Agent Runtimes)* | Backend microservices, data ingestion pipelines, and ANSI C99 firmware passing dense floating-point or compact integer tensors directly via the Python/TypeScript SDK or REST API. | Autonomous LLM agents, coding IDEs (Cursor, Claude Desktop, Antigravity), and multi-agent orchestrators passing natural language prompts and JSON tool schemas via the native `apical-mcp` stdio daemon. |
| **Lifecycle Primitives**<br>*(Backbone Migration vs. Associative Recall)* | Upgrading foundation backbones (`MiniLM-384D` $\rightarrow$ `BGE-512D`) with zero stored customer data via `arbor.bridge` (**`< 20 ms`**), or 1-step GDPR Art. 17 class erasure via `arbor.unlearn` (**`< 1 ms`**). | Reconstructing 50% occluded or corrupted sensor frames without a neural backbone via `plexus.edge` associative recall, or aligning developer coding styles via `DeveloperPreferenceAdapter`. |

---

### 2. Production Use Cases Across 8 Domains

| Domain / Vertical | Apical Arbor<br>*(Classifier · MemoryRouter · Bridge · Unlearn · Edge)* | Apical Plexus<br>*(apical-mcp · Router · Network · PreferenceAdapter · Edge)* |
| :--- | :--- | :--- |
| **Autonomous Agents, IDEs & Tool Routing** | **Fixed-Embedding API Gateway Shortlisting (`arbor.MemoryRouter`)**<br>• **Microservice API Gateways:** Backend services that already compute `MiniLM-384D` embeddings prune 100–250 internal REST endpoints down to Top-3 in `< 0.05 ms` before calling an LLM.<br>• **Voice-Bot Command Dispatch:** Sub-millisecond intent matching from Whisper + MiniLM embeddings to deterministic IVR state-machine transitions. | **Native MCP Agent Orchestration & Prompt Deflation (`apical-mcp` / `plexus.Router`)**<br>• **IDE Coding Agents (Cursor / Claude / AGY):** Intercepts natural language prompts via stdio JSON-RPC to route across 500+ MCP tools (`git`, `stripe`, `postgres`, `k8s`), cutting prompt context from ~15,000 tokens to ~350 tokens (**`>92%` savings**) and reinforcing winning tools on accepted turns.<br>• **Enterprise 10,000-Tool Mesh:** Multi-layer semantic dispatch across massive internal OpenAPI catalogs where tool descriptions overlap non-linearly.<br>• **Zero-Rot Coding Style Alignment (`DeveloperPreferenceAdapter`):** Continuously learns a developer's architectural preferences (strict typing, DI patterns, test granularity) from accepted/rejected diffs without bloating `.cursorrules` or `AGENTS.md`. |
| **E-Commerce, Retail & Catalog Intelligence** | **Continual Visual & Text Taxonomy Readout (`arbor.Classifier`)**<br>• **Zero-Downtime Marketplace SKUs:** Classifies newly uploaded merchant product photos downstream of frozen `DINOv2` or `CLIP` backbones into 5,000+ dynamic categories in `< 0.10 ms`.<br>• **Flash-Trend Cold-Start Tagging:** Instantly adds a new seasonal category from 3–5 human-verified examples online without GPU batch retraining.<br>• **Brand & Counterfeit Logo Detection:** Real-time visual prototype matching at upload time with instant class addition when new brands onboard. | **Multi-Modal & Causal LLM Shopping Assistants (`plexus.Network`)**<br>• **Conversational Commerce State Tracking:** Adapts directly atop frozen `Llama-3` / `Qwen-2.5` hidden layer activations to predict complex, multi-turn buyer intent (e.g., distinguishing *"looking for a compatible replacement part"* vs. *"requesting a warranty return"*).<br>• **Cross-Category Bundle Association:** Uses multi-layer associative weights to map non-linear combinations of user session trajectories to high-conversion product bundles in constant $O(1)$ RAM. |
| **Customer Operations, SaaS & B2B Multi-Tenancy** | **Per-Tenant Continual Ticket & Document Triage (`arbor.Classifier` / `Unlearn`)**<br>• **Outage-Resilient Support Routing:** Auto-routes high-confidence tickets ($\ge 0.82$) while escalating novel outage symptoms to human operators whose single-click resolutions immediately train the readout head in `< 0.10 ms`.<br>• **Per-Customer Invoice & GL Filing:** Gives 10,000 B2B SaaS tenants their own isolated, fixed-byte classifier container (`persisted_bytes_delta == 0`) that learns custom general-ledger account codes on CPU.<br>• **Instant Tenant Offboarding (`arbor.Unlearn`):** Executes deterministic `< 1 ms` class erasure when a vendor or department is removed, emitting an `Ed25519` GDPR Art. 17 certificate. | **Entangled Enterprise Workflow & Fleet Coordination (`plexus.Network`)**<br>• **Multi-Hop Incident Escalation:** Routes complex enterprise IT/DevOps incidents where symptoms span logs, stack traces, and natural language descriptions that fail linear separation.<br>• **Shared-Manifold Multi-Agent Fleet Coordination:** A fleet of 10 specialized support subagents (Billing, Technical, Legal, Security) shares a single `Plexus Pro` container so when one agent learns that a new error code belongs to Billing, all peer agents route it accurately in real time. |
| **Healthcare, Finance & Regulated Compliance (HIPAA / GDPR)** | **Zero-Retention Clinical & Financial Readouts (`Classifier` + `OrthogonalPrivacyProjector`)**<br>• **Blinded Clinical ICD-10 Coding:** Hospital client blinds clinical note embeddings via a client-held secret key (`OrthogonalPrivacyProjector`), adapts an Arbor readout head to local hospital billing codes in constant memory, and retains zero patient text or raw vectors.<br>• **Statutory Right-to-be-Forgotten Webhooks (`arbor.Unlearn`):** Financial institutions or biobanks legally required to prove a specific cohort or identity class has been purged from production weights (`target_class_weight_norm == 0.0`) without spending £40,000+ retraining a foundation model. | **Regulated Multi-Layer Risk & Clinical Triage (`plexus.Network` on Regulated Tier)**<br>• **Real-Time Wire & Payment Fraud Typology Adaptation:** Adapts a multi-layer non-linear head atop transaction graph activations as new fraud rings emerge mid-day, under a pre-signed HIPAA BAA / GDPR Art. 28 DPA with an immutable append-only cryptographic receipt ledger.<br>• **Zero-Retention Clinical Trial Cohort Matching:** Matches complex multi-variable patient eligibility criteria atop medical LLM hidden states (`BioMistral` / `Med-PaLM`) where linear classifiers collapse due to non-linear feature interactions (*Drug A + Biomarker B, excluding Comorbidity C*). |
| **MLOps & Foundation Model Lifecycle** | **Zero-Downtime Embedding Upgrades (`arbor.Bridge`)**<br>• **Encoder Swapping Without Historical Data:** Upgrades 5,000 tenant classifiers trained over 12 months on `all-MiniLM-L6-v2` (384-D) to `bge-base-en-v1.5` (768-D) or `text-embedding-3-small` (512-D) with **zero stored customer training samples**. Uses public anchor pairs to realign synaptic weights in `< 20 ms` with **93.02%** relative retention.<br>• **Cross-Modal Anchor Alignment:** Bridges a classifier trained on text embeddings so it can be queried directly by aligned image embeddings. | **Replacing Nightly LoRA & Experience Replay Pipelines (`plexus.Network`)**<br>• **Eliminating GPU Fine-Tuning Queues:** Replaces nightly PyTorch LoRA batch jobs and persistent vector replay buffers on frozen causal LLMs (`Qwen-2.5`, `Llama-3`) with continuous, single-pass CPU updates (`< 0.18 ms/step`) that prevent catastrophic forgetting (**74.77% vs 12.40%** Naive SGD on 150-class continual intent).<br>• **Drift-Immune Feature Extraction:** Maintains **84.84%** continual accuracy across 10 sequential task shifts on `DINOv2 Split CIFAR-100` with zero replay storage. |
| **Embedded IoT, Robotics & Bare-Metal Edge MCUs** | **Ultra-Compact Linear MCU Readout (`arbor.Edge` in `< 64 KB` SRAM)**<br>• **Smart Factory Acoustic & Vibration Fault Tagging:** Runs ANSI C99 `arbor_step()` on an `ARM Cortex-M4` or `ESP32` attached to a fixed FFT/DSP pipeline to learn normal vs. bearing-wear signatures directly on the machine via a physical "Teach" button.<br>• **Wearable Biosignal Calibration:** On-device ECG/EMG gesture classification that adapts to a specific user's electrode placement in `< 0.05 ms` with `0 malloc` calls and zero cloud connectivity. | **Backbone-Free Sensor Microcircuit & Occlusion Recovery (`plexus.Edge` in `101–429 KB` SRAM)**<br>• **Autonomous Drone & Rover Terrain Adaptation:** Learns directly from raw 2D/1D LiDAR, ToF, or optical sensor windows on a microcontroller **with zero neural backbone** (**90.67%** on raw Split-MNIST in `101 KB` SRAM).<br>• **Corrupted & Occluded Sensor Reconstruction (`associative_recall()`):** Recovers full control states when dust, glare, or packet loss occludes up to 50% of a camera or sensor array on an embedded robot.<br>• **Air-Gapped Industrial Anomaly Detection:** Learns non-linear multi-sensor correlations (pressure + temperature + current) on bare-metal `RISC-V` / `Cortex-M7` controllers. |
| **Cybersecurity, SOC & Threat Intelligence** | **Line-Rate Signature & Phishing Triage (`arbor.Classifier` / `Unlearn`)**<br>• **Zero-Day Phishing & URL Classification:** Classifies vectorized email and DOM embeddings at line rate (`< 0.10 ms`) and adapts immediately when SOC analysts flag a new phishing campaign.<br>• **False-Positive Quarantine Erasure (`arbor.Unlearn`):** Deterministically unlearns a benign partner domain accidentally trained into a blocklist class in `< 1 ms` without rolling back subsequent threat intelligence updates. | **Behavioral Telemetry & Prompt-Injection Firewall (`plexus.Network` / `Edge`)**<br>• **Adaptive LLM Jailbreak & Prompt-Injection Gate:** Sits in front of enterprise LLM endpoints and adapts to novel adversarial jailbreak activation patterns in real time without storing sensitive user prompts.<br>• **Encrypted Network Flow Anomaly Detection:** Multi-layer continual classification over raw packet-timing and window tensors to detect C2 (Command & Control) beaconing drift on edge network appliances. |
| **Media, Gaming & Real-Time Personalization** | **Zero-History Session Personalization (`arbor.Classifier` / `MemoryRouter`)**<br>• **Zero-Retention Feed Ranking:** Updates a user's interest vector readout in real time as they skip or watch clips while guaranteeing `persisted_bytes_delta == 0` (zero behavioral history database to breach or subpoena).<br>• **Live Moderation Queue Triage:** Fast continual classification of user-generated images and comments downstream of `CLIP` as new community guideline categories emerge. | **Adaptive NPCs & Non-Linear Player Modeling (`plexus.Network` / `Edge`)**<br>• **Continual Combat & Strategy NPCs:** Game AI agents that learn a player's non-linear tactical habits (flanking patterns + cooldown timing) within 3–5 encounters in `< 0.18 ms` on CPU without frame-rate drops.<br>• **Associative World-State Memory:** Recalls complex multi-entity game events from partial contextual cues in fixed memory. |

---

### 3. Summary Selection Guide

- **Apical Arbor:** Choose **Apical Arbor** when you already have a **contrastive embedding model** (`BGE`, `MiniLM`, `OpenAI`, `DINOv2`, `CLIP`) or a **DSP feature pipeline**, and you need the fastest possible (`< 0.10 ms`) continual readout classifier, deterministic 1-step GDPR Art. 17 class unlearning (`erase_class`), or zero-downtime embedding model migration (`bridge`).
- **Apical Plexus:** Choose **Apical Plexus** when you are building **LLM Agents & MCP workflows** (`apical-mcp`) that require **`>92%` prompt token deflation**, when you are tapping **raw causal LLM hidden layers** (`Qwen-2.5`, `Llama-3`) where features are non-linearly entangled, or when you are deploying **backbone-free sensor microcircuits** on edge hardware requiring associative pattern recall.

---

## Quickstarts: Apical Arbor (`client.arbor`)

### 1. Continual Classifier & Instant Machine Unlearning (`Classifier`)

```python
from apical import ApicalClient

# Initialize client (optionally pass privacy_secret for client-side vector blinding)
client = ApicalClient(
    api_key="apk_live_...",
    privacy_secret="my-customer-held-secret-key",
)

# Provision a fixed-byte model container (384-D input embedding, 10 output classes)
classifier = client.arbor.create_model(
    model_id="support-ticket-triage-v1",
    in_features=384,
    num_classes=10,
)

# Single-pass online plasticity update (zero data retention: persisted_bytes_delta == 0)
embedding = [0.05] * 384
learn_res = classifier.learn(embedding, target_class=2)
print(f"Status: {learn_res.status} | Delta Bytes: {learn_res.persisted_bytes_delta}")

# Sub-millisecond inference
pred = classifier.predict(embedding)
print(f"Predicted Class: {pred.prediction} (Confidence: {pred.confidence:.4f})")

# Instant GDPR Article 17 Machine Unlearning (< 1 ms)
unlearn_res = classifier.unlearn(class_id=2)
print(f"Erased Class 2 | Status: {unlearn_res.status} | Delta Bytes: {unlearn_res.persisted_bytes_delta}")
```

---

### 2. Real-Time Dynamic Memory Router (`MemoryRouter`)

```python
from apical import ApicalClient

client = ApicalClient(api_key="apk_live_...")

# Bind to a 512-D embedding router across 64 destination rules/endpoints
router = client.arbor.memory_router(
    router_id="microservice-gateway",
    embedding_dim=512,
    num_rules=64,
)

# Route incoming query embedding to Top-3 matching rules in < 0.10 ms
query_embedding = [0.12] * 512
route_res = router.route(query_embedding, top_k=3)
for item in route_res.routes:
    print(f"Rule ID: {item.rule_id} | Affinity: {item.affinity:.4f}")

# Reinforce winning destination rule online with zero token retention
router.reinforce(query_embedding, target_rule=5)
```

---

### 3. Zero-Downtime Cross-Backbone Migration (`arbor.bridge`)

```python
# Migrate a trained classifier across embedding revisions using public anchor pairs
migration = client.arbor.bridge(
    model_id="support-ticket-triage-v1",
    public_anchors={
        "source": [[0.05] * 384 for _ in range(32)],
        "target": [[0.08] * 384 for _ in range(32)],
    },
)
print(f"Status: {migration.status} | Alignment: {migration.alignment_score:.4f} | Signature: {migration.signature[:16]}...")
```

---

## Quickstarts: Apical Plexus (`client.plexus`)

### 4. Multi-Layer Non-Linear Continual Adapter (`PlexusNetwork`)

Use `client.plexus(...)` when adapting atop entangled causal LLM hidden states (`Qwen-2.5`, `Llama-3`) or routing across non-linearly overlapping tool manifolds:

```python
from apical import ApicalClient

client = ApicalClient(api_key="apk_live_...")

# Instantiate a multi-layer PlexusNetwork adapter
plexus_net = client.plexus(
    model_id="qwen-150-intent-head",
    in_features=1536,
    num_classes=150,
    tier="standard",  # "standard", "pro", or "ultra"
)

hidden_state = [0.02] * 1536

# Continual non-linear update in constant O(1) memory
learn_res = plexus_net.learn(hidden_state, target_class=42)
print(f"Adapted class {learn_res.target_class} | Winner: {learn_res.winner_class} | Delta Bytes: {learn_res.persisted_bytes_delta}")

# Multi-layer inference
pred_res = plexus_net.predict(hidden_state)
print(f"Predicted Class: {pred_res.prediction} (Confidence: {pred_res.confidence:.4f})")

# Multi-layer Top-K semantic tool routing
route_res = plexus_net.route(hidden_state, top_k=5)
print(f"Top-K Rules: {route_res.top_k_rules} | Scores: {route_res.scores}")

# Atomic class erasure on multi-layer container
unlearn_res = plexus_net.unlearn(class_id=42)
print(f"Erased Class: {unlearn_res.erased_class_id} | Delta Bytes: {unlearn_res.persisted_bytes_delta}")
```

---

### 5. Backbone-Free Bare-Metal Edge Microcircuit (`EdgeMicrocircuit`)

Use `client.plexus.edge(...)` for raw 2D/1D sensor streams and top-down associative pattern completion under spatial occlusion:

```python
import numpy as np
from apical import ApicalClient

client = ApicalClient(api_key="apk_live_...")

# Initialize standalone edge microcircuit (101 KB SRAM profile, zero neural backbone)
edge = client.plexus.edge(
    input_shape=(28, 28),
    num_classes=10,
    sram_budget_kb=101,
    circuit_id="rover-terrain-sensor",
)

sensor_frame = np.random.rand(28, 28).astype(np.float32)

# Learn directly from raw sensor window in constant memory
edge.learn(sensor_frame, target_class=3)

# Reconstruct 50% occluded sensor frame via top-down associative recall
occluded_frame = sensor_frame.copy()
occluded_frame[:14, :] = 0.0  # Simulate 50% spatial sensor occlusion
recall_res = edge.associative_recall(occluded_frame)
print(f"Associative Recovery Gain: +{recall_res.recovery_gain_pct:.2f} pp in {recall_res.latency_ms:.3f} ms")
```

---

### 6. Continual Developer Preference Adapter (`DeveloperPreferenceAdapter`)

```python
from apical import DeveloperPreferenceAdapter

adapter = DeveloperPreferenceAdapter(tenant_id="developer", model_id="preferences")

# Classify natural language prompt or code review diff against 9 architectural preference dimensions
matches = adapter.classify("Enforce declare(strict_types=1) and immutable DTO boundaries", top_k=2)
for m in matches:
    print(f"[{m.name}] Affinity={m.affinity:.4f} -> {m.directive}")

# Reinforce a specific architectural dimension from accepted code reviews
adapter.learn("Use constructor injection with max 3 dependencies", dimension_id=3)
```

---

## Universal MCP Server (`apical-mcp`) for AI Coding IDEs

Equip **Cursor**, **Claude Desktop**, **Windsurf**, **Cline / Roo-Code**, and **Google Antigravity** with sub-millisecond dynamic tool routing, **`>92%` prompt token deflation**, and zero-retention online plasticity.

### Option A: One-Command Automatic Setup (Recommended)

Automatically detects your operating system (Linux, macOS, Windows) and registers `apical-mcp` across all installed coding platforms with timestamped configuration backups:

```bash
# Auto-detect and configure installed IDEs (Cursor, Claude, Windsurf, Cline, Antigravity)
apical mcp setup

# Preview configuration diffs without writing to disk
apical mcp setup --dry-run

# Cleanly uninstall while preserving all other MCP servers
apical mcp setup --uninstall
```

### Option B: Manual IDE Configuration

Add the `plexus` entry to your IDE's MCP configuration file:

- **Cursor IDE:** `~/.cursor/mcp.json` or `.cursor/mcp.json`
- **Claude Desktop:** `~/.config/Claude/claude_desktop_config.json` (Linux) or `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
- **Windsurf / Cascade:** `~/.codeium/windsurf/mcp_config.json`
- **Google Antigravity:** `~/.gemini/antigravity/mcp_config.json`

```json
{
  "mcpServers": {
    "plexus": {
      "command": "apical-mcp",
      "args": []
    }
  }
}
```

### Real-Time Telemetry & Health CLI

```bash
# View live zero-retention telemetry dashboard (p50/p95 µs latency, tokens saved, ΔBytes == 0)
apical mcp --dashboard

# Output structured JSON runtime telemetry
apical mcp --stats

# Cryptographically wipe local telemetry log (with optional backup)
apical mcp --wipe
```

### Closed-Loop Agent Feedback (`hooks.json`)

To enable automatic background reinforcement on accepted agent tool executions without manual commands, configure lifecycle hooks in `.agents/hooks.json` or `~/.gemini/config/hooks.json`:

```json
{
  "apical-plexus-router": {
    "enabled": true,
    "PreInvocation": [
      {
        "type": "command",
        "command": "python -m apical.hooks.pre_invocation",
        "timeout": 10
      }
    ],
    "PostToolUse": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "python -m apical.hooks.post_tool_reinforce",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

---

## Offline `Ed25519` Zero-Retention Receipt Verification

Every operation returns a signed `Ed25519` cryptographic receipt certifying:
1. `persisted_bytes_delta == 0` (proof that zero raw vector or text bytes were appended to storage)
2. `input_buffer_zeroed == True` (transient RAM buffer scrubbed prior to return)
3. Canonical JSON signature verified against the enclave's `Ed25519` public key

```python
from apical import verify_zero_retention_receipt

# Verify receipt completely offline without network calls
is_valid = verify_zero_retention_receipt(learn_res.receipt)
assert is_valid is True
print("Ed25519 Zero-Retention Receipt verified offline!")
```

---

## Exported SDK Reference

| Symbol | Module | Description |
| :--- | :--- | :--- |
| `ApicalClient` (`Client`) | `apical.client` | Primary HTTP/2 connection-pooled client with `.arbor` and `.plexus` namespaces |
| `Classifier` | `apical.classifier` | Apical Arbor continual linear readout head (`.predict`, `.learn`, `.unlearn`) |
| `MemoryRouter` | `apical.router` | Apical Arbor embedding router (`.route`, `.reinforce`) |
| `PlexusNetwork` | `apical.plexus` | Apical Plexus multi-layer continual adapter & router (`.predict`, `.learn`, `.route`, `.unlearn`) |
| `EdgeMicrocircuit` | `apical.plexus` | Apical Plexus backbone-free edge microcircuit (`.predict`, `.learn`, `.associative_recall`) |
| `DeveloperPreferenceAdapter` | `apical.preference` | Continual 9-dimension developer coding preference adapter (`.classify`, `.learn`) |
| `OrthogonalPrivacyProjector` | `apical.orthogonal` | Client-side keyed isometric privacy projector (`.project`) |
| `EnclaveDirectClient` | `apical.enclave_client` | Low-latency Unix domain socket / TCP JSON-RPC client for local enclave workers |
| `DynamicToolRegistry` | `apical.catalog` | Auto-discovering MCP tool schema registry and prompt token savings calculator |
| `ZeroRetentionTelemetryLogger` | `apical.telemetry` | Append-only SHA-256 blinded telemetry logger (`.record`, `.summary`, `.wipe`) |
| `verify_zero_retention_receipt` | `apical.verifier` | Offline `Ed25519` signature and `persisted_bytes_delta == 0` invariant verifier |

---

## License

Apache-2.0. Copyright &copy; 2026 Apical Labs Ltd. All rights reserved.
