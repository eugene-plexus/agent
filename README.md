# Eugene Plexus - `agent`

The per-host node agent for [Eugene Plexus](https://github.com/eugene-plexus/specs#readme),
a self-hosted control plane for local LLM inference. One agent runs on each host.
It supervises processes; it does not perform inference or route model requests.

## Responsibilities

- **Component supervision:** starts and monitors `gateway`, `inference-driver`,
	`library`, and `control` processes. Safe mode keeps configuration repair reachable.
- **Engine lifecycle:** constructs argv, probes readiness, and captures logs for
	upstream llama.cpp and user-installed vLLM. A live engine loading weights is
	reported as `loading`, not confused with a crashed process.
- **Engine acquisition:** downloads and verifies supported llama.cpp binaries.
	vLLM installation is operator-managed; MLX has no adapter yet.
- **Runtime admission:** estimates memory demand against live hardware and refuses
	an oversized launch with the arithmetic. A dry run and explicit force override
	are available; unknown capacity does not refuse a launch.
- **Companion drivers:** declares an inference-driver for each runtime so a launch
	can become routable when the runtime is ready.
- **Node identity and enrollment:** adopts the install's signing key, reports local
	runtimes and devices, advertises peer-reachable addresses, and accepts signed
	rekeying from the control root with epoch fencing.
- **UI hosting:** currently serves configured UI assets. Moving UI ownership to
	the control root remains undecided; the first-run wizard still needs a rewrite.

The [`control`](https://github.com/eugene-plexus/control) component owns install-wide
trust, topology and replicated control state. The agent supervises it but does not
inject the normal child auth credentials into it. The
[`gateway`](https://github.com/eugene-plexus/gateway) decides idle unload and wake on
demand; the owning agent executes those actions. This data path survives a control
root outage.

## API Overview

| Surface                                           | Purpose                                                    |
| ------------------------------------------------- | ---------------------------------------------------------- |
| `/v1/components`                                  | Local component declarations and status                    |
| `/v1/runtimes`                                    | Engine runtime declarations, status and lifecycle actions  |
| `POST /v1/runtimes/admission`                     | Memory admission dry run                                   |
| `/v1/engines`                                     | Adapter capabilities, flag schemas and engine installation |
| `GET /v1/node`                                    | Node identity, enrollment state and devices                |
| `POST /v1/node/enroll`                            | Enroll this host with the control root                     |
| `POST /v1/node/rekey`                             | Control-identity-signed key/epoch update, not bearer auth  |
| `GET`/`PATCH /v1/config`, `GET /v1/config/schema` | Configuration and UI metadata                              |
| `/v1/auth/*`                                      | Agent authentication/bootstrap surface                     |
| `GET /healthz`                                    | Liveness and degraded-mode signal                          |

The full contract is
[`specs/openapi/agent.yaml`](https://github.com/eugene-plexus/specs/blob/main/openapi/agent.yaml).
The agent itself is not a `ComponentKind`; engines are runtimes, not components.

## Running From Source

Use Python 3.12 to match CI:

```bash
pip install -e ".[dev]"
python -m eugene_plexus_agent
```

The default port is **8079**, configurable with `EUGENE_PLEXUS_AGENT_BIND_PORT`.
`EUGENE_PLEXUS_AGENT_CONFIG_FILE` selects the topology/config file. Install the
Python components the agent will supervise into **the agent's environment**: its
children run with its own interpreter, not each sibling repo's virtualenv.
Engine binaries are separate upstream installations.

Initialize authentication before network exposure. For a multi-host install,
enroll every node, including the control host, and configure reachable advertised
URLs over a Tailscale/WireGuard network. Enrollment replaces the local signing key
and invalidates the session that requested it. The install signing key is stored
locally in `node.yaml`; protect that file and never commit runtime configs or keys.
See the [M7 design](https://github.com/eugene-plexus/specs/blob/main/docs/design/m7-second-host-readiness.md)
for enrollment and rekey semantics.

## Verification Status

As of **2026-09-10**, llama.cpp lifecycle and admission passed the M6 live run;
enrollment, advertised addressing, and signed rekeying passed M7 with two agents
on one Windows host. A real two-machine run, a vLLM launch, two-GPU placement,
and AMD/Intel/Apple hardware detection remain unverified. M7 also exposed a short
post-unload routing window in the gateway that remains open.

The [project overview](https://github.com/eugene-plexus/specs#current-status) links
the acceptance records and current limitations. The former consciousness and
training architectures are retired, not the purpose of this agent.

## Codegen

Pydantic models for the agent and shared schemas are generated from the pinned `eugene-plexus/specs` commit:

```bash
python scripts/codegen.py
```

`SPECS_REF` records the commit SHA. Bump it to track a newer specs release; CI re-runs codegen and fails if the working tree drifts.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`CONTRIBUTING.md`](CONTRIBUTING.md) (DCO sign-off required).
