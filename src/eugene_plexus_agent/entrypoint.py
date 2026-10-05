"""Opt-in container HTTPS ingress. Caddy owns HTTP/TLS; destinations are local.

Configuration is startup-only. A file that exists and is invalid stops the
agent; a variable naming a file that does not exist falls back to the direct
ports with a warning (`resolve`). No URL or forwarding header from a request can
select an upstream. See specs/docs/design/single-port-entrypoint.md.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import re
import secrets
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import urlsplit

import certifi
import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ._http import internal_client
from ._private_files import write_private

log = logging.getLogger(__name__)
TOKEN_HEADER = "x-eugene-entry-token"
CLIENT_HEADER = "x-eugene-entry-client"
STRIP_HEADERS = [
    "Forwarded",
    "X-Forwarded-For",
    "X-Forwarded-Host",
    "X-Forwarded-Port",
    "X-Forwarded-Prefix",
    "X-Forwarded-Proto",
    "X-Real-IP",
    "X-Eugene-Plexus-Peer",
    "X-Eugene-Plexus-Forwarded-Host",
    "X-Eugene-Plexus-Forwarded-Proto",
    "X-Eugene-Plexus-Forwarded-For",
    TOKEN_HEADER,
    CLIENT_HEADER,
]
_HOST = re.compile(r"(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z][a-z0-9-]{0,62}$")
_INFERENCE_PATHS = [
    "/v1/models",
    "/v1/models/*",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/responses",
    "/v1/responses/*",
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/rerank",
    "/v1/moderations",
    "/v1/audio/speech",
    "/v1/audio/transcriptions",
    "/v1/audio/translations",
    "/v1/images/generations",
    "/v1/images/edits",
    "/v1/images/variations",
    "/v1/videos",
    "/v1/videos/*",
    "/v1/systemone",
]


class Service(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    origin: str
    networks: list[str] = Field(min_length=1)

    @field_validator("origin")
    @classmethod
    def valid_origin(cls, value: str) -> str:
        if any(char.isspace() for char in value):
            raise ValueError("origin cannot contain whitespace")
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
            or not _HOST.fullmatch(parsed.hostname or "")
            or parsed.netloc != parsed.netloc.lower()
            or "\\" in value
        ):
            raise ValueError("use an exact lowercase HTTPS hostname origin, without a path")
        port = parsed.port if parsed.port is not None else 443
        if port == 0:
            raise ValueError("origin must name a nonzero port")
        return f"https://{parsed.hostname}" + (f":{port}" if port != 443 else "")

    @field_validator("networks")
    @classmethod
    def valid_networks(cls, values: list[str]) -> list[str]:
        return [str(ipaddress.ip_network(value, strict=True)) for value in values]

    @property
    def host(self) -> str:
        return str(urlsplit(self.origin).hostname)


class AutomaticCertificates(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    email: str
    accept_terms: bool
    staging: bool = False

    @field_validator("email")
    @classmethod
    def valid_email(cls, value: str) -> str:
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
            raise ValueError("provide a contact email for certificate management")
        return value


class TrustedProxy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    addresses: list[str] = Field(min_length=1, max_length=32)
    transport: Literal["http", "https"] = "http"

    @field_validator("addresses")
    @classmethod
    def exact_addresses(cls, values: list[str]) -> list[str]:
        result = []
        for value in values:
            network = ipaddress.ip_network(value, strict=True)
            if network.num_addresses != 1:
                raise ValueError("trust individual proxy IPs, not an entire network")
            address = network.network_address
            if address.is_unspecified or address.is_multicast:
                raise ValueError("use a unicast proxy address")
            result.append(str(network))
        return list(dict.fromkeys(result))


class EntryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    listen_port: int = Field(default=8443, ge=1024, le=65535)
    console: Service
    workbench: Service
    inference: Service | None = None
    nodes: Service | None = None
    certificate: Path | None = None
    private_key: Path | None = None
    trusted_ca: Path | None = None
    internal_ca: bool = False
    acme: AutomaticCertificates | None = None
    proxy: TrustedProxy | None = None

    @property
    def private_http(self) -> bool:
        return self.proxy is not None and self.proxy.transport == "http"

    @model_validator(mode="after")
    def distinct_services(self) -> EntryConfig:
        services = self.services()
        if len({s.host for s in services}) != len(services):
            raise ValueError("each service needs a different hostname")
        if len({urlsplit(s.origin).port or 443 for s in services}) != 1:
            raise ValueError("all public origins must share one HTTPS port")
        for service in (self.console, self.nodes):
            if service and any(ipaddress.ip_network(n).prefixlen == 0 for n in service.networks):
                raise ValueError("console and node administration require specific source networks")
        if self.private_http:
            if self.internal_ca or self.certificate or self.private_key or self.acme:
                raise ValueError("private HTTP uses the trusted proxy's certificates only")
        elif self.acme:
            if (
                self.proxy
                or self.internal_ca
                or self.certificate
                or self.private_key
                or self.trusted_ca
            ):
                raise ValueError("automatic public certificates require direct HTTPS mode")
            if not self.acme.accept_terms:
                raise ValueError("accept the Let's Encrypt subscriber agreement to enable ACME")
            for service in services:
                if urlsplit(service.origin).port not in (None, 443):
                    raise ValueError("automatic certificates require public HTTPS port 443")
                if service.host.endswith(
                    (
                        ".local",
                        ".internal",
                        ".home.arpa",
                        ".localhost",
                        ".test",
                        ".invalid",
                        ".example",
                    )
                ):
                    raise ValueError(
                        "automatic public certificates need names in a domain you control"
                    )
        elif self.internal_ca:
            if self.certificate or self.private_key:
                raise ValueError("choose either internal_ca or certificate/private_key")
        elif not self.certificate or not self.private_key:
            raise ValueError(
                "choose automatic certificates, a trusted HTTP proxy, "
                "internal_ca, or certificate/private_key"
            )
        for item in (self.certificate, self.private_key, self.trusted_ca):
            if item and not (item.is_absolute() or PurePosixPath(item.as_posix()).is_absolute()):
                raise ValueError("certificate paths must be absolute container paths")
        return self

    def services(self) -> list[Service]:
        return [s for s in (self.console, self.workbench, self.inference, self.nodes) if s]

    def public_urls(self) -> dict[str, str]:
        result = {"consoleUrl": self.console.origin, "workbenchUrl": self.workbench.origin}
        if self.inference:
            result["inferenceUrl"] = self.inference.origin
        if self.nodes:
            result["nodesUrl"] = self.nodes.origin
        return result

    @classmethod
    def load(cls, path: Path) -> EntryConfig:
        config = cls.model_validate_json(path.read_text(encoding="utf-8"))
        for item in (config.certificate, config.private_key, config.trusted_ca):
            if item and (not item.is_absolute() or not item.is_file()):
                raise ValueError(f"TLS file must exist at an absolute path: {item}")
        return config


ENV_VARIABLE = "EUGENE_PLEXUS_AGENT_ENTRYPOINT_CONFIG"


class EntryPointConfigError(ValueError):
    """The configuration file exists and cannot be used. Stops the agent."""


def resolve(settings: Any) -> EntryConfig | None:
    """The configuration to run with, and what a missing file means.

    **A missing file falls back to the direct ports** (Troy, 2026-10-04,
    on the first live NAS migration). The variable set and the file absent
    crashed the agent with a traceback on every restart, which left the
    owner with no console to fix it from. Now the variable is cleared for
    this process, the reason is kept on `settings._entrypoint_fallback` for
    the log and `/healthz`, and the install runs exactly as it did before
    the variable was set. Nothing is opened that the old mode did not open:
    the direct ports are only reachable where the owner still publishes them.

    **A file that exists and cannot be used still stops the agent**, with one
    sentence instead of a traceback. Someone is part-way through configuring
    it, and quietly running the old mode would hide a mistake in a file that
    decides who can reach the console.
    """
    path = settings.entrypoint_config
    if path is None:
        return None
    try:
        return EntryConfig.load(path)
    except FileNotFoundError:
        settings.entrypoint_config = None
        settings._entrypoint_fallback = (
            f"{ENV_VARIABLE} names {path}, which does not exist, so the HTTPS entry "
            "point is off and Eugene is serving on its direct ports, as it did before "
            "the variable was set. To use one HTTPS port, save the configuration from "
            f"Settings, Container access setup, as {path} and restart the container. "
            "To stay on the direct ports, remove the variable."
        )
        return None
    except ValidationError as exc:
        reason = "; ".join(
            (".".join(str(part) for part in error["loc"]) + ": " if error["loc"] else "")
            + error["msg"]
            for error in exc.errors(include_input=False, include_context=False)
        )
    except (OSError, ValueError) as exc:
        reason = str(exc)
    raise EntryPointConfigError(
        f"the HTTPS entry point configuration at {path} cannot be used: {reason}. "
        "Eugene does not start with a broken file, so it never opens ports you meant "
        "to close. Fix the file (Settings, Container access setup prepares a valid one) "
        f"and restart the container, or remove {ENV_VARIABLE} to go back to the direct "
        "ports."
    )


def caddy_config(
    config: EntryConfig, directory: Path, token: str, ports: dict[str, int | None]
) -> dict[str, Any]:
    """Generate structured configuration; never interpolate a Caddyfile or shell."""
    routes: list[dict[str, Any]] = []

    if config.proxy:
        # A trusted hop is not itself a trusted user. In particular, missing
        # or invalid XFF must not fall back to the proxy's privileged LAN IP.
        for refused in (
            {"not": [{"remote_ip": {"ranges": config.proxy.addresses}}]},
            {"client_ip": {"ranges": config.proxy.addresses}},
            {"not": [{"expression": "{http.request.header.X-Forwarded-Proto} == 'https'"}]},
        ):
            routes.append(
                {
                    "match": [refused],
                    "handle": [
                        {
                            "handler": "static_response",
                            "status_code": 403,
                            "body": "A trusted HTTPS proxy and verified client IP are required.",
                        }
                    ],
                    "terminal": True,
                }
            )

    def route(service: Service, target: str, paths: list[str] | None = None) -> None:
        match: dict[str, Any] = {
            "host": [service.host],
            "client_ip" if config.proxy else "remote_ip": {"ranges": service.networks},
        }
        if paths:
            match["path"] = paths
        port = ports.get(target)
        handlers: list[dict[str, Any]]
        if port is None:
            handlers = [
                {
                    "handler": "static_response",
                    "status_code": 503,
                    "body": "This service is unavailable. Open Eugene to update or start it.",
                }
            ]
        else:
            if not 1 <= port <= 65535 or port == config.listen_port:
                raise ValueError("invalid local upstream port")
            headers = {"Host": [urlsplit(service.origin).netloc]}
            if target == "agent":
                headers[TOKEN_HEADER] = [token]
                headers[CLIENT_HEADER] = [
                    "{http.vars.client_ip}" if config.proxy else "{http.request.remote.host}"
                ]
            # Caddy applies delete after set. Metadata we replace must not
            # also be deleted; Set replaces every caller-supplied value.
            remove = [name for name in STRIP_HEADERS if name not in headers]
            handlers = [
                {"handler": "request_body", "max_size": 32 * 1024 * 1024},
                {
                    "handler": "reverse_proxy",
                    "upstreams": [{"dial": f"127.0.0.1:{port}"}],
                    "headers": {"request": {"delete": remove, "set": headers}},
                    "transport": {"protocol": "http", "dial_timeout": "5s"},
                    "flush_interval": -1,
                },
            ]
        routes.append({"match": [match], "handle": handlers, "terminal": True})

    # Public sign-in does not grant access to the adjacent console/API routes.
    sign_in = config.console.model_copy(update={"networks": config.workbench.networks})
    route(sign_in, "agent", ["/oidc/*"])
    route(config.console, "agent")
    route(config.workbench, "workbench")
    if config.inference:
        route(config.inference, "gateway", _INFERENCE_PATHS)
    if config.nodes:
        route(config.nodes, "control")
    # Refuse known but disallowed hosts/paths; never a catch-all upstream.
    routes.append(
        {
            "match": [{"host": [s.host for s in config.services()]}],
            "handle": [{"handler": "static_response", "status_code": 403}],
        }
    )
    routes.append({"handle": [{"handler": "static_response", "status_code": 421}]})
    hosts = [s.host for s in config.services()]
    tls: dict[str, Any] = {}
    if config.internal_ca:
        tls["automation"] = {"policies": [{"subjects": hosts, "issuers": [{"module": "internal"}]}]}
    elif config.acme:
        authority = (
            "https://acme-staging-v02.api.letsencrypt.org/directory"
            if config.acme.staging
            else "https://acme-v02.api.letsencrypt.org/directory"
        )
        tls["automation"] = {
            "policies": [
                {
                    "subjects": hosts,
                    "issuers": [
                        {
                            "module": "acme",
                            "ca": authority,
                            "email": config.acme.email,
                            "challenges": {
                                "http": {"disabled": True},
                                "tls-alpn": {"alternate_port": config.listen_port},
                            },
                        }
                    ],
                }
            ]
        }
    elif not config.private_http:
        tls["certificates"] = {
            "load_files": [
                {
                    "certificate": str(config.certificate),
                    "key": str(config.private_key),
                }
            ]
        }
    document: dict[str, Any] = {
        "admin": {"listen": "unix/" + str(directory / "admin.sock"), "config": {"persist": False}},
        "storage": {"module": "file_system", "root": str(directory / "tls")},
        # HTTP error logs can include request URIs (OIDC codes/state). Keep
        # proxy lifecycle/TLS errors, without recording browser requests.
        "logging": {
            "logs": {
                "default": {
                    "level": "ERROR",
                    "exclude": ["http.log.access", "http.log.error"],
                }
            }
        },
        "apps": {
            "pki": {"certificate_authorities": {"local": {"install_trust": False}}},
            "tls": tls,
            "http": {
                "servers": {
                    "entry": {
                        "listen": [f":{config.listen_port}"],
                        "protocols": ["h1", "h2"],
                        "strict_sni_host": True,
                        "read_header_timeout": "10s",
                        "read_timeout": "60s",
                        "idle_timeout": "2m",
                        "max_header_bytes": 32768,
                        "automatic_https": {
                            "disable_redirects": True,
                            "disable_certificates": not (config.internal_ca or config.acme),
                        },
                        "tls_connection_policies": [{"match": {"sni": hosts}}],
                        "routes": routes,
                    }
                }
            },
        },
    }
    server = document["apps"]["http"]["servers"]["entry"]
    if config.proxy:
        server.update(
            {
                "trusted_proxies": {"source": "static", "ranges": config.proxy.addresses},
                "trusted_proxies_strict": 1,
                "client_ip_headers": ["X-Forwarded-For"],
            }
        )
    if config.private_http:
        server.pop("tls_connection_policies")
        server.pop("strict_sni_host")
        server["automatic_https"] = {"disable": True}
        server["protocols"] = ["h1"]
        document["apps"].pop("tls")
    if not config.internal_ca:
        document["apps"].pop("pki")
    return document


class ProxyMetadata:
    """Only this process's local proxy may supply client IP / HTTPS metadata."""

    def __init__(self, app: Any, *, config: EntryConfig, token: str) -> None:
        self.app, self.config, self.token = app, config, token

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] in ("http", "websocket"):
            headers = scope.get("headers", [])
            supplied = [v for k, v in headers if k.lower() == TOKEN_HEADER.encode()]
            clients = [v for k, v in headers if k.lower() == CLIENT_HEADER.encode()]
            peer = scope.get("client")
            trusted = False
            try:
                trusted = bool(
                    peer
                    and ipaddress.ip_address(peer[0]).is_loopback
                    and len(supplied) == 1
                    and len(clients) == 1
                    and hmac.compare_digest(supplied[0], self.token.encode())
                )
                if trusted:
                    address = str(ipaddress.ip_address(clients[0].decode("ascii")))
                    scope = dict(scope, scheme="https", client=(address, 0))
            except (ValueError, UnicodeError):
                trusted = False
            if supplied and not trusted:
                from starlette.responses import PlainTextResponse

                await PlainTextResponse("Invalid proxy metadata", status_code=403)(
                    scope, receive, send
                )
                return
            stripped = {name.lower().encode() for name in STRIP_HEADERS}
            scope = dict(scope, headers=[(k, v) for k, v in headers if k.lower() not in stripped])
        await self.app(scope, receive, send)


class EntryPoint:
    def __init__(self, app: Any, config: EntryConfig, directory: Path, token: str) -> None:
        self.app, self.config, self.directory, self.token = app, config, directory, token
        self.process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._admin: httpx.AsyncClient | None = None
        self._applied: dict[str, int | None] | None = None
        self._applied_certificates: str | None = None
        self._blocked_workbench = False
        if app.state.apps:
            app.state.apps.before_stop = self.before_stop
            app.state.apps.after_start = self.after_start

    def after_start(self, app_id: str) -> None:
        if app_id == "workbench":
            self._blocked_workbench = False

    async def before_stop(self, app_id: str) -> None:
        if app_id != "workbench":
            return
        # Remove the public route BEFORE replacing/removing the process. In
        # particular, rollback to an older app must never inherit its route
        # for even the short interval before the next compatibility probe.
        self._blocked_workbench = True
        async with self._lock:
            if self.process and self.process.returncode is None and self._admin:
                try:
                    current = self.ports()
                    answer = await self._admin.post(
                        "http://localhost/load",
                        json=caddy_config(
                            self.config,
                            self.directory,
                            self.token,
                            current,
                        ),
                    )
                    answer.raise_for_status()
                    self._applied = current
                except httpx.HTTPError:
                    log.error(
                        "Could not remove the app route; stopping the proxy "
                        "before replacing the app"
                    )
                    await self._stop()

    async def _stop(self) -> None:
        if self.process and self.process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=10)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        self.process = None
        self._applied = None

    def ports(self) -> dict[str, int | None]:
        state = self.app.state.agent_state
        result: dict[str, int | None] = {
            "agent": int(self.app.state.settings.bind_port),
            "workbench": None,
            "control": None,
            "gateway": None,
        }
        for component in state.list_components():
            if component.spawn is not None and component.kind.value in ("control", "gateway"):
                # The topology chooses a port, never a destination host for this proxy.
                result[component.kind.value] = urlsplit(str(component.url)).port
        manager = self.app.state.apps
        if manager and not self._blocked_workbench:
            record = manager.store.get("workbench")
            if record and record.enabled and manager.supervisor.is_running("workbench"):
                result["workbench"] = record.port
        return result

    async def _sync(self, client: httpx.AsyncClient, backend: httpx.AsyncClient) -> None:
        from .child_env import child_environment

        current = self.ports()
        certificates = await asyncio.to_thread(self.certificate_fingerprint)
        if current["workbench"] is not None:
            # Older builds cannot enforce HTTPS origin isolation. Probe the
            # new process before publishing it after an update or rollback.
            try:
                health = await backend.get(f"http://127.0.0.1:{current['workbench']}/healthz")
                report = health.json() if health.status_code == 200 else {}
            except (httpx.HTTPError, ValueError):
                report = {}
            if not isinstance(report, dict):
                report = {}
            if (
                report.get("publicOrigin") != self.config.workbench.origin
                or report.get("originIsolation") != 1
            ):
                current["workbench"] = None
        document = caddy_config(self.config, self.directory, self.token, current)
        if self.process is None or self.process.returncode is not None:
            path = self.directory / "caddy.json"
            write_private(path, json.dumps(document))
            self.process = await asyncio.create_subprocess_exec(
                self.app.state.settings.entrypoint_binary,
                "run",
                "--config",
                str(path),
                env=child_environment(),
            )
            self._applied = current
            self._applied_certificates = certificates
            log.info("HTTPS entry point starting on port %d", self.config.listen_port)
        elif self._applied != current or self._applied_certificates != certificates:
            answer = await client.post(
                "http://localhost/load",
                json=document,
                headers={"Cache-Control": "must-revalidate"},
            )
            answer.raise_for_status()
            self._applied = current
            self._applied_certificates = certificates
            log.info("HTTPS entry point updated its local service routes")

    def certificate_fingerprint(self) -> str | None:
        if not self.config.certificate or not self.config.private_key:
            return None
        digest = hashlib.sha256()
        for path in (self.config.certificate, self.config.private_key):
            digest.update(path.read_bytes())
        return digest.hexdigest()

    async def run(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.chmod(0o700)
        path = self.directory / "caddy.json"
        transport = httpx.AsyncHTTPTransport(uds=str(self.directory / "admin.sock"))
        async with (
            internal_client(transport=transport, timeout=5) as client,
            internal_client(timeout=2) as backend,
        ):
            self._admin = client
            try:
                while True:
                    try:
                        async with self._lock:
                            await self._sync(client, backend)
                    except (OSError, ValueError, httpx.HTTPError):
                        log.exception("HTTPS entry point unavailable; backend ports remain private")
                    await asyncio.sleep(2)
            finally:
                await self._stop()
                self._admin = None
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()


def install(app: Any) -> None:
    settings = app.state.settings
    named = settings.entrypoint_config
    config = resolve(settings)
    if named is not None and config is None:
        # `build_server` normally resolves first and says this itself; an app
        # built without it (the Windows service, tests) says it here.
        log.warning("%s", settings._entrypoint_fallback)
    app.state.entrypoint_config = config
    app.state.entrypoint_token = secrets.token_urlsafe(32) if config else None
    if config:
        settings._entrypoint_console_origin = config.console.origin
        if os.name != "posix":
            raise ValueError("the managed HTTPS entry point is supported in Linux containers")
        # Provision the local CA before children construct their cached SSL
        # contexts. Caddy validate provisions config without opening listeners.
        # All artifacts stay in this installation; no OS trust is modified.
        from ._http import egress_ssl_context, ssl_context
        from .child_env import child_environment

        directory = settings.config_file.resolve().parent / "entrypoint"
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
        path = directory / "caddy.json"
        write_private(
            path,
            json.dumps(
                caddy_config(
                    config,
                    directory,
                    app.state.entrypoint_token,
                    {"agent": settings.bind_port},
                )
            ),
        )
        checked = subprocess.run(
            [settings.entrypoint_binary, "validate", "--config", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
            env=child_environment(),
        )
        if checked.returncode:
            raise ValueError("HTTPS entry point configuration failed: " + checked.stderr[-2000:])
        ca = (
            (directory / "tls/pki/authorities/local/root.crt")
            if config.internal_ca
            else config.trusted_ca
        )
        if ca:
            public_roots = Path(certifi.where()).read_bytes()
            existing = os.environ.get("SSL_CERT_FILE")
            bundle = directory / "ca-bundle.pem"
            if existing and Path(existing).resolve() != bundle.resolve():
                public_roots += b"\n" + Path(existing).read_bytes()
            write_private(bundle, public_roots + b"\n" + ca.read_bytes())
            os.environ["SSL_CERT_FILE"] = str(bundle)
            ssl_context().load_verify_locations(cafile=str(bundle))
            egress_ssl_context().load_verify_locations(cafile=str(bundle))
        app.add_middleware(ProxyMetadata, config=config, token=app.state.entrypoint_token)
