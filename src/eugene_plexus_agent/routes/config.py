"""Standard config trio for the agent: UI prefs + firstRunComplete only.

Topology lives under /v1/components, deliberately NOT here — editing UI
prefs in the generic config editor must never accidentally restructure
the install. See `state.py` for the rationale.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from typing import Any

from fastapi import APIRouter, Request

from .. import enrollment, keyring_store, library_folders, model_paths, share_credentials
from .._generated.common_models import (
    ConfigDocument,
    ConfigField,
    ConfigFieldError,
    ConfigFieldStatus,
    ConfigFieldStatusLevel,
    ConfigSchema,
    ConfigTestRequest,
    ConfigTestResult,
    ConfigUpdateRequest,
    ConfigUpdateResult,
)
from .._generated.models import LibraryFolderCheckRequest, LibraryFolderReach
from ..admission import model_size_bytes
from ..state import AgentState
from .runtimes import folder_cache_for, library_client_for, refresh_library_folders

log = logging.getLogger(__name__)

router = APIRouter(tags=["config"])


@router.get("/v1/config", response_model=ConfigDocument)
async def get_config(request: Request) -> ConfigDocument:
    state: AgentState = request.app.state.agent_state
    document = state.as_config_document()
    # **The one field here that holds a secret.** Redaction is per entry
    # rather than per field: host and user name are the half a person
    # edits, so a row has to come back renderable, and a UI has to be
    # able to tell *a password is stored* from *there is none*. The merge
    # in `_seal_share_credentials` is what makes writing the redacted row
    # back safe.
    stored = getattr(document, SHARE_CREDENTIALS_KEY, None)
    if stored:
        setattr(document, SHARE_CREDENTIALS_KEY, share_credentials.redact_entries(stored))
    return document


@router.get("/v1/config/schema", response_model=ConfigSchema)
async def get_config_schema(request: Request) -> ConfigSchema:
    state: AgentState = request.app.state.agent_state
    schema = state.as_config_schema()
    keyring: bool | None = None
    if state.get_config("securityMode") == "os_keyring":
        # Memoised per process, but a locked Secret Service can block on a
        # prompt nobody answers: the same deadline `/v1/auth/status` keeps.
        from .auth import KEYRING_PROBE_BUDGET_SECONDS

        try:
            keyring = await asyncio.wait_for(
                asyncio.to_thread(keyring_store.probe_sync), timeout=KEYRING_PROBE_BUDGET_SECONDS
            )
        except TimeoutError:
            keyring = None
    live = await asyncio.to_thread(_live_facts, request, state)
    live["keyring"] = keyring
    schema.fields = [_with_live_facts(f, live) for f in schema.fields]
    return schema


def _live_facts(request: Request, state: AgentState) -> dict[str, Any]:
    """What this machine can say, now, about what an unset or chosen value
    does. Blocking lookups (PATH, the keyring probe): off the event loop.

    **Settings never lie** (2026-09-30): an empty engine path is not
    "nothing" -- it is whatever PATH finds; an unset advertise address is
    the one derived from the route to the root; a security mode this
    machine cannot carry out is not the mode it runs.
    """
    from .. import apps, node_identity, reach
    from .node import _listeners

    settings = getattr(request.app.state, "settings", None)
    record = request.app.state.node_identity.record
    advertise = node_identity.effective_advertise_url(
        state.get_config("advertiseUrl"), record.advertise_url
    )
    try:
        bound = reach.bound_addresses(_listeners(request))
        agent_bound = next((b for b in bound if b.process == "agent"), None)
        restart = reach.restart_required(advertise_url=advertise, agent_bound=agent_bound)
    except Exception:  # pragma: no cover - a status line is never a reason to fail the schema
        restart = False
    uv = apps.find_uv(None)
    return {
        "derived_advertise": (
            node_identity.effective_advertise_url(None, record.advertise_url)
            if record.advertise_url
            else None
        ),
        "advertise_restart": restart,
        "vllm": shutil.which("vllm"),
        "mlx": shutil.which("mlx_lm.server"),
        "uv": str(uv) if uv is not None else None,
        "passphrase_path": getattr(settings, "passphrase_file", None),
        "mode": state.get_config("securityMode"),
        "copy_enabled": state.get_config("modelCopyEnabled") is True,
        "copy_dir": state.get_config("modelCopyDir"),
    }


def _with_live_facts(field: ConfigField, live: dict[str, Any]) -> ConfigField:
    update: dict[str, Any] = {}
    key = field.key
    if key == "advertiseUrl":
        derived = live["derived_advertise"]
        update["unsetMeans"] = (
            f"Derived when this machine joined the install: {derived}."
            if derived
            else "Not set, and nothing derived: other machines cannot reach this one, "
            "which is right for a single machine."
        )
        if derived:
            update["unsetResolvesTo"] = derived
        if live["advertise_restart"]:
            update["status"] = ConfigFieldStatus(
                level=ConfigFieldStatusLevel.warning,
                text=(
                    "This machine's own agent still listens only on 127.0.0.1, and a "
                    "listening socket is fixed for the life of a process: it answers on "
                    "this address after its next restart."
                ),
            )
    elif key in ("vllmBinary", "mlxBinary"):
        found = live["vllm" if key == "vllmBinary" else "mlx"]
        script = "vllm" if key == "vllmBinary" else "mlx_lm.server"
        update["unsetMeans"] = (
            f"Not set: uses the `{script}` found on PATH ({found}), unless a runtime names "
            "its own binary."
            if found
            else f"Not set, and no `{script}` is on PATH: a runtime on this engine needs its "
            "own binary until this is set."
        )
        if found:
            update["unsetResolvesTo"] = found
    elif key == "uvBinary":
        found = live["uv"]
        update["unsetMeans"] = (
            f"Not set: uses {found}, the uv beside this install or on PATH."
            if found
            else "Not set, and no uv is beside this install or on PATH: apps cannot be "
            "installed until this is set."
        )
        if found:
            update["unsetResolvesTo"] = found
    elif key == "kevPython":
        update["unsetMeans"] = (
            "Not set: Kev reads as not installed. It is never looked for on PATH, because a "
            "bare python is not evidence of a Kev environment."
        )
    elif key == "modelCopyDir":
        update["unsetMeans"] = "Not set: nothing is copied, even with copying turned on."
        if live["copy_enabled"] and not live["copy_dir"]:
            update["status"] = ConfigFieldStatus(
                level=ConfigFieldStatusLevel.warning,
                text="Copying is on, but with no folder nothing is copied.",
            )
    elif key == "securityMode":
        status = _security_mode_status(live)
        if status is not None:
            update["status"] = status
    return field.model_copy(update=update) if update else field


def _security_mode_status(live: dict[str, Any]) -> ConfigFieldStatus | None:
    mode = live["mode"]
    if mode == "os_keyring" and live["keyring"] is False:
        return ConfigFieldStatus(
            level=ConfigFieldStatusLevel.warning,
            text=(
                "This machine's keyring refused a test entry, so it cannot keep the key: "
                "it asks for the passphrase at every start, as Prompt on startup does."
            ),
        )
    if mode == "passphrase_file":
        path = live["passphrase_path"]
        if path is None:
            return ConfigFieldStatus(
                level=ConfigFieldStatusLevel.warning,
                text=(
                    "This agent has no passphrase file configured "
                    "(EUGENE_PLEXUS_AGENT_PASSPHRASE_FILE, which the Linux system install "
                    "sets), so it asks for the passphrase at every start, as Prompt on "
                    "startup does."
                ),
            )
        if not path.is_file():
            return ConfigFieldStatus(
                level=ConfigFieldStatusLevel.info,
                text=f"{path} is written at your next sign-in; until then a restart asks "
                "for the passphrase.",
            )
    return None


@router.post("/v1/config/test", response_model=ConfigTestResult)
async def test_config(
    request: Request,
    body: ConfigTestRequest | None = None,
) -> ConfigTestResult:
    """Probe the agent's effective config without committing.

    Most config fields here (UI prefs, firstRunComplete) have no
    external dependency to verify, so they always succeed. The
    interesting case is `securityMode=os_keyring` — the OS keyring
    backend varies by platform and can fail silently (Linux without
    a running secret service, Windows without Credential Manager,
    etc.). We probe a round-trip read here so the operator finds
    out at config-time rather than at the next-boot auto-unlock
    attempt.

    Body's `overrides` are honored so the operator can test a
    pending switch BEFORE saving it. Two fields are consumed:
    `securityMode`, and (M11) `pathMappings` -- each mapping's target is
    stat'd, and when the library can be reached every model it lists
    under a mapping's `from` is resolved and checked for being here and
    being the size the library says. That is the verification that
    launches nothing, and the Test button beside the mapping editor.
    """
    start = time.perf_counter()
    state: AgentState = request.app.state.agent_state
    overrides: dict[str, Any] = (
        body.overrides.model_dump() if body is not None and body.overrides is not None else {}
    )
    effective_mode = (
        overrides.get("securityMode")
        if "securityMode" in overrides
        else state.get_config("securityMode")
    )

    problems: list[str] = []
    notes: list[str] = []

    if effective_mode == "os_keyring":
        # A real round trip — write, read back, delete a throwaway
        # entry — rather than a read of whatever is there. A read
        # returning None cannot tell a working keyring with nothing
        # stored from a `fail` backend, and the old version of this
        # check could not either.
        if await asyncio.to_thread(keyring_store.probe_sync):
            notes.append(
                "OS keyring accepted a probe entry. Auto-unlock will work "
                "on next boot once the operator has logged in (the "
                "master key is persisted on login or on securityMode "
                "transition to os_keyring)."
            )
        else:
            problems.append(
                "This host's OS keyring did not accept a probe entry, so "
                "securityMode=os_keyring will not auto-unlock on the next "
                "boot. Switch to prompt_on_startup, or install / start the "
                "platform's keyring backend (Credential Manager on Windows, "
                "Secret Service on Linux, Keychain on macOS)."
            )
    elif effective_mode == "passphrase_file":
        path = getattr(getattr(request.app.state, "settings", None), "passphrase_file", None)
        if path is None:
            problems.append(
                "securityMode=passphrase_file needs EUGENE_PLEXUS_AGENT_PASSPHRASE_FILE in "
                "this agent's environment, which the Linux system install sets. Without it "
                "there is nowhere to keep the passphrase and every restart leaves this agent "
                "locked. Switch to prompt_on_startup, or set the variable where the service "
                "is defined."
            )
        elif await asyncio.to_thread(path.is_file):
            notes.append(f"The passphrase is kept in {path}; this agent unlocks itself on restart.")
        else:
            notes.append(
                f"{path} does not exist yet. Signing in writes it; until then a restart "
                "leaves this agent locked."
            )
    else:
        notes.append(
            f"securityMode={effective_mode}; no external dependency to "
            f"verify. (Switch to os_keyring to probe the OS secret "
            f"store round-trip.)"
        )

    raw_rules = (
        overrides[model_paths.CONFIG_KEY]
        if model_paths.CONFIG_KEY in overrides
        else state.get_config(model_paths.CONFIG_KEY)
    )
    # The EFFECTIVE rules (2026-09-14): the overrides under test, then
    # the Library folders' mounts they do not shadow -- so the Test
    # button on the agent's own field stays honest about what a spawn
    # would open.
    await refresh_library_folders(request)
    cache = folder_cache_for(request)
    rules = library_folders.effective_rules(
        model_paths.parse_rules(raw_rules),
        cache.inherited_rules() if cache is not None else (),
    )
    if rules:
        library = await library_client_for(request)
        models = await library.list_models() if library is not None else None
        # Real filesystem calls, off the event loop: a mapping whose
        # target is a dead share blocks for as long as the OS allows.
        checks = await asyncio.to_thread(
            model_paths.check_rules, rules, models, size_of=model_size_bytes
        )
        _ok, summary, error = model_paths.describe_checks(
            checks, library_consulted=models is not None
        )
        notes.append(summary)
        if error is not None:
            problems.append(error)

    # Share logins, which is the one thing here that can be tried for
    # real without changing anything: `WNetAddConnection2W` either
    # establishes a session or says why, and a session this host would
    # have made at the next restart anyway is not a side effect worth
    # avoiding. An override in the body is honoured so the Test button
    # beside the editor answers about the row being typed, not the row
    # last saved -- and a password the operator just typed is used as
    # given, because it has not been sealed yet.
    auth_state = request.app.state.auth_state
    master_key = auth_state.master_key if auth_state.has_master_key() else None
    saved = share_credentials.unseal_entries(state.get_config(SHARE_CREDENTIALS_KEY), master_key)
    credential_entries = (
        share_credentials.overlay_typed(overrides[SHARE_CREDENTIALS_KEY], saved)
        if SHARE_CREDENTIALS_KEY in overrides
        else saved
    )
    if credential_entries:
        results = await asyncio.to_thread(share_credentials.connect_all, credential_entries)
        for outcome in results:
            (notes if outcome.ok else problems).append(outcome.summary)

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return ConfigTestResult(
        ok=not problems,
        component="agent",
        latencyMs=elapsed_ms,
        summary=" ".join(notes) or None,
        error="; ".join(problems) or None,
    )


@router.patch("/v1/config", response_model=ConfigUpdateResult)
async def patch_config(request: Request, body: ConfigUpdateRequest) -> ConfigUpdateResult:
    state: AgentState = request.app.state.agent_state

    # Snapshot the prior securityMode so we can react to a transition.
    # The keyring side-effects (write on flip to os_keyring, delete on
    # flip away) belong at this layer — `state` is a YAML serializer
    # and shouldn't know about OS secret stores.
    prior_mode = state.get_config("securityMode")
    prior_advertise = state.get_config("advertiseUrl")
    body, folder_rejection = await _check_overrides_name_folders(request, body)
    body, credential_rejection = _seal_share_credentials(request, body)
    result = state.apply_config_patch(body)
    if folder_rejection is not None:
        result.rejected.append(folder_rejection)
    if credential_rejection is not None:
        result.rejected.append(credential_rejection)
    new_mode = state.get_config("securityMode")

    # Log in to whatever the new list names, now, rather than at the next
    # restart. An operator who has just typed a password is watching, and
    # the model that would not open is the reason they typed it.
    if SHARE_CREDENTIALS_KEY in result.applied:
        await connect_shares(request)

    # An operator who changes where this host is reachable has to reach
    # the control root with it, or the root keeps routing to the old
    # address. The other half of the same fix announces on every boot;
    # this is the half that does not wait for one.
    if state.get_config("advertiseUrl") != prior_advertise:
        await _announce_advertise_url(request)

    # A new channel is checked now, so Versions says what the new channel
    # holds rather than nothing until the next six-hourly check. A result
    # for the old channel is already set aside (`UpdateChecker.current`).
    if "updateChannel" in result.applied and state.get_config("updateChecks") is not False:
        _check_updates_soon(request)

    if prior_mode == "os_keyring" and new_mode != "os_keyring":
        # Operator moved to the stronger boundary. The stored auto-
        # unlock secret must go — otherwise the install would still
        # auto-recover, contradicting the promise of the new mode.
        if keyring_store.delete_master_key(_install_id(state)):
            log.info(
                "securityMode changed from os_keyring to %s; deleted stored "
                "master key from OS keyring",
                new_mode,
            )
    elif prior_mode != "os_keyring" and new_mode == "os_keyring":
        # Operator opted into auto-unlock. If we already have the
        # master key in memory (logged in), persist it now so the
        # next restart actually auto-recovers. If we don't have it
        # in memory, the next /v1/auth/login will save it instead —
        # both paths converge to "next restart works".
        auth = request.app.state.auth_state
        if auth.has_master_key() and keyring_store.set_master_key(
            auth.master_key, _install_id(state)
        ):
            log.info("securityMode changed to os_keyring; persisted master key for auto-unlock")

    return result


SHARE_CREDENTIALS_KEY = "shareCredentials"


def _check_updates_soon(request: Request) -> None:
    from .. import install_info
    from .updates import checker_for

    app = request.app

    async def run() -> None:
        try:
            install = await asyncio.to_thread(install_info.describe)
            await checker_for(app).check(install)
        except Exception:  # pragma: no cover - a background check never fails a PATCH
            log.exception("update check after a channel change failed")

    tasks = getattr(app.state, "background_checks", None)
    if tasks is None:
        tasks = set()
        app.state.background_checks = tasks
    task = asyncio.get_running_loop().create_task(run(), name="update-check-after-channel")
    tasks.add(task)
    task.add_done_callback(tasks.discard)


def _seal_share_credentials(
    request: Request, body: ConfigUpdateRequest
) -> tuple[ConfigUpdateRequest, ConfigFieldError | None]:
    """Seal the passwords in a `shareCredentials` patch before it lands.

    **At this layer and not in `AgentState`, for the reason the
    securityMode side-effects are here**: the state object owns a YAML
    file and a lock and has deliberately never known about the master
    key. It is also the only layer that can merge — `GET` redacts, so an
    entry arriving with no password is a UI writing back a row it was
    shown, not an operator clearing one, and the stored secret has to
    survive that round trip or it is lost at the next reboot with
    nothing saying so.
    """
    patch = body.model_dump(exclude_unset=True)
    if SHARE_CREDENTIALS_KEY not in patch:
        return body, None
    if patch[SHARE_CREDENTIALS_KEY] is None:
        # `null` is the contract's "back to the default", and the default is
        # no logins. It used to be refused ("expected a list"), so the UI's
        # Reset to default could not reset this field at all.
        patch[SHARE_CREDENTIALS_KEY] = []
    state: AgentState = request.app.state.agent_state
    auth = request.app.state.auth_state
    sealed, error = share_credentials.merge_and_seal(
        patch[SHARE_CREDENTIALS_KEY],
        state.get_config(SHARE_CREDENTIALS_KEY),
        auth.master_key if auth.has_master_key() else None,
    )
    if error is not None:
        del patch[SHARE_CREDENTIALS_KEY]
        return ConfigUpdateRequest.model_validate(patch), ConfigFieldError(
            key=SHARE_CREDENTIALS_KEY, message=error
        )
    patch[SHARE_CREDENTIALS_KEY] = sealed
    return ConfigUpdateRequest.model_validate(patch), None


async def connect_shares(request: Request) -> list[share_credentials.ConnectResult]:
    """Ask the OS to log this host in to every configured file server.

    Off the event loop: `WNetAddConnection2W` against an unreachable
    server blocks for as long as the network stack allows, and this runs
    inside a config PATCH the browser is waiting on.
    """
    state: AgentState = request.app.state.agent_state
    auth = request.app.state.auth_state
    entries = share_credentials.unseal_entries(
        state.get_config(SHARE_CREDENTIALS_KEY),
        auth.master_key if auth.has_master_key() else None,
    )
    if not entries:
        return []
    return await asyncio.to_thread(share_credentials.connect_all, entries)


def _install_id(state: AgentState) -> str:
    """The keyring scope for this install — see `keyring_store.install_id_for`.

    Every caller here runs behind auth, so a passphrase and therefore a
    salt exist; the empty fallback only keeps a corrupt state file from
    raising inside a config write.
    """
    salt_b64 = state.get_master_salt_b64()
    return keyring_store.install_id_for(salt_b64) if salt_b64 else ""


async def _check_overrides_name_folders(
    request: Request, body: ConfigUpdateRequest
) -> tuple[ConfigUpdateRequest, ConfigFieldError | None]:
    """`pathMappings` is this node's overrides of Library folders, so a
    `from` that is no Library folder is rejected -- when the folder list
    is known. Never fetched means accepted with a warning: refusing an
    edit because the library is down would be the wrong kind of strict.
    The rest of the patch still applies."""
    patch = body.model_dump(exclude_unset=True)
    raw = patch.get(model_paths.CONFIG_KEY)
    if raw is None:
        return body, None
    await refresh_library_folders(request)
    cache = folder_cache_for(request)
    if cache is None or not cache.known:
        log.warning(
            "pathMappings saved without checking against the Library's folders: this node "
            "has never read them"
        )
        return body, None
    known = {library_folders.identity(f.path) for f in cache.folders or []}
    for index, rule in enumerate(model_paths.parse_rules(raw)):
        if library_folders.identity(rule.source) not in known:
            del patch[model_paths.CONFIG_KEY]
            return ConfigUpdateRequest.model_validate(patch), ConfigFieldError(
                key=model_paths.CONFIG_KEY,
                message=(
                    f"entry {index}: {rule.source!r} is not a Library folder. An override says "
                    f"where THIS machine mounts a Library folder; add the directory to the "
                    f"Library first (Library -> Folders), then say where it is here."
                ),
            )
    return body, None


@router.post("/v1/library/folders/check", response_model=LibraryFolderReach)
async def check_library_folders(
    request: Request, body: LibraryFolderCheckRequest | None = None
) -> LibraryFolderReach:
    """Where each Library folder is on this host, and whether it is there.

    One row per folder: the path this node would open, which rule said
    so (the folder's own mount, this node's override, or none), whether
    it exists here, and how many of the library's models under it are
    reachable. `pathMappings` in the body stands in for the saved
    overrides -- the Test beside an unsaved edit. Refreshes this node's
    copy of the folder list on the way when the library can be reached.
    """
    state: AgentState = request.app.state.agent_state
    cache = folder_cache_for(request)
    if cache is None:
        return LibraryFolderReach(libraryConsulted=False, folders=[])
    consulted = await refresh_library_folders(request)
    if body is not None and body.pathMappings is not None:
        overrides = model_paths.parse_rules(
            [m.model_dump(by_alias=True) for m in body.pathMappings]
        )
    else:
        overrides = model_paths.parse_rules(state.get_config(model_paths.CONFIG_KEY))
    library = await library_client_for(request)
    models = await library.list_models() if library is not None else None
    # Real filesystem calls, off the event loop: a dead share blocks for
    # as long as the OS allows.
    return await asyncio.to_thread(
        library_folders.check_reach,
        cache,
        overrides,
        models,
        library_consulted=consulted,
    )


async def _announce_advertise_url(request: Request) -> None:
    """Tell the control root this node moved, after a config edit.

    Never raises and never fails the edit: the config change is already
    persisted and correct locally, and a management-plane call that
    could undo an operator's save would be worse than one that logs.
    """
    identity = getattr(request.app.state, "node_identity", None)
    if identity is None or not identity.record.enrolled:
        return
    state: AgentState = request.app.state.agent_state
    settings = request.app.state.settings
    url = await enrollment.resolve_advertise_url(
        configured=state.get_config("advertiseUrl"),
        control_url=identity.record.control_url,
        bind_port=int(settings.bind_port),
        persisted=identity.record.advertise_url,
    )
    if url is None:
        return
    identity.record_advertise_url(url)
    await enrollment.announce_address(
        store=identity,
        url=url,
        transport=getattr(request.app.state, "control_transport", None),
    )
