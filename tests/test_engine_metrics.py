"""`GET /metrics` on the agent: this machine's engines for a scraper (row 12).

What these show: every series is labelled with the runtime it came from; a
counter an engine lacks has no series and is never a 0; an engine that does not
answer says so in `eugene_engine_scrape_ok` and nothing else; and the page is
read by an operator or by a key with `readMetrics`, and by nobody else.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import engine_metrics
from eugene_plexus_agent.routes import metrics as metrics_route

LLAMA = """\
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 120
# HELP llamacpp:tokens_predicted_total Number of generation tokens processed.
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 45
# HELP llamacpp:requests_processing Number of requests processing.
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 2
# HELP llamacpp:requests_deferred Number of requests deferred.
# TYPE llamacpp:requests_deferred gauge
llamacpp:requests_deferred 3
# TYPE llamacpp:kv_cache_usage_ratio gauge
llamacpp:kv_cache_usage_ratio 0.25
"""

VLLM = """\
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="Big"} 7
vllm:num_requests_running{engine="0",model_name="Big"} 1
vllm:kv_cache_usage_perc{engine="0",model_name="Big"} 0.5
vllm:prompt_tokens_total{engine="0",model_name="Big"} 900
vllm:generation_tokens_total{engine="0",model_name="Big"} 300
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="Big"} 40
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="Big"} 30
"""


def _targets() -> list[engine_metrics.Target]:
    return [
        engine_metrics.Target("n1", "qwen", "Qwen", "llama_cpp", "http://qwen.test"),
        engine_metrics.Target("n1", "big", "Big", "vllm", "http://big.test"),
        engine_metrics.Target("n1", "down", "Down", "llama_cpp", "http://down.test"),
    ]


def _client() -> httpx.AsyncClient:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/metrics"
        if request.url.host == "qwen.test":
            return httpx.Response(200, text=LLAMA)
        if request.url.host == "big.test":
            return httpx.Response(200, text=VLLM)
        raise httpx.ConnectError("connection refused", request=request)

    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


async def _page() -> str:
    reads = await engine_metrics.gather(_client(), _targets())
    return engine_metrics.render(reads)


@pytest.mark.asyncio
async def test_engine_metrics_are_labelled_per_runtime() -> None:
    page = await _page()
    qwen = 'node="n1",runtime="qwen",model="Qwen",engine="llama_cpp"'
    big = 'node="n1",runtime="big",model="Big",engine="vllm"'
    assert f"eugene_engine_requests_waiting{{{qwen}}} 3" in page
    assert f"eugene_engine_requests_running{{{qwen}}} 2" in page
    assert f"eugene_engine_kv_cache_usage_ratio{{{qwen}}} 0.25" in page
    assert f"eugene_engine_prompt_tokens_total{{{qwen}}} 120" in page
    assert f"eugene_engine_generated_tokens_total{{{qwen}}} 45" in page
    assert f"eugene_engine_requests_waiting{{{big}}} 7" in page
    assert f"eugene_engine_generated_tokens_total{{{big}}} 300" in page
    assert f"eugene_engine_draft_tokens_total{{{big}}} 40" in page
    assert f"eugene_engine_draft_accepted_tokens_total{{{big}}} 30" in page
    assert f"eugene_engine_scrape_ok{{{qwen}}} 1" in page


@pytest.mark.asyncio
async def test_a_counter_the_engine_lacks_has_no_series_never_a_zero() -> None:
    page = await _page()
    # llama.cpp here is not drafting: it said nothing, so the page says nothing.
    assert 'eugene_engine_draft_tokens_total{node="n1",runtime="qwen"' not in page
    assert 'eugene_engine_draft_accepted_tokens_total{node="n1",runtime="qwen"' not in page


@pytest.mark.asyncio
async def test_an_engine_that_does_not_answer_says_so_and_nothing_else() -> None:
    page = await _page()
    down = 'node="n1",runtime="down",model="Down",engine="llama_cpp"'
    assert f"eugene_engine_scrape_ok{{{down}}} 0" in page
    assert page.count('runtime="down"') == 1


@pytest.mark.asyncio
async def test_the_engines_own_series_pass_through_with_the_same_labels() -> None:
    page = await _page()
    qwen = 'node="n1",runtime="qwen",model="Qwen",engine="llama_cpp"'
    assert f"llamacpp:requests_deferred{{{qwen}}} 3" in page
    assert "# TYPE llamacpp:requests_deferred gauge" in page
    # vLLM's own `engine` label would repeat ours, which Prometheus refuses.
    assert (
        'vllm:num_requests_waiting{exported_engine="0",model_name="Big",'
        'node="n1",runtime="big",model="Big",engine="vllm"} 7'
    ) in page


@pytest.mark.asyncio
async def test_a_slow_engine_costs_its_own_series_not_the_scrape() -> None:
    import asyncio

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, text=LLAMA)

    client = httpx.AsyncClient(transport=httpx.MockTransport(slow))
    reads = await engine_metrics.gather(client, _targets()[:1], seconds=0.05)
    page = engine_metrics.render(reads)
    assert 'eugene_engine_scrape_ok{node="n1",runtime="qwen"' in page
    assert page.count("eugene_engine_scrape_ok") == 3  # HELP, TYPE and one sample
    assert "requests_waiting{" not in page


def test_the_parser_drops_a_line_it_cannot_read_and_keeps_the_rest() -> None:
    parsed = engine_metrics.parse(
        'a_total 1\nnot a metric line\nb{x="a}b"} 2 1700000000\nc nope\nd{x="q\\"r"} 3\n'
    )
    assert [(s.name, s.value) for s in parsed.samples] == [("a_total", "1"), ("b", "2"), ("d", "3")]
    assert parsed.samples[1].labels == [("x", "a}b")]


# --------------------------------------------------------------------- #
# who may read it
# --------------------------------------------------------------------- #


def _mint(client: TestClient, name: str, **limits: object) -> tuple[str, str]:
    resp = client.post("/v1/auth/client-keys", json={"name": name, "limits": limits})
    assert resp.status_code == 201, resp.text
    made = resp.json()
    return made["token"], made["key"]["id"]


@pytest.fixture
def anonymous(app: FastAPI, authed_client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(metrics_route, "_targets", lambda request: _targets())
    monkeypatch.setattr(metrics_route, "engine_client", _client)
    with TestClient(app) as c:
        yield c


def _get(client: TestClient, token: str | None):
    headers = {"authorization": f"Bearer {token}"} if token else {}
    return client.get("/metrics", headers=headers)


def test_a_key_that_may_read_metrics_reads_them(
    authed_client: TestClient, anonymous: TestClient
) -> None:
    token, _ = _mint(authed_client, "prometheus", readMetrics=True, allowedModels=[])
    resp = _get(anonymous, token)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/plain")
    assert 'eugene_engine_requests_waiting{node="n1",runtime="qwen"' in resp.text


def test_no_credential_is_401_and_a_key_without_the_grant_is_403(
    authed_client: TestClient, anonymous: TestClient
) -> None:
    assert _get(anonymous, None).status_code == 401
    token, _ = _mint(authed_client, "Continue")
    refused = _get(anonymous, token)
    assert refused.status_code == 403
    assert "May read metrics" in refused.text
    # A key that may run models reads no metrics by that alone.
    chat, _ = _mint(authed_client, "chat", allowedModels=["*"])
    assert _get(anonymous, chat).status_code == 403


def test_the_operator_session_reads_them(authed_client: TestClient, anonymous: TestClient) -> None:
    session = authed_client.headers["Authorization"].split(" ", 1)[1]
    assert _get(anonymous, session).status_code == 200


def test_a_revoked_scrape_key_is_refused(authed_client: TestClient, anonymous: TestClient) -> None:
    token, key_id = _mint(authed_client, "prometheus", readMetrics=True)
    assert _get(anonymous, token).status_code == 200
    assert authed_client.delete(f"/v1/auth/client-keys/{key_id}").status_code == 204
    # The standing is kept for a few seconds, as the log ingress keeps it.
    anonymous.app.state.metrics_keys.clear()
    assert _get(anonymous, token).status_code in (401, 403)


def test_a_scrape_key_that_loses_the_grant_is_refused(
    authed_client: TestClient, anonymous: TestClient
) -> None:
    token, key_id = _mint(authed_client, "prometheus", readMetrics=True)
    assert _get(anonymous, token).status_code == 200
    edited = authed_client.put(
        f"/v1/auth/client-keys/{key_id}/limits", json={"limits": {"readMetrics": False}}
    )
    assert edited.status_code == 200, edited.text
    anonymous.app.state.metrics_keys.clear()
    assert _get(anonymous, token).status_code == 403
