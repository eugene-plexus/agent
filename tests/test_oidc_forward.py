"""The shared upstream pool must not become a browser cookie jar."""

from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from eugene_plexus_agent.routes.oidc_forward import router


@pytest.mark.asyncio
async def test_shared_pool_keeps_visitors_separate_and_forwards_only_sign_in_cookies():
    seen = []

    def root(request):
        seen.append(request)
        return httpx.Response(
            200,
            text="Sign in",
            headers=[
                ("set-cookie", "ep-signin=person-a; Path=/oidc; HttpOnly; SameSite=lax"),
                ("set-cookie", "ep-signin-flow=flow-a; Path=/oidc; HttpOnly; SameSite=lax"),
                ("set-cookie", "unrelated=do-not-forward; Path=/"),
            ],
        )

    app = FastAPI()
    app.include_router(router)
    app.state.node_identity = SimpleNamespace(
        record=SimpleNamespace(enrolled=True, control_url="http://root.test")
    )
    app.state.auth_state = SimpleNamespace(
        trust=SimpleNamespace(agent_token=lambda _: "node-token")
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(root)) as pool:
        app.state.oidc_forward_client = pool
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://agent.test"
        ) as visitor:
            first = await visitor.get(
                "/oidc/account",
                headers={
                    "cookie": "ep-signin=first; workbench=private",
                    "user-agent": "Visitor browser",
                },
            )
            assert seen[0].headers["cookie"] == "ep-signin=first"
            assert seen[0].headers["user-agent"] == "Visitor browser"
            assert len(first.headers.get_list("set-cookie")) == 2
            assert "unrelated" not in first.headers["set-cookie"]
            assert pool.cookies.get("ep-signin") == "person-a"
            visitor.cookies.clear()
            await visitor.get("/oidc/account")
            assert seen[1].headers["cookie"] == ""
            await visitor.get(
                "/oidc/account", headers={"cookie": "ep-signin=second; ep-signin-flow=second-flow"}
            )
            assert seen[2].headers["cookie"] == "ep-signin=second; ep-signin-flow=second-flow"
