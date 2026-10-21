"""HTTP API + dashboard.

    uvicorn semcache.api:app --reload        # then open http://localhost:8000
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .gateway import Gateway

DASHBOARD = Path(__file__).parent / "dashboard.html"


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    namespace: str = "default"
    force_tier: str | None = Field(None, pattern="^(small|large)$")


def create_app(gateway: Gateway | None = None) -> FastAPI:
    app = FastAPI(title="semcache-router", version="0.1.0")
    app.state.gateway = gateway

    def gw() -> Gateway:
        if app.state.gateway is None:  # lazy so importing the module doesn't load the embedder
            app.state.gateway = Gateway.from_settings()
        return app.state.gateway

    @app.post("/ask")
    def ask(req: AskRequest):
        try:
            return asdict(gw().ask(req.query, req.namespace, req.force_tier))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/stats")
    def stats():
        g = gw()
        s = g.log.summary()
        s["cache_entries"] = len(g.cache)
        s["config"] = {
            "similarity_threshold": g.cache.threshold,
            "entity_guard": g.cache.entity_guard,
            "route_threshold": g.router.threshold,
            "small_model": g.settings.small.name,
            "large_model": g.settings.large.name,
            "embedder": type(g.embedder).__name__,
            "llm": type(g.llm).__name__,
        }
        return s

    @app.get("/requests")
    def requests(limit: int = 50):
        return gw().log.recent(min(limit, 500))

    @app.post("/reset")
    def reset():
        g = gw()
        g.cache.clear()
        g.log.clear()
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return DASHBOARD.read_text(encoding="utf-8")

    return app


app = create_app()
