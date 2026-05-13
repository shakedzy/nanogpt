"""FastAPI app: holds the model in memory and exposes analysis endpoints.

Run with:
    uv run uvicorn server.main:app --reload --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from server import model_runner
from server.analyses import ablation as ablation_analysis
from server.analyses import attention as attention_analysis
from server.analyses import induction as induction_analysis
from server.analyses import logit_lens as logit_lens_analysis
from server.analyses import neurons as neurons_analysis


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_runner.load()
    yield


app = FastAPI(title="NanoGPT Interp", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForwardRequest(BaseModel):
    prompt: str


@app.get("/", include_in_schema=False)
def root():
    # The UI lives on the Vite dev server; backend is API-only.
    return RedirectResponse(url="http://localhost:5173/")


@app.get("/health")
def health() -> dict:
    r = model_runner.load()
    return {
        "status": "ok",
        "device": str(r.device),
        "vocab_size": r.vocab_size,
        "context_length": r.model.context_length,
    }


@app.post("/forward")
def forward(req: ForwardRequest) -> dict:
    return model_runner.run(req.prompt)


@app.post("/attention")
def attention(req: ForwardRequest) -> dict:
    return attention_analysis.compute(req.prompt)


@app.post("/logit_lens")
def logit_lens(req: ForwardRequest) -> dict:
    return logit_lens_analysis.compute(req.prompt)


@app.get("/neurons/{layer}")
def neurons_layer(layer: int) -> dict:
    return neurons_analysis.get_layer_summary(layer)


@app.get("/neuron/{layer}/{idx}")
def neuron(layer: int, idx: int) -> dict:
    return neurons_analysis.get_neuron(layer, idx)


class AblateRequest(BaseModel):
    prompt: str
    ablate_heads: list[list[int]] = []      # [[layer, head], ...]
    ablate_neurons: list[list[int]] = []    # [[layer, neuron_idx], ...]


@app.post("/ablate")
def ablate(req: AblateRequest) -> dict:
    return ablation_analysis.compute(
        prompt=req.prompt,
        ablate_heads=req.ablate_heads,
        ablate_neurons=req.ablate_neurons,
    )


@app.post("/induction_scan")
def induction_scan() -> dict:
    return induction_analysis.compute()
