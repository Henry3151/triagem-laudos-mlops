"""API de triagem de laudos (FastAPI)."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.routing import Match

from triagem import __version__
from triagem.api.metricas import MetricasApi
from triagem.api.schemas import PedidoTriagem, RespostaSaude, RespostaTriagem
from triagem.classificadores import Classificador, carregar_classificador

logger = logging.getLogger("triagem.api")
ROTA_DESCONHECIDA = "desconhecida"


def _rota(request: Request) -> str:
    """Template da rota (ex.: /predict) para manter baixa a cardinalidade dos labels."""
    for rota in request.app.router.routes:
        correspondencia, _ = rota.matches(request.scope)
        if correspondencia == Match.FULL:
            return rota.path
    return ROTA_DESCONHECIDA


def criar_app(classificador: Classificador | None = None) -> FastAPI:
    metricas = MetricasApi()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        clf = classificador or carregar_classificador(
            Path(os.environ.get("MODEL_DIR", "models/producao")),
            os.environ.get("MODEL_BACKEND", "onnx"),
        )
        app.state.classificador = clf
        metricas.modelo.info({"backend": clf.backend, "versao": clf.versao})
        logger.info("Modelo %s carregado (backend=%s)", clf.versao, clf.backend)
        yield

    app = FastAPI(
        title="Triagem de Laudos",
        description="Classifica laudos médicos em normal, atenção ou urgente.",
        version=__version__,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def medir_requisicoes(request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)
        inicio = time.perf_counter()
        status = 500
        try:
            resposta = await call_next(request)
            status = resposta.status_code
            return resposta
        finally:
            rota = _rota(request)
            metricas.requisicoes.labels(request.method, rota, str(status)).inc()
            metricas.duracao.labels(request.method, rota).observe(time.perf_counter() - inicio)

    @app.post("/predict", response_model=RespostaTriagem)
    def predict(pedido: PedidoTriagem, request: Request) -> RespostaTriagem:
        clf: Classificador = request.app.state.classificador
        inicio = time.perf_counter()
        try:
            predicao = clf.prever(pedido.texto)
        except Exception as erro:
            logger.exception("Falha na inferência")
            raise HTTPException(500, "Erro interno ao classificar o laudo") from erro
        metricas.inferencia.labels(clf.backend).observe(time.perf_counter() - inicio)
        metricas.predicoes.labels(predicao.classe).inc()
        return RespostaTriagem(
            classe=predicao.classe,
            probabilidades=predicao.probabilidades,
            versao_modelo=clf.versao,
            backend=clf.backend,
        )

    @app.get("/health", response_model=RespostaSaude)
    def health(request: Request) -> RespostaSaude:
        clf: Classificador = request.app.state.classificador
        return RespostaSaude(status="ok", versao_modelo=clf.versao, backend=clf.backend)

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(metricas.registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = criar_app()
