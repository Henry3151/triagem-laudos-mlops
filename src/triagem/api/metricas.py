"""Métricas Prometheus da API (registry próprio por instância de app)."""

from prometheus_client import CollectorRegistry, Counter, Histogram, Info, ProcessCollector

# Buckets abaixo de 1 ms: a inferência ONNX leva ~0,1 ms (senão o P95 vira pura interpolação).
BUCKETS = (
    0.0001,
    0.00025,
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
)


class MetricasApi:
    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        ProcessCollector(registry=self.registry)
        self.requisicoes = Counter(
            "triagem_http_requests",
            "Total de requisições HTTP",
            ["method", "rota", "status"],
            registry=self.registry,
        )
        self.duracao = Histogram(
            "triagem_http_request_duration_seconds",
            "Latência das requisições HTTP",
            ["method", "rota"],
            buckets=BUCKETS,
            registry=self.registry,
        )
        self.inferencia = Histogram(
            "triagem_inferencia_duration_seconds",
            "Tempo apenas da predição do modelo",
            ["backend"],
            buckets=BUCKETS,
            registry=self.registry,
        )
        self.predicoes = Counter(
            "triagem_predicoes",
            "Predições por classe",
            ["classe"],
            registry=self.registry,
        )
        self.modelo = Info("triagem_modelo", "Versão e backend do modelo", registry=self.registry)
