"""Contratos de entrada e saída da API."""

from pydantic import BaseModel, ConfigDict, Field

EXEMPLO = (
    "Tomografia computadorizada de crânio sem contraste. Indicação: trauma após queda. "
    "Hematoma subdural agudo com efeito de massa."
)


class PedidoTriagem(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True, json_schema_extra={"examples": [{"texto": EXEMPLO}]}
    )

    texto: str = Field(min_length=1, max_length=5000, description="Texto livre do laudo")


class RespostaTriagem(BaseModel):
    classe: str
    probabilidades: dict[str, float]
    versao_modelo: str
    backend: str


class RespostaSaude(BaseModel):
    status: str
    versao_modelo: str
    backend: str
