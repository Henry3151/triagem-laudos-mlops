# Triagem Automática de Laudos — Spec de Design

- **Data:** 2026-09-25
- **Status:** aprovada para planejamento (execução autônoma autorizada pelo usuário)
- **Base:** `docs/enunciado_tech_challenge.md` (requisitos e pesos) e `docs/Resumo_Aulas_Tech_Challenge.md` (conteúdo das aulas)

## 1. Entendimento do objetivo

**O que foi pedido (enunciado):** um sistema que classifica laudos médicos em texto por urgência
(`normal` / `atencao` / `urgente`), servido por uma API REST FastAPI em Docker, com:

- CI/CD no GitHub Actions (≥ 2 automações: lint e testes, mais build);
- DAG Airflow funcional (carregar dados → treinar → salvar modelo);
- stack local API + Prometheus + Grafana via Docker Compose, dashboard com ≥ 3 painéis;
- pelo menos uma otimização de latência vista em aula (ONNX, quantização ou pruning), com comparação
  entre o modelo original e o otimizado;
- README com a decisão de arquitetura em nuvem (batch vs. real-time) e instruções de execução;
- histórico de commits semântico.

**Critério de sucesso:** cada linha da tabela de avaliação do enunciado tem um entregável verificável
no repositório (seção 11). O vídeo STAR é gravado pelo usuário; o repositório entrega um roteiro e os
números que o vídeo mostra.

**Premissas (decisões tomadas sem consulta, por delegação do usuário):**

- Dataset sintético em português, gerado por script determinístico (seção 4).
- Um único provedor de nuvem na decisão documental: GCP (Cloud Run), padrão real-time (seção 3).
- Técnica de otimização: conversão para ONNX + ONNX Runtime (seção 6).
- Nada é implantado de fato em nuvem; a nuvem entra apenas na documentação, como o enunciado pede.

## 2. Ambiente verificado (2026-09-25)

| Ferramenta | Versão encontrada | Uso no projeto |
|---|---|---|
| Docker Engine | 29.3.1 | build e execução de todos os serviços |
| Docker Compose | v5.1.1 | stacks de monitoramento e de Airflow |
| Airflow | não instalado localmente; última versão no PyPI: 3.3.2 | roda em container `apache/airflow:3.3.2-python3.12` |
| Python local | 3.14.3 e 3.13 | dev local via `uv` com Python 3.12 gerenciado pelo `uv` |
| uv | 0.11.26 | gerenciador de ambiente e lockfile |
| gh | 2.89.0 (sem login) | criação do repositório e push |

**Python 3.12 em todo lugar** (imagem da API, imagem do Airflow, CI e venv local): evita
*environment drift* entre treino e serving (Aula 5 de CI/CD).

## 3. Decisão arquitetural de deploy em nuvem (vai para o README)

**Padrão escolhido: real-time (API síncrona)**, com batch apenas como complemento futuro.

- A triagem existe para mudar o fluxo imediato do atendimento: um laudo `urgente` precisa subir na
  fila no momento em que é emitido. Isso é o critério de "resposta altera o fluxo da aplicação" das
  aulas (Deploy em Nuvem, Aula 1; Pipeline de Treino, Aula 4). Batch noturno atrasaria casos críticos
  em horas.
- O modelo (TF-IDF + Random Forest) tem custo de inferência constante e pequeno por requisição, e o
  artefato cabe em memória: é compatível com uma API de baixa latência (Deploy em Nuvem, Aula 2).
- Serverless puro é descartado como padrão principal por causa do **cold start** em um caso de uso
  crítico; o desenho usa um serviço serverless de containers **com instância mínima sempre ativa**.

**Provedor: GCP.**

| Camada | Serviço GCP | Equivalente AWS / Azure |
|---|---|---|
| Registro de imagem | Artifact Registry | ECR / ACR |
| Serving real-time | Cloud Run (`min-instances=1`, ingress interno, porta via `PORT`) | ECS Fargate ou SageMaker Endpoint / Container Apps |
| Artefatos de modelo e dados | Cloud Storage | S3 / Blob Storage |
| Orquestração de retreino | Cloud Composer (Airflow gerenciado) | MWAA / Azure Data Factory com Airflow |
| Observabilidade | Managed Service for Prometheus + Grafana, Cloud Logging | CloudWatch / Azure Monitor |
| CI/CD | GitHub Actions com OIDC (Workload Identity Federation) | OIDC com IAM Roles / Managed Identity |

O README também registra: segurança (dados de saúde sob a LGPD, anonimização antes do treino,
menor privilégio, sem credenciais estáticas), FinOps (instância mínima única, escala por
concorrência, labels de custo) e a alternativa batch (reprocessamento noturno de laudos antigos via
Cloud Run Jobs). **Nada disso é provisionado**; o enunciado pede apenas a análise textual.

## 4. Dados

**Dataset sintético em português**, gerado por `triagem.dados.gerar_dataset(n=3000, seed=42)` e
versionado em `data/raw/laudos_sinteticos.csv`.

- **Por quê:** os datasets sugeridos não servem diretamente. O MIMIC-III exige credenciamento, e o
  Medical Abstracts TC Corpus classifica especialidade, não urgência (mapeá-lo para urgência seria
  inventar rótulos). Um gerador determinístico entrega exatamente as três classes do enunciado,
  roda offline no CI e no Airflow e é reprodutível byte a byte. A limitação fica declarada no README.
- **Esquema:** `id` (int), `texto` (str), `classe` (`normal` | `atencao` | `urgente`).
- **Tamanho e balanceamento:** 3.000 linhas (≥ 2.000 exigidas), aproximadamente 45% normal,
  35% atenção e 20% urgente (desbalanceado de propósito, como na realidade).
- **Realismo mínimo para o modelo não ser trivial:** templates por modalidade de exame (raio-X,
  tomografia, ultrassom, ECG, laboratório); vocabulário compartilhado entre classes; negações
  ("sem sinais de hemorragia" → normal); termos de gravidade graduados; ~3% de ruído de rótulo. A
  meta é F1-macro entre 0,85 e 0,97, e não 1,0.
- **Validação (contrato de dados, fail-fast):** `triagem.dados.validar_dataset(df)` verifica colunas,
  tipos, ausência de nulos e de textos vazios, classes dentro do conjunto permitido, ≥ 2.000 linhas e
  as três classes presentes. Se falhar, lança `DatasetInvalidoError` e o treino não roda.

## 5. Modelo

- **Pré-processamento compartilhado** (`triagem.preprocessamento.normalizar_texto`): minúsculas,
  remoção de acentos (NFKD), colapso de espaços. A mesma função é usada no treino, na API e nos dois
  backends, o que evita *training-serving skew* (Pipeline de Treino, Aula 1). Remover acentos antes
  também deixa a tokenização idêntica entre scikit-learn e ONNX.
- **Pipeline scikit-learn:** `TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000,
  sublinear_tf=False)` (na implementação ficou comprovado que o skl2onnx não reproduz `sublinear_tf=True`:
  termos repetidos no laudo divergiam e a paridade quebrava) e `lowercase=False` (a caixa já é tratada
  por `normalizar_texto`; isso evita o operador ONNX `StringNormalizer`, que exige o locale
  `en_US.UTF-8`, ausente nas imagens slim) → `RandomForestClassifier(n_estimators=200, class_weight="balanced",
  random_state=42, n_jobs=1)`. O `n_jobs=1` garante uma comparação de latência justa por requisição.
- **Split:** estratificado 80/20, `random_state=42`.
- **Métricas registradas:** acurácia, F1-macro, recall da classe `urgente` (o erro mais caro é
  deixar um urgente passar como normal) e a matriz de confusão.
- **Artefatos por versão** em `models/<versao>/`: `modelo.joblib`, `modelo.onnx` e `metadata.json`
  (versão, data UTC, sha256 do CSV, versões de sklearn/skl2onnx/onnxruntime, métricas, classes).
  `versao` = timestamp UTC `YYYYMMDDTHHMMSSZ`.
- **Modelo em produção:** `models/producao/` recebe uma cópia da versão promovida (seção 8).

## 6. Otimização de latência: ONNX + ONNX Runtime

- **Conversão:** `skl2onnx.to_onnx` do pipeline inteiro (TF-IDF + RF) com entrada string `[None, 1]`,
  `zipmap=False` (saída de probabilidades como tensor) e `tokenexp` equivalente ao `token_pattern`
  do sklearn.
- **Paridade (bloqueante):** no conjunto de teste, a concordância de classe entre sklearn e ONNX deve
  ser ≥ 99% e a diferença absoluta máxima de probabilidade ≤ 1e-2 (o ONNX usa float32 nas árvores;
  no spike a diferença medida foi de ~5e-3, com 100% de concordância). Se falhar, a exportação lança
  `ParidadeOnnxError` e a versão não é promovida.
- **Benchmark** (`triagem.benchmark`), com a metodologia das aulas de Latência e Monitoração:
  percentis P50/P95/P99 (não a média), warmup de 50 chamadas, 1.000 chamadas com batch 1, textos
  reais do conjunto de teste, sklearn com `n_jobs=1` contra ONNX Runtime com uma thread
  (`intra_op_num_threads=1`).
  - Modo `inprocess`: mede só a chamada de predição, incluindo o pré-processamento.
  - Modo `http`: mede ponta a ponta contra a API em Docker, uma vez com `MODEL_BACKEND=sklearn` e
    outra com `MODEL_BACKEND=onnx`. É o "baseline de latência local" da Etapa 1.
  - Também mede o tamanho do artefato e o tempo de carga do modelo.
  - A saída vai para `reports/latencia.json` e `reports/latencia.md`, ambos versionados, com a máquina
    e as versões registradas.
- **Expectativa (a verificar, não prometer):** ONNX Runtime reduz de forma relevante a latência de
  árvore por requisição. Se o ganho medido for pequeno, o relatório diz isso honestamente, e o
  critério "melhoria demonstrada" passa a depender dos números reais.

## 7. API de inferência (FastAPI)

Pacote `triagem.api`, app criada por factory `criar_app(classificador=None)`. A injeção do
classificador facilita os testes.

| Método | Rota | Resposta |
|---|---|---|
| POST | `/predict` | `{"classe", "probabilidades": {classe: float}, "versao_modelo", "backend"}` |
| GET | `/health` | `{"status": "ok", "versao_modelo", "backend"}` |
| GET | `/metrics` | formato texto do Prometheus |

- **Entrada:** `{"texto": str}` com 1 ≤ tamanho ≤ 5.000 após `strip`. Violações retornam **422**.
  Exceções inesperadas retornam **500** com corpo genérico (sem stack trace) e log estruturado.
- **Backends:** a interface `Classificador` (`prever(texto) -> Predicao`) tem duas implementações,
  `SklearnClassificador` (joblib) e `OnnxClassificador` (onnxruntime). A escolha vem de
  `MODEL_BACKEND` (`onnx` por padrão) e o diretório de `MODEL_DIR` (`/app/models/producao` por
  padrão). O modelo é carregado **uma vez** no startup (lifespan), nunca por requisição.
- **Rota síncrona** (`def`, não `async def`): a inferência é CPU-bound e roda no threadpool do
  Starlette sem bloquear o event loop (Monitoração, Aula 6).
- **Métricas Prometheus** (middleware + instrumentação explícita):
  - `triagem_http_requests_total{method, rota, status}`: Counter.
  - `triagem_http_request_duration_seconds{method, rota}`: Histogram, buckets de 1 ms a 2,5 s.
  - `triagem_inferencia_duration_seconds{backend}`: Histogram, só a predição.
  - `triagem_predicoes_total{classe}`: Counter.
  - `triagem_modelo_info{versao, backend}`: Info.
  - O label `rota` usa o template da rota (baixa cardinalidade). `/metrics` não é contado.
- **Imagem Docker** (`Dockerfile` na raiz), em multi-stage:
  1. Estágio `treino`: instala as dependências de treino, gera o dataset, treina e exporta o ONNX
     com o mesmo CLI do projeto (`python -m triagem.treinar --saida /build/models`).
  2. Estágio `runtime`: `python:3.12-slim`, só as dependências de serving (sem skl2onnx e sem pandas),
     copia o código e `models/producao`, roda como usuário não-root, tem `HEALTHCHECK` em `/health`
     e roda `uvicorn` na porta `${PORT:-8000}` (compatível com Cloud Run).
  - A imagem é autocontida: `docker build` sozinho produz um serviço funcional, o que o CI verifica.

## 8. Orquestração de retreino (Airflow 3.3.2)

**DAG `triagem_retreino`** (`dags/triagem_retreino.py`, TaskFlow API, `schedule="@weekly"`,
`catchup=False`, também disparável manualmente). As tasks são finas: cada uma chama uma função de
`triagem.pipeline`, que é testada sem Airflow. Entre as tasks trafegam só **caminhos e métricas via
XCom**, nunca os dados (Pipeline de Treino, Aula 5).

1. `ingerir_dados`: lê `data/raw/laudos_sinteticos.csv`, executa `validar_dataset` (fail-fast), faz
   o split estratificado e grava `data/processed/<run_id>/{treino,teste}.csv`. Devolve os caminhos.
2. `treinar_modelo`: treina o pipeline e salva `models/<versao>/modelo.joblib`. Devolve o diretório.
3. `avaliar_modelo`: calcula as métricas no teste e grava `metricas.json`. Devolve as métricas.
4. `exportar_onnx`: converte, verifica a paridade e grava `modelo.onnx` e `metadata.json`.
5. `promover_modelo`: **quality gate** (champion vs. challenger, CI/CD Aula 8). Promove para
   `models/producao/` somente se F1-macro ≥ 0,80, recall de urgente ≥ 0,85 e F1-macro ≥ F1 do modelo
   atual em produção − 0,01 (quando existir um). Caso contrário, registra o motivo e não promove; a
   task não falha, porque rejeitar um challenger é um resultado válido. Devolve a decisão.

**Execução local:** `docker-compose.airflow.yml` com um único serviço `airflow standalone`,
construído de `airflow/Dockerfile` (`FROM apache/airflow:3.3.2-python3.12` mais as dependências de
treino fixadas nas mesmas versões do `uv.lock`). Monta `dags/`, `src/`, `data/` e `models/`. Usa
`AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_ALL_ADMINS=True` para não exigir login na demonstração local,
e o README avisa que isso é só para dev. UI na porta 8080.

**Relação com o serving:** a DAG produz versões e promove para `models/producao/` no host. Em
produção (seção 3), o equivalente é o Composer gravar no Cloud Storage e um deploy do Cloud Run
carregar a nova versão. Localmente, a API do compose de monitoramento usa o modelo embutido na
imagem. Para servir um modelo retreinado, basta subir o compose com o override opcional
`MODEL_DIR` montando `./models/producao`, que fica documentado no README. Esse passo não é
automático, de propósito, para manter o escopo pequeno.

## 9. Monitoramento (Prometheus + Grafana)

**`docker-compose.yml`**, com os serviços:

- `api`: build do `Dockerfile`, porta 8000.
- `prometheus` (`prom/prometheus:v3.5.0`): coleta `api:8000/metrics` a cada 5 s, configurado em
  `monitoring/prometheus/prometheus.yml`.
- `grafana` (`grafana/grafana:12.1.0`): porta 3000, com datasource e dashboard **provisionados como
  código** (`monitoring/grafana/provisioning/...` e `monitoring/grafana/dashboards/triagem.json`).
  Login anônimo com papel Viewer habilitado para a demo; admin/admin continua disponível.

**Dashboard "Triagem de Laudos — API"** (7 painéis, contra o mínimo de 3; o item 6 vira 2 stats):

1. Requisições por segundo por status: `sum by (status) (rate(triagem_http_requests_total{rota="/predict"}[1m]))`.
2. Latência HTTP P50/P95/P99: `histogram_quantile(0.95, sum by (le) (rate(triagem_http_request_duration_seconds_bucket{rota="/predict"}[1m])))`, e o mesmo para 0.5 e 0.99.
3. Taxa de erro (%): razão entre 4xx+5xx e o total em `/predict`.
4. Distribuição de predições por classe: `sum by (classe) (increase(triagem_predicoes_total[5m]))`.
5. Latência de inferência P95 por backend, a partir de `triagem_inferencia_duration_seconds`.
6. Stat do total de requisições e da versão/backend do modelo, a partir de `triagem_modelo_info`.

**Gerador de carga** (`scripts/gerar_carga.py`, httpx): envia laudos do dataset a uma taxa e duração
configuráveis, com ~5% de payloads inválidos de propósito, para o painel de erro ter o que mostrar.
Um print do dashboard fica em `docs/img/dashboard.png` (entregável da Etapa 3).

## 10. CI/CD (GitHub Actions)

`.github/workflows/ci.yml`, disparado em `push` e `pull_request` para `main`. Quatro jobs:

1. `lint`: `ruff check` e `ruff format --check`.
2. `test`: `uv sync --locked` e `pytest` com cobertura mínima de 80% sobre `src/triagem`.
3. `build`, que depende de lint e test: `docker build`, depois um smoke test que sobe o container,
   espera o `/health`, faz um `POST /predict` válido (espera 200), um inválido (espera 422) e confere
   que `/metrics` contém `triagem_http_requests_total`.
4. `dag-check`: constrói `airflow/Dockerfile` e carrega a DAG com `DagBag` dentro do container,
   falhando se houver erros de importação ou se as 5 tasks esperadas não estiverem presentes.

Cache do uv (`astral-sh/setup-uv` com cache) e do Docker (`docker/build-push-action` com cache do
GitHub Actions). Não há deploy em nuvem no CD: o pipeline termina na imagem validada. O README
descreve o passo seguinte (push para o Artifact Registry via OIDC e deploy no Cloud Run) como
evolução.

## 11. Estrutura do repositório

```
.github/workflows/ci.yml
airflow/Dockerfile, airflow/check_dag.py  # imagem do Airflow e validação da DAG no CI
dags/triagem_retreino.py
data/raw/laudos_sinteticos.csv          # versionado; data/processed/ fica no .gitignore
docs/
  enunciado_tech_challenge.md
  arquitetura.md                        # diagrama (Mermaid) e fluxo ponta a ponta
  roteiro_video_star.md
  img/dashboard.png
  superpowers/specs/ e superpowers/plans/
monitoring/
  prometheus/prometheus.yml
  grafana/provisioning/datasources/prometheus.yml
  grafana/provisioning/dashboards/dashboards.yml
  grafana/dashboards/triagem.json
reports/latencia.json, reports/latencia.md
scripts/gerar_carga.py
src/triagem/
  __init__.py
  preprocessamento.py                   # normalizar_texto
  dados.py                              # gerar_dataset, carregar_dataset, validar_dataset, dividir
  modelo.py                             # construir_pipeline, treinar, avaliar
  onnx_export.py                        # exportar_onnx, verificar_paridade
  classificadores.py                    # Predicao, SklearnClassificador, OnnxClassificador, carregar_classificador
  artefatos.py                          # nova_versao, salvar/ler metadata, promover
  pipeline.py                           # etapas chamadas pela DAG e pelo CLI
  treinar.py                            # CLI: python -m triagem.treinar
  benchmark.py                          # CLI: python -m triagem.benchmark
  api/app.py, api/schemas.py, api/metricas.py
tests/                                  # espelha src/triagem
Dockerfile
docker-compose.yml
docker-compose.modelo-local.yml         # override: monta ./models/producao na API
docker-compose.airflow.yml
pyproject.toml, uv.lock, .gitignore, .dockerignore, README.md
```

`models/` fica fora do Git (artefatos gerados). `.claude/` e `docs/Resumo_Aulas_Tech_Challenge.md`
também ficam no `.gitignore`: o primeiro é configuração local, o segundo é material de aula.

## 12. Tratamento de erros

| Situação | Comportamento |
|---|---|
| Dataset fora do contrato | `DatasetInvalidoError` com a lista de violações; a task de ingestão falha |
| Paridade ONNX violada | `ParidadeOnnxError`; a versão não recebe `modelo.onnx` e não é promovida |
| Challenger pior que o champion | não promove, registra o motivo em log e no XCom; a DAG termina com sucesso |
| Modelo ausente no startup da API | a API não sobe (erro claro no log); o healthcheck do container falha |
| Payload inválido | 422 com o detalhe do Pydantic; contado como erro no painel |
| Exceção na inferência | 500 com mensagem genérica; log com stack trace; contado como erro |

## 13. Estratégia de testes (pytest, TDD na implementação)

- `test_preprocessamento`: acentos, caixa, espaços, string vazia.
- `test_dados`: gerador determinístico (mesma seed produz o mesmo hash), tamanho, proporções dentro
  de ±3 p.p., esquema; `validar_dataset` rejeita cada violação; split estratificado.
- `test_modelo`: treino em dataset pequeno atinge F1-macro ≥ 0,80; métricas com as chaves esperadas.
- `test_onnx_export`: paridade ≥ 99% e a exceção quando a paridade é forçada a falhar.
- `test_classificadores`: os dois backends dão a mesma classe para os mesmos textos; probabilidades
  somam 1.
- `test_artefatos` / `test_pipeline`: versão, metadata, regras do quality gate (promove, rejeita por
  limiar, rejeita por champion melhor, primeiro modelo sem champion).
- `test_api`: 200 com o esquema correto, 422 para vazio e para texto grande demais, `/health`,
  `/metrics` com os contadores incrementados e o 500 com um classificador fake que lança exceção.
- `test_benchmark`: a função de percentis e a estrutura do relatório, com um classificador fake.
- A DAG é validada no job `dag-check` do CI e numa execução real local (`airflow dags test`).
- Uma fixture de sessão treina um modelo pequeno uma única vez, compartilhado pelos testes.

## 14. Mapeamento para os critérios de avaliação

| Critério (peso) | Entregável |
|---|---|
| Modelagem e Otimização (20%) | `triagem.modelo`, `triagem.onnx_export`, `reports/latencia.md` com sklearn vs. ONNX |
| CI/CD (15%) | `.github/workflows/ci.yml` com 4 jobs verdes no GitHub |
| Orquestração (15%) | `dags/triagem_retreino.py` com 5 tasks e execução local comprovada |
| Monitoramento (20%) | `docker-compose.yml`, dashboard provisionado com 7 painéis e print |
| Documentação (15%) | README (arquitetura em nuvem, execução passo a passo, resultados) e `docs/arquitetura.md` |
| Vídeo STAR (15%) | `docs/roteiro_video_star.md`; a gravação fica com o usuário |

## 15. Fora de escopo (YAGNI)

Deploy real em nuvem, Kubernetes, MLflow/DVC, detecção de drift, autenticação na API, alertas do
Grafana, canary/shadow, modelos transformer. Esses itens aparecem no README como evolução, com
referência às aulas.

## 16. Riscos

- **Operadores ONNX de tokenização:** o regex do tokenizer do skl2onnx pode divergir do sklearn. A
  mitigação é a remoção de acentos antes, `tokenexp` explícito e o teste de paridade bloqueante.
- **Peso da imagem do Airflow no CI:** o job `dag-check` usa cache de camadas e roda em paralelo,
  sem bloquear o `build`.
- **Ganho de latência pequeno:** o relatório reporta os números reais; se preciso, a análise inclui o
  tamanho e o tempo de carga do artefato.

## 17. Git e GitHub

- Repositório novo (`git init -b main`), commits semânticos (`feat:`, `test:`, `ci:`, `docs:`,
  `build:`, `chore:`), pequenos e por tarefa do plano.
- Repositório GitHub **público** `triagem-laudos-mlops` na conta autenticada no `gh`, porque a
  avaliação precisa de acesso. O push acontece depois que os testes locais passam, e o CI verde no
  GitHub é verificado.
