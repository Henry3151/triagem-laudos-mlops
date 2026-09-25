# Tech Challenge — Enunciado Oficial

## Tema Central
Deploy de Modelo em Produção com Pipeline CI/CD, Monitoramento e Otimização de Latência.

## Contexto
Um hospital de referência precisa de um sistema de triagem automática de exames de texto (laudos médicos) para classificar urgência (ex.: normal / atenção / urgente). O modelo central será um classificador de texto (NLP) leve, servido via API REST em container Docker. O foco do projeto é garantir que o ciclo de vida do modelo funcione com um pipeline CI/CD (GitHub Actions), orquestração básica de retreino (Airflow), monitoramento essencial (Prometheus + Grafana) e otimizações de latência.

## Requisitos Obrigatórios

**Repositório GitHub**
- Pipeline CI/CD básico com GitHub Actions (ex.: lint → test → build).
- Script ou DAG Airflow simples para pipeline de treino/retreino.
- Dockerfile funcional para o serviço de inferência.
- Stack de monitoramento local: API + Prometheus + Grafana via Docker Compose.
- Histórico de commits semântico e organizado.

**Vídeo (5 minutos — método STAR)**
- Situation: Problema clínico e importância da triagem rápida.
- Task: Requisitos técnicos da fase (latência, CI/CD, monitoramento).
- Action: Arquitetura escolhida, como o modelo foi otimizado e como a monitoração foi configurada.
- Result: Demonstração do pipeline funcionando, latência alcançada e lições aprendidas.

## Bibliotecas Requeridas
- Scikit-Learn ou framework de preferência — modelo base de classificação de texto (ex.: TF-IDF + Random Forest ou modelo leve similar).
- FastAPI — construção da API.
- Prometheus-client — instrumentação de métricas.
- Airflow — orquestração de tarefas.

## Boas Práticas Obrigatórias
- CI/CD contendo pelo menos 2 automações (ex.: verificação de código e testes).
- DAG Airflow funcional (ex.: carregamento de dados → treino → salvamento do modelo).
- Dashboard Grafana com pelo menos 3 painéis (ex.: total de requisições, latência/tempo de resposta, taxa de erro).
- Otimização de performance: aplicação de pelo menos uma técnica vista em aula (ex.: conversão para ONNX, quantização básica ou pruning).

## Etapas de Desenvolvimento (4 Etapas)

### Etapa 1 — Decisão Arquitetural e API Inicial (Deploy em Nuvem)
- Analisar qual estratégia de deploy em nuvem (AWS, Azure ou GCP) seria ideal (batch vs. real-time) e documentar de forma textual no README.
- Criar API simples com FastAPI que receba o texto do laudo e retorne a classificação.
- Empacotar a API em container Docker e medir o tempo de resposta (baseline de latência local).
- **Entregável:** API funcional rodando em Docker + documento/texto de decisão arquitetural no README.

### Etapa 2 — CI/CD e Pipeline Automatizado (CI/CD e Pipeline de Treino)
- Workflow no GitHub Actions que execute automaticamente testes (pytest) e lint do código no push.
- DAG no Airflow simulando o treinamento (task ler CSV de dados + task treinar e salvar modelo).
- **Entregável:** Workflow YAML + arquivo .py da DAG do Airflow.

### Etapa 3 — Monitoramento e Observabilidade (Monitoração de Performance e Serviços)
- Instrumentar a API (prometheus_client) para expor métricas básicas: tempo de requisição e contagem de chamadas.
- docker-compose.yml subindo API + Prometheus + Grafana juntos.
- Dashboard simples no Grafana para visualizar as métricas.
- **Entregável:** Docker Compose rodando a stack completa + print/JSON do dashboard configurado.

### Etapa 4 — Otimização de Latência e Entrega (Latência em Modelos Não Estruturados)
- Treinar o classificador de texto.
- Aplicar técnica de otimização de latência (ex.: ONNX Runtime ou quantização).
- Comparar latência do modelo original vs. otimizado.
- Gravar o vídeo STAR demonstrando o projeto.
- **Entregável:** Modelo otimizado, resultados comparativos de latência e link do vídeo.

## Critérios de Avaliação

| Critério | Peso | Descrição |
|---|---|---|
| Modelagem e Otimização | 20% | Modelo funcional de NLP, conversão/otimização (ex.: ONNX) bem-sucedida e melhoria de latência demonstrada. |
| CI/CD (GitHub Actions) | 15% | Workflow configurado e rodando testes básicos. |
| Orquestração (Airflow) | 15% | DAG funcional realizando as etapas de ingestão e treino. |
| Monitoramento | 20% | Compose funcional (API + Prometheus + Grafana) com dashboard exibindo as métricas propostas. |
| Documentação (README) | 15% | Explicação da arquitetura em nuvem escolhida e instruções claras de execução. |
| Vídeo STAR | 15% | Clareza na demonstração técnica e explicação do impacto (≤ 5 min). |

## Dataset Sugerido
Datasets públicos simples de classificação de textos médicos ou triagem. Exemplos: Medical Abstracts TC Corpus (Kaggle), recortes do MIMIC-III (open access) ou qualquer dataset tabular contendo uma coluna de texto (sintoma/laudo) e uma coluna de target (classificação/urgência) com pelo menos 2.000 amostras.

## Passo a Passo Resumido
- [Etapa 1] Escolher arquitetura teórica + API FastAPI em Docker + Medir latência base.
- [Etapa 2] Configurar GitHub Actions (lint/test) + Criar DAG Airflow simples de treino.
- [Etapa 3] Configurar Docker Compose com Prometheus e Grafana + Gerar requisições para ver o gráfico.
- [Etapa 4] Aplicar otimização (ex.: converter modelo para ONNX) + Documentar resultados + Gravar o Vídeo.
