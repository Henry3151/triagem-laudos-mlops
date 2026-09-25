# Roteiro do vídeo STAR (5 minutos)

Preparação antes de gravar:

```bash
docker compose up -d --build                                  # API + Prometheus + Grafana
docker compose -f docker-compose.airflow.yml up -d --build    # Airflow (use AIRFLOW_PORT=8081 se a 8080 estiver ocupada)
```

Deixe abertos: o GitHub (aba Actions com o CI verde), o Swagger (`localhost:8000/docs`), o
Grafana (`localhost:3000`), o Airflow e `reports/latencia.md`.

---

## S — Situation (0:00–0:40)

**Tela:** título do README.

> "Um hospital de referência recebe centenas de laudos por dia. Um achado crítico, como uma
> hemorragia intracraniana ou um pneumotórax, perdido no meio da fila pode significar horas de
> atraso no atendimento. A proposta é um sistema que lê o texto do laudo e o classifica na hora
> como normal, atenção ou urgente, para priorizar a fila."

## T — Task (0:40–1:20)

**Tela:** tabela de critérios do enunciado, ou a seção de sumário do README.

> "O desafio não era só o modelo, mas o ciclo de vida dele em produção: servir por uma API REST em
> Docker com baixa latência; um pipeline de CI no GitHub Actions; retreino orquestrado pelo
> Airflow; monitoramento com Prometheus e Grafana; e pelo menos uma otimização de latência, com
> comparação antes e depois."

## A — Action (1:20–3:30)

1. **Arquitetura (20 s)** — diagrama Mermaid do README.
   > "Um único pacote Python concentra a lógica. O CLI, a DAG e o build Docker chamam as mesmas
   > funções, e a normalização do texto é compartilhada entre treino e serving."
2. **Decisão de nuvem (20 s)** — seção do README.
   > "Escolhi real-time: o laudo urgente precisa subir na fila na hora, e um batch noturno
   > atrasaria horas. Na GCP, seria Cloud Run com uma instância mínima para evitar cold start,
   > deploy via OIDC, sem chaves no repositório."
3. **Modelo e ONNX (30 s)** — `reports/latencia.md`.
   > "TF-IDF com bigramas mais Random Forest, F1-macro de 0,948 e recall de urgente de 0,871. Converti o
   > pipeline inteiro para ONNX, com uma verificação de paridade que bloqueia a promoção se o
   > ONNX divergir. Na conversão descobri que o `sublinear_tf` não é reproduzido pelo conversor,
   > então desliguei, e a paridade ficou exata."
4. **Airflow (30 s)** — grafo da DAG e uma execução com as 5 tasks verdes.
   > "A DAG ingere e valida os dados, treina, avalia, exporta para ONNX e só promove se o novo
   > modelo passar no quality gate e não for pior que o campeão em produção."
5. **CI (20 s)** — aba Actions com os 4 jobs verdes.
   > "A cada push: lint, 74 testes com cobertura acima de 80%, build da imagem com smoke test e
   > validação da DAG num Airflow real."

## R — Result (3:30–5:00)

1. **Demo ao vivo (40 s)**
   - Swagger: `POST /predict` com "Tomografia de crânio. Hematoma subdural agudo com efeito de massa."
     → `urgente`.
   - Terminal: `docker compose --profile carga run --rm carga`.
   - Grafana: os painéis de requisições, a latência P50/P95/P99, a taxa de erro (~5% de payloads
     inválidos de propósito) e as predições por classe.
2. **Latência (30 s)** — tabela do `reports/latencia.md`.
   > "Em processo, o ONNX é 117 vezes mais rápido: 0,11 ms contra 13 ms no P50. Ponta a ponta via
   > HTTP, o ganho é de 7 vezes (2,2 ms contra 15 ms), porque o overhead fixo da requisição passa a
   > dominar: a Lei de Amdahl na prática."
3. **Lições aprendidas (20 s)**
   > "Primeiro: otimização precisa de teste de paridade, porque a conversão ONNX mudou o
   > comportamento do modelo em silêncio. Segundo: a medição importa. Do Windows, o port-forward
   > do Docker somava 40 ms que não existem na nuvem, então o benchmark roda dentro da rede
   > Docker. Terceiro: um dataset sintético permitiu reprodutibilidade total; o próximo passo é
   > trocá-lo por laudos reais anonimizados e monitorar drift."
