# Etapa 4 — MLflow, Model Registry e Docker

Rastreia cada treino no MLflow, registra e promove o melhor modelo no Model
Registry e containeriza o pipeline completo. Parte do `model.joblib` e do
`metrics.json` gerados pela Etapa 3.

## Fluxo

```
train → evaluate → track → register
                     │         │
                     ▼         ▼
     metrics/mlflow_run.json   metrics/registered_model.json
          (run no MLflow)      (versão no Registry + alias @champion)
```

| Estágio | Entradas | Saídas |
|---|---|---|
| `track` | `models/model.joblib`, `metrics/metrics.json`, `data/prepared/test.csv` | run no MLflow, `metrics/mlflow_run.json` |
| `register` | `metrics/mlflow_run.json`, `mlflow.*` do config | versão no Registry, `metrics/registered_model.json` |

## Como executar

```bash
poetry install
poetry run dvc repro
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000

# ou, tudo em containers:
docker compose up -d mlflow
docker compose run --rm pipeline
docker compose --profile serve up -d serve                     # http://localhost:8000
```

## Decisões

- **Tracking como estágio DVC próprio.** `track` lê os artefatos de `train` e
  `evaluate` em vez de alterar esses módulos. Assim cada um continua com uma
  responsabilidade, e o DVC decide quando há uma run nova: só quando modelo ou
  métricas mudam.
- **O que é logado:** todos os hiperparâmetros (`model.get_params()`, incluindo
  defaults), `model_class`, `random_seed` e `test_size`; as cinco métricas; o
  dataset de teste (`mlflow.log_input`, com digest); o modelo com assinatura e
  `input_example`.
- **Serialização skops.** O MLflow 3.17 salva modelos sklearn com skops, mais
  seguro que pickle. A árvore do Random Forest é declarada em
  `SKOPS_TRUSTED_TYPES`.
- **Champion/challenger.** `should_promote()` promove o candidato se não existir
  campeão ou se `candidate > champion + min_improvement`. No empate o campeão se
  mantém, então reexecutar os mesmos parâmetros não cria versões duplicadas. Só
  modelos que melhoram viram versão no Registry.
- **Alias em vez de stages.** O MLflow 3 descontinuou os stages
  `Staging`/`Production`. O modelo promovido recebe o alias `@champion` e o
  serving usa `models:/purchase_propensity_classifier@champion`.
- **Registro pelo LoggedModel.** O `register` usa `models:/<model_id>` lido de
  `run.outputs.model_outputs`, o formato nativo do MLflow 3, em vez de
  `runs:/<id>/model`.
- **Configuração:** nomes de experimento, modelo, alias e métrica ficam em
  `configs/config.yaml` (`mlflow.*`). A URI do servidor muda conforme o ambiente,
  então vem de `MLFLOW_TRACKING_URI` (`.env` ou variável de ambiente).
- **Docker: uma imagem, três serviços.** Servidor, pipeline e serving usam a mesma
  imagem, garantindo a mesma versão do MLflow no cliente e no servidor. A imagem
  instala só os grupos `main` e `pipeline` (sem pytest/ruff/jupyter), roda como
  usuário não root e não leva `data/`, `models/`, `metrics/` nem `.dvc/cache`, então
  `dvc repro` no container sempre executa do download ao Registry.
- **`--allowed-hosts`.** O servidor MLflow 3 valida o header `Host`. O compose
  libera `mlflow:5000` (entre containers) e `localhost:5000` (host).

## Resultados

Experimentos executados com `dvc exp run -S ...` (F1 é a métrica de seleção):

| Run | Parâmetros | F1 | ROC-AUC | Registry |
|---|---|---|---|---|
| base | `max_depth=10`, `n_estimators=200` | 0.663 | 0.925 | v1 → `@champion` |
| exp1 | `max_depth=20`, `n_estimators=300` | **0.679** | 0.926 | v2 → `@champion` |
| exp2 | `max_depth=4`, `n_estimators=300` | 0.613 | 0.900 | não promovido |

No Docker (Linux, Python 3.12) o pipeline do zero reproduziu exatamente as
métricas da execução local (Windows, Python 3.13). A segunda execução manteve o
campeão (empate) e o serving respondeu `POST /invocations` com o modelo `@champion`.

## Arquivos da etapa

| Arquivo | Papel |
|---|---|
| `src/purchase_propensity/models/tracking.py` | Configura o MLflow e loga a run |
| `src/purchase_propensity/models/registry.py` | Política de promoção, registro e alias |
| `tests/test_tracking.py`, `tests/test_registry.py` | Testes com SQLite temporário |
| `configs/config.yaml` | Seção `mlflow` |
| `.env.example` | `MLFLOW_TRACKING_URI` |
| `dvc.yaml` | Estágios `track` e `register` |
| `Dockerfile`, `docker-compose.yml` | Imagem e serviços `mlflow`, `pipeline`, `serve` |

## Validação das atividades

| Atividade (PDF) | Status | Evidência |
|---|---|---|
| Logar parâmetros, métricas e modelo de cada run | Feito | `tracking.py`, estágio `track` |
| Registrar o melhor modelo no Model Registry | Feito | `registry.py`, estágio `register` |
| Modelo promovido no Registry | Feito | alias `@champion` |
| Dockerfile instalando via Poetry e executando o projeto | Feito | `Dockerfile` (`CMD dvc repro`) |
| README com instruções completas | Feito | seções MLflow e Docker |

Verificações executadas: `pytest` (67 passed), `ruff check` e
`ruff format --check` sem pendências, `dvc repro` local e `docker compose run
--rm pipeline` completos.

## Observações

- `download_raw` ainda depende de todo o `configs/config.yaml` (pendência citada
  na Etapa 3). Mudar a seção `mlflow` refaz o download, mas os demais estágios são
  pulados porque o arquivo sai idêntico.
- O exp1 (`max_depth=20`, `n_estimators=300`) teve F1 melhor que os parâmetros
  atuais do `config.yaml`. Adotá-lo como padrão é uma decisão de modelagem do grupo.
