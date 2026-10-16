# Etapa 3 — Modelagem e Avaliação

Implementa o modelo de Machine Learning e o processo de treinamento e avaliação,
integrados ao pipeline DVC. Parte dos dados preparados pela Etapa 2.

## Fluxo

```
download_raw → preprocess → build_features → train → evaluate
                                               │         │
                                               ▼         ▼
                                      models/model.joblib   metrics/metrics.json
```

| Estágio | Entradas | Saídas |
|---|---|---|
| `train` | `data/prepared/train.csv`, `configs/config.yaml` (`project.random_seed`, `model.params`) | `models/model.joblib` |
| `evaluate` | `models/model.joblib`, `data/prepared/test.csv` | `metrics/metrics.json` |

## Como executar

```bash
poetry install
poetry run dvc repro          # pipeline completo (ou só train/evaluate se o resto estiver atualizado)
poetry run dvc metrics show   # resultados da avaliação
poetry run pytest
poetry run ruff check src tests
```

## Decisões

- **Modelo: `RandomForestClassifier` (Scikit-Learn).** Clássico, adequado a dados
  tabulares, não exige escalonamento e lida bem com a mistura de variáveis numéricas
  e one-hot geradas na Etapa 2.
- **`class_weight: balanced`.** O dataset é desbalanceado (cerca de 15% de sessões
  com compra). O peso evita que o modelo ignore a classe minoritária.
- **Métricas:** accuracy, precision, recall, F1 e ROC-AUC. Como há desbalanceamento,
  F1 e ROC-AUC são as principais; accuracy serve só de referência.
- **Seed fixa:** `project.random_seed` (42), usada em `random_state`.
- **Funções simples, sem classes.** Um único modelo não justifica abstrações;
  cada função tem uma responsabilidade (carregar, separar X/y, construir, treinar,
  salvar, calcular métricas). Treino e avaliação ficam em módulos e estágios separados.
- **Logs** com `utils.logger.get_logger`: caminhos, tamanho dos dados, hiperparâmetros
  e métricas finais.

## Configuração de parâmetros

Os hiperparâmetros ficam em `configs/config.yaml`, seção `model.params`:

```yaml
model:
  params:
    n_estimators: 200
    max_depth: 10
    min_samples_leaf: 2
    class_weight: balanced
```

Alterar um valor e rodar `dvc repro` reexecuta `train` e, em seguida, `evaluate`
(o DVC rastreia `model.params` e `project.random_seed` como `params` do estágio).
`train_model()` também aceita um dicionário `params` para uso programático.

## Resultados (conjunto de teste, 2.441 linhas)

| accuracy | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.866 | 0.548 | 0.840 | 0.663 | 0.925 |

O recall alto com precision menor é consequência do `class_weight: balanced`.

## Arquivos da etapa

| Arquivo | Papel |
|---|---|
| `src/purchase_propensity/models/train.py` | Treino e salvamento do modelo |
| `src/purchase_propensity/models/evaluate.py` | Cálculo e registro das métricas |
| `tests/test_train.py`, `tests/test_evaluate.py` | Testes com dados sintéticos |
| `configs/config.yaml` | `model.params`, caminhos do modelo e das métricas |
| `dvc.yaml` | Estágios `train` e `evaluate` |

## Validação das atividades

| Atividade | Status | Evidência |
|---|---|---|
| Treinar com Scikit-Learn | Feito | `train.py` |
| Escolher modelo clássico | Feito | Random Forest |
| Integrar aos dados do pipeline anterior | Feito | Lê `data/prepared/train.csv` e `test.csv` |
| Seeds fixos | Feito | `random_state=42`; retreino com `dvc repro -f train evaluate` gerou `metrics.json` idêntico |
| Definir o processo de treinamento | Feito | `train()` compõe as funções pequenas |
| Definir as métricas | Feito | `compute_metrics()` |
| Avaliar no conjunto de teste | Feito | `evaluate.py` usa só `test.csv` |
| Registrar os resultados | Feito | `metrics/metrics.json`, via `dvc metrics show` |
| Diferentes configurações/parâmetros | Feito | `model.params`; `dvc status` marca `train` como alterado ao mudar um parâmetro |
| Gerar e armazenar o modelo | Feito | `models/model.joblib` (saída do DVC) |
| Integrar ao estágio do pipeline DVC | Feito | Estágios `train` e `evaluate` no `dvc.yaml` |

Verificações executadas: `pytest` (59 passed), `ruff check` e `ruff format --check`
sem pendências, `dvc repro` completo e segunda execução com
"Data and pipelines are up to date".

## Pendências para outras etapas

- **Etapa 2:** `download_raw` depende de todo o `configs/config.yaml`. Mudar um
  hiperparâmetro invalida esse estágio e o DVC o reexecuta (refaz o download). Os
  estágios seguintes são pulados se o arquivo baixado for idêntico. Restringir a
  dependência às chaves `data.*` evitaria isso.
- **Etapa 2:** `build_features.py`, `preprocessing.py` e `main.py` importam de
  `src.purchase_propensity...`; o restante do pacote usa `purchase_propensity...`.
- **Dependências:** `joblib` é usado diretamente, mas vem como dependência transitiva
  do scikit-learn e não está declarado no `pyproject.toml`.
- **Etapa 4:** MLflow (tracking e Registry) e Docker. O `metrics.json` e o
  `model.joblib` são os pontos de entrada naturais.
