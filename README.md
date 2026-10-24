# Purchase Propensity Prediction

Uma empresa de e-commerce deseja prever a propensão de compra dos usuários com
base em seu comportamento de navegação.

## Dataset utilizado

O **Online Shoppers Purchasing Intention Dataset** contém 12.330 sessões de
navegação, 17 features e a coluna target `Revenue`, que indica se a sessão
terminou em compra. Entre as features estão páginas visitadas, duração da
navegação, taxas de saída e tipo de visitante. O arquivo CSV tem aproximadamente
1 MB.

- Fonte original e download: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).
- Cópia no [Kaggle](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset).
- Licença informada pela UCI: **Creative Commons Attribution 4.0 (CC BY 4.0)**.
- Referência: Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention
  Dataset*. UCI Machine Learning Repository.
  [DOI: 10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q).

## Boas práticas utilizadas

- **Clean Code e responsabilidade única:** funções pequenas, com leitura,
  limpeza, separação e logging em módulos próprios; `main.py` apenas os compõe.
- **Naming conventions:** funções e variáveis em `snake_case`; constantes em
  `UPPER_CASE`; nomes descritivos em inglês conforme o padrão Python/PEP 8.
- **Type hints:** entradas e retornos tipados nas funções principais, sem `Any`.
- **Docstrings Google Style:** contratos, retornos e erros documentados.
- **Organização modular:** pacote instalável no layout `src/`, sem alterações
  manuais de `sys.path`.
- **Preservação da entrada:** funções de limpeza e separação devolvem cópias e
  preservam o índice, evitando alterações no DataFrame do chamador.
- **Validações claras:** arquivos vazios ou ausentes, target inexistente e nomes
  vazios ou duplicados após normalização geram erros explícitos.
- **Ruff:** verifica estilo, imports, problemas comuns, type hints e docstrings;
  o padrão de docstrings não é exigido nas funções de teste.
- **Testes unitários:** pytest usa arquivos temporários e dados sintéticos
  pequenos, sem depender de um dataset real.
- **Configurações**: Centralizar todas configurações em `configs/config.yaml`.

## Requisitos e instalação

Utilize Python **3.11 ou superior** e Poetry **2.x**. Com ambos disponíveis,
execute na raiz do projeto:

```bash
poetry install
cp .env.example .env   # opcional: define MLFLOW_TRACKING_URI
```

Grupos de dependências do `pyproject.toml`:

- **main (produção):** pandas, numpy, scikit-learn, MLflow, PyYAML, python-dotenv, joblib.
- **pipeline:** DVC, necessário para executar o pipeline (também instalado no Docker).
- **dev:** pytest, Ruff, ipykernel e matplotlib, usados só em desenvolvimento.

## DVC — Versionamento e pipeline de dados

O **DVC (Data Version Control)** é utilizado para orquestrar o pipeline de dados e 
versionar os artefatos gerados em cada etapa.

O pipeline é composto pelas seguintes etapas:

1. **Download:** obtém o dataset original da fonte configurada.
2. **Preprocessamento:** normaliza os nomes das colunas, valida o esquema, normaliza 
valores categóricos e booleanos e remove linhas totalmente vazias e duplicatas exatas.
3. **Preparação dos dados:** cria features derivadas, realiza a separação estratificada 
entre treino e teste e aplica as transformações de features, incluindo imputação e codificação one-hot. 
4. **Treino (`train`):** treina o `RandomForestClassifier` com `model.params` e seed fixa.
5. **Avaliação (`evaluate`):** calcula accuracy, precision, recall, F1 e ROC-AUC no teste.
6. **Tracking (`track`):** registra parâmetros, métricas, dataset de avaliação e modelo
em uma run do MLflow.
7. **Registry (`register`):** registra o modelo no MLflow Model Registry e o promove
ao alias `@champion` se superar o campeão atual.

```
download_raw → preprocess → build_features → train → evaluate → track → register
```

Para reproduzir o pipeline:

```bash
poetry run dvc repro
```

### Armazenamento dos artefatos

Para este projeto acadêmico, foi escolhido um **remote DVC local**, evitando a necessidade de 
configurar uma infraestrutura de armazenamento em nuvem. 

O remote local pode ser configurado com:

```bash
dvc remote add -d local ../dvc-storage
```

Esse comando define `../dvc-storage` como o armazenamento padrão dos artefatos do DVC. Execute-o uma 
única vez na configuração inicial do projeto; o remote fica registrado na configuração do DVC.

## Como executar

### 1. Reproduzir o pipeline de dados

Na raiz do projeto, execute:

```bash
poetry run dvc repro
```

O DVC executará as etapas necessárias conforme as dependências e os arquivos de 
saída declarados em `dvc.yaml`.

### 2. Executar a aplicação

Após preparar os dados, execute:

```bash
poetry run python main.py
```

O `main.py` consome os arquivos preparados pelo DVC e registra suas dimensões no log.
Ele não executa novamente o download, o preprocessamento, a engenharia de features 
ou a separação entre treino e teste.

### 3. Rodar experimentos com outros parâmetros

Cada execução com parâmetros diferentes gera uma nova run no MLflow:

```bash
poetry run dvc exp run -S configs/config.yaml:model.params.max_depth=20
poetry run dvc exp show
```

## MLflow — Tracking e Model Registry

- **Tracking:** cada execução do estágio `track` cria uma run no experimento
  `purchase_propensity` com todos os hiperparâmetros do modelo, a seed, as métricas
  de teste, o dataset de avaliação e o modelo (com assinatura e exemplo de entrada).
- **Model Registry:** o estágio `register` usa a estratégia *champion/challenger*.
  A run nova só vira uma versão de `purchase_propensity_classifier` e recebe o alias
  `@champion` se a métrica `mlflow.selection_metric` (F1) for maior que a do campeão
  atual somada a `mlflow.min_improvement`. A decisão fica em
  `metrics/registered_model.json`.
- **Onde ficam os dados:** sem `.env`, o MLflow usa `sqlite:///mlflow.db` e a pasta
  `mlruns/` na raiz (ambos ignorados pelo Git). A variável `MLFLOW_TRACKING_URI`
  aponta para outro servidor, por exemplo o do Docker (`http://localhost:5000`).

Para abrir a interface localmente:

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Acesse http://localhost:5000 e veja **Experiments** (runs) e **Models** (Registry).

## Docker

O `Dockerfile` instala as dependências `main` e `pipeline` via Poetry, sem o grupo
dev, e executa `dvc repro`. O `docker-compose.yml` usa a mesma imagem em três serviços:

| Serviço | Função | Porta |
|---|---|---|
| `mlflow` | Tracking server e Model Registry (SQLite + artefatos em volume) | 5000 |
| `pipeline` | Executa o pipeline DVC completo e loga no `mlflow` | - |
| `serve` | Serve o modelo `@champion` via REST (perfil `serve`) | 8000 |

```bash
docker compose build
docker compose up -d mlflow             # UI em http://localhost:5000
docker compose run --rm pipeline        # pipeline completo, do download ao Registry
docker compose --profile serve up -d serve
docker compose --profile serve down     # o volume mlflow-data mantém o histórico
```

O `configs/` é montado no container: alterar `config.yaml` no host e rodar
`docker compose run --rm pipeline` de novo gera uma nova run e reavalia o campeão.

Exemplo de predição (colunas de `data/prepared/test.csv`, sem `revenue`):

```bash
curl -X POST http://localhost:8000/invocations   -H "Content-Type: application/json"   -d '{"dataframe_split": {"columns": ["..."], "data": [[...]]}}'
```

## Testes

Para executar a suíte de testes:

```bash
poetry run pytest
```

Os testes utilizam dados sintéticos e arquivos temporários, evitando dependência do 
dataset real ou de serviços externos durante a execução dos testes unitários.

## Verificação do código

```bash
poetry run ruff check .
```

Para verificar também a formatação:

```bash
poetry run ruff format --check .
```

Referências de configuração: [Poetry](https://python-poetry.org/docs/pyproject/)
e [Ruff](https://docs.astral.sh/ruff/configuration/).
