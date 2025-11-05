# Transformer for Depression Detection

O projeto treina um Transformer simples para classificar tweets (depressivos vs. não depressivos) e utiliza metaheurísticas para encontrar uma combinação melhor de hiperparâmetros.

## Pré-requisitos

```bash
python -m pip install -r requirements.txt
```

## Como executar

```bash
python main.py
```

O pipeline executa duas etapas na sequência:

1. **Baseline** – treina o modelo com uma configuração fixa (limitada) e salva `baseline_model.pt` junto com `baseline_report.json`.
2. **Metaheurísticas** – roda Differential Evolution, Particle Swarm Optimization e Simulated Annealing (via SciPy e PySwarms), avaliando várias combinações. A melhor acurácia é re-treinada e salva em `best_model.pt`.

Cada execução cria uma pasta `artifacts/run-YYYY-MM-DD_HH-MM-SS/` contendo:

- `baseline_model.pt` e `baseline_report.json`
- `best_model.pt`
- `meta_results.json` (histórico completo das tentativas com todas as métricas)
- `pipeline_summary.json` (resumo da execução, incluindo o baseline e o resultado das metaheurísticas)

## Estrutura principal

- `src/config.py` – dicionários de configuração usados no baseline e na busca metaheurística.
- `src/training.py` – tokenização, montagem do Transformer e rotina de treino/validação.
- `src/meta_heuristics.py` – integração com SciPy (DE/SA) e PySwarms (PSO), chamando `train_model` a cada avaliação.
- `src/pipeline.py` – orquestra baseline + metaheurísticas e grava os artefatos.
- `main.py` – ponto de entrada (`python main.py`).

Os arquivos de dados limpos (`depressed_tweets.csv` e `non_depressed_tweets.csv`) ficam na pasta `data/`. Para ajustar hiperparâmetros padrão (baseline ou busca), edite os dicionários em `src/config.py`.
