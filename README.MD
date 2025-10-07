# SubtForest — Fluxo de Análise de Subtipos Moleculares (TCGA-STAD)

Este repositório implementa um fluxo modular para:
- Preparar dados clínicos e somáticos (SNV),
- Gerar a tabela Gene × Subtipo,
- Treinar e avaliar modelos (Random Forest) com validação cruzada,
- Calcular métricas por painel genético e visualizar resultados.

Todos os passos estão orquestrados no notebook `subtypes_analysis_workflow.ipynb`, utilizando os módulos em `modules/`.

## 1) Preparação do ambiente

### Requisitos
- Python 3.12 ou 3.13 (recomendado 3.13)
- pip

### Passo a passo
1. Crie e ative um ambiente virtual:
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - Linux/Mac:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. (Opcional) Abra o Jupyter Lab/Notebook:
   ```bash
   python -m pip install jupyterlab
   jupyter lab
   ```

## 2) Estrutura do projeto

```
G.SubtForest/
├── datas/
│   ├── Dados STAD clínico e patológico.xlsx - data.csv
│   ├── TCGA-STAD.varscan2_snv - TCGA-STAD.varscan2_snv.csv
│   └── genetic_panels.json
├── modules/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── modeling.py
│   ├── model_evaluation.py
│   ├── panel_manager.py
│   └── visualization.py
├── outputs/
├── results/
├── etapa3-gerando-metricas.ipynb
└── subtypes_analysis_workflow.ipynb
```

## 3) Dados necessários
- `datas/Dados STAD clínico e patológico TCGA doutorado Julio.xlsx - data.csv`: tabela clínica (amostra × subtipo);
- `datas/TCGA-STAD.varscan2_snv - TCGA-STAD.varscan2_snv.csv`: SNVs por amostra;
- `datas/genetic_panels.json`: definição dos painéis genéticos (lista de genes por painel).

Observação importante: o arquivo `genetic_panels.json` foi gerado manualmente e sua construção está detalhada na metodologia do artigo. No código, ele é carregado automaticamente pelo `PanelManager` (caminho padrão `datas/genetic_panels.json`).

## 4) Como executar e obter os resultados (passo a passo)

Abra e rode o notebook `subtypes_analysis_workflow.ipynb` do início ao fim. Ele está organizado nas etapas:

### Etapa 1 — Geração da Tabela Gene × Subtipo
Responsável por: carregar, padronizar e integrar os dados clínicos e somáticos, construindo a matriz de genes por amostra, filtrando casos e gerando a tabela final.
- Módulos usados: `DataLoader`, `Preprocessor`.
- Saída principal:
  - `outputs/gene_subtype_table.csv` — Tabela final com colunas `Sample ID`, `Subtype` e genes binários/contínuos.
- Visualizações exploratórias:
  - `outputs/figures/subtype_distribution.(png|pdf)`
  - `outputs/figures/gene_frequency_top20.(png|pdf)`
  - `outputs/figures/gene_subtype_heatmap.(png|pdf)`

### Etapa 2 — Modelagem e Métricas Globais (K-Fold)
Treina Random Forest com validação cruzada e otimização de hiperparâmetros. Calcula métricas médias e valores SHAP.
- Módulos usados: `RandomForestKFoldRunner` (em `modules/modeling.py`).
- Principais saídas:
  - `outputs/kfold_random_forest_results/rf_test_df_fold{0..9}_results.csv` — Resultados por amostra do conjunto de validação/teste em cada fold.
  - `outputs/metricas_media_kfold_hyper.csv` — Métricas médias (acurácia, F1, precisão, recall) agregadas.
  - `outputs/gene_importance_shap.csv` — Importância SHAP global dos genes.
  - `outputs/gene_importance_shap_per_subtype.csv` — Importância SHAP por subtipo.
  - `outputs/val_indices_per_fold.csv` — Índices de validação utilizados por fold.

### Etapa 3 — Métricas por Painel Genético e Visualizações
Executa o fluxo de avaliação por painel, separando treino/validação e teste conforme a lista de amostras hold-out e gera visualizações.
- Módulos usados: `PanelManager`, `ModelEvaluator`, `Visualization`.
- Pré-requisitos dentro do notebook:
  - `df_final` deve estar carregado (Etapa 1 cria `outputs/gene_subtype_table.csv` e o notebook o carrega);
  - `test_list` é definido na seção de hold-out do notebook.
- O notebook adiciona ao final a seção “Etapa 3 (modular)”, que:
  - Carrega os painéis automaticamente com `PanelManager`;
  - Para cada painel, usa `ModelEvaluator.metrics_from_json(...)` para executar K-Fold e salvar resultados;
  - Gera e salva visualizações de métricas médias (heatmap por classe e barras macro/weighted).

## 5) Onde os resultados são salvos

- Resultados gerais (Etapas 1 e 2):
  - `outputs/gene_subtype_table.csv`
  - `outputs/kfold_random_forest_results/` — CSVs por fold.
  - `outputs/metricas_media_kfold_hyper.csv`
  - `outputs/gene_importance_shap.csv`
  - `outputs/gene_importance_shap_per_subtype.csv`
  - `outputs/val_indices_per_fold.csv`
  - Figuras: `outputs/figures/`

- Resultados por painel (Etapa 3):
  - `outputs/panels_results/<nome_do_painel>/kfold_random_forest_results/` — CSVs por fold do painel.
  - `outputs/panels_results/<nome_do_painel>/figures/avg_report_heatmap.(png|pdf)` — Heatmap de métricas médias por classe.
  - `outputs/panels_results/<nome_do_painel>/figures/avg_metrics_bar.(png|pdf)` — Barras de macro/weighted médio com linha de accuracy.

## 6) Como consultar os resultados de um painel

### Via arquivos CSV
Cada painel possui seus resultados por fold em:
```
outputs/panels_results/<painel>/kfold_random_forest_results/
```
Você pode abrir no Excel ou Python. Exemplo em Python:
```python
import pandas as pd
import os

panel_name = 'panel_imuno'
panel_dir = os.path.join('outputs', 'panels_results', panel_name, 'kfold_random_forest_results')

dfs = []
for i in range(10):
    fp = os.path.join(panel_dir, f'rf_test_df_fold{i}_results.csv')
    if os.path.exists(fp):
        dfs.append(pd.read_csv(fp))

df_panel = pd.concat(dfs, ignore_index=True)
print(df_panel.head())
```

### Via figuras
As figuras são salvas em:
```
outputs/panels_results/<painel>/figures/
```
Arquivos principais:
- `avg_report_heatmap.(png|pdf)` — precision/recall/f1-score médios por classe;
- `avg_metrics_bar.(png|pdf)` — macro/weighted médios com linha de accuracy.

## 7) Personalizações úteis
- Alterar a lista de amostras de teste (`test_list`) no notebook para ajustar o conjunto hold-out.
- Limitar os painéis a avaliar: na célula final, ajuste `selected_panels = [...]` para uma lista específica.
- Se você editar módulos (`modules/*.py`) durante a sessão do Jupyter, reinicie o kernel ou use `importlib.reload` conforme já incluído na célula da Etapa 3.

## 8) Notas sobre `genetic_panels.json`
O arquivo `datas/genetic_panels.json` foi definido manualmente com base na metodologia descrita no artigo. Ele contém listas de genes por painel (por exemplo, `panel_imuno`, `panel_top11`, `panel_top18`, etc.). O `PanelManager` usa, por padrão, este caminho para carregar os painéis. Caso deseje usar outro arquivo de painéis, passe o caminho explicitamente ao inicializar `PanelManager(default_json_path=...)`.

---
Em caso de dúvidas ou erros de execução (por exemplo, funções novas não reconhecidas após alterações nos módulos), reinicie o kernel do Jupyter e execute novamente o notebook.