# Iridologia sob Lentes Científicas: Avaliação Crítica com Aprendizado de Máquina

Projeto de análise de iridologia para detecção de Diabetes Mellitus Tipo 2 (DM2) utilizando técnicas de aprendizado de máquina, com ênfase em rigor metodológico e controle de vieses.

> **Referência:** Baseado na metodologia descrita no artigo "Iridologia sob Lentes Científicas: Avaliação Crítica com Aprendizado de Máquina" - UFABC, Programa de Iniciação Científica.

![](img/Figure6.png)

## 📋 Visão Geral

Este projeto implementa um pipeline completo e auditável para:
- Classificação binária (DM2 vs. Controle) a partir de imagens de íris
- Validação cruzada estratificada com separação por pessoa (controle de data leakage)
- Testes de robustez com transformações fotométricas
- Análises locais por região (setores angulares e região pancreática)

## 🏗️ Estrutura do Projeto

```
diabetes-iridology-master/
├── main.py                 # Pipeline principal (ponto de entrada)
├── requirements.txt        # Dependências Python 3.8+
├── README.md               # Este arquivo
├── src/                    # Módulos do projeto
│   ├── __init__.py         # Inicialização do pacote
│   ├── config.py           # Configurações centralizadas
│   ├── segmentation.py     # Segmentação de íris/pupila + validação de pupila
│   ├── normalization.py    # Normalização rubber sheet (Daugman)
│   ├── preprocessing.py    # Transformações fotométricas + realce de vascularização
│   ├── feature_extraction.py # Extração de features
│   ├── classifiers.py      # Classificadores ML
│   ├── metrics.py          # Métricas de avaliação
│   ├── local_analysis.py   # Análise por regiões
│   └── results_generator.py # Geração de relatórios
├── experiments/            # Demonstração e experimentos do PGC II
│   ├── demo_vascularization_pupil.py
│   └── report_experiments.py
├── tests/                  # Testes unitários (unittest)
│   └── test_vascularization_and_pupil.py
├── Data/
│   └── Data/
│       ├── all/            # Todos os dados
│       ├── L_split/        # Apenas olhos esquerdos
│       ├── R_split/        # Apenas olhos direitos
│       ├── personBase/     # Separação por pessoa
│       └── personBase_invert/
├── legacy/                 # Código original (referência)
│   ├── Main.py
│   ├── Classifying.py
│   ├── featureExtraction.py
│   └── pre processing/
├── img/                    # Imagens e figuras
└── docs/                   # Relatório e pôster da IC1 (LaTeX e PDF)
    └── PGC_2/              # Relatório parcial do PGC II (ABNT)
```

## 🚀 Instalação

1. Clone o repositório ou extraia os arquivos

2. Crie um ambiente virtual Python 3.8+:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# ou
source venv/bin/activate  # Linux/Mac
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Baixe e extraia os dados em `Data/Data/`

## 💻 Uso

### Pipeline Completo do Artigo

O código implementa todas as etapas descritas na metodologia do artigo:

1. **Segmentação de íris e pupila** (active contours)
2. **Normalização geométrica** (rubber sheet model de Daugman)
3. **Pré-processamento fotométrico** (CLAHE, histograma, blur)
4. **Extração de features** (pixel, LBP, GLCM/Haralick)
5. **Classificação** (LR, SVM, RF, MLP, AdaBoost)
6. **Validação cruzada estratificada** (K=5 folds)
7. **Análise local por região** (setores 10°, malha pancreática 12×12)

### Análise com Dados Pré-processados (Pickle)

```bash
# Análise em todos os datasets
python main.py --run-all

# Dataset específico
python main.py --dataset personBase --channel gray

# Teste de robustez fotométrica
python main.py --robustness-test --dataset personBase

# Análise local por região
python main.py --local-analysis --eye-side left --dataset personBase
```

### Processamento de Imagens Raw (Novas Imagens)

```bash
# Segmentar e normalizar uma única imagem (visualização)
python main.py --segment-single --image-path ./minha_imagem.jpg

# Processar lote de imagens raw
# Estrutura esperada: raw_images/control/ e raw_images/diabetic/
python main.py --process-raw --raw-path ./raw_images --output-processed ./processed

# Pipeline completo: raw -> segmentação -> normalização -> classificação
python main.py --full-pipeline --raw-path ./raw_images
```

## 📊 Metodologia

### Pipeline do Artigo (Implementação Completa)

```
┌─────────────────────────────────────────────────────────────────┐
│                    IMAGENS RAW DO OLHO                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ SEGMENTAÇÃO (src/segmentation.py)                               │
│ • Detecção de pupila (limiarização + active contours)          │
│ • Detecção de íris (active contours expansivos)                │
│ • Refinamento com canais HSV                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ NORMALIZAÇÃO RUBBER SHEET (src/normalization.py)               │
│ • Modelo de Daugman: mapeamento polar → retangular             │
│ • Saída: 201×720 pixels (radial × angular)                     │
│ • Permite comparação ponto-a-ponto                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ PRÉ-PROCESSAMENTO FOTOMÉTRICO (src/preprocessing.py)           │
│ • Original | Histogram EQ | CLAHE | Gaussian Blur              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ EXTRAÇÃO DE FEATURES (src/feature_extraction.py)               │
│ • Pixel features (baseline)                                     │
│ • LBP (Local Binary Patterns)                                   │
│ • GLCM / Haralick (textura)                                     │
│ • Estatísticas de intensidade                                   │
│ • Gabor / HOG                                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ CLASSIFICAÇÃO (src/classifiers.py)                             │
│ • LR, SVM, RF, MLP, AdaBoost                                   │
│ • Validação cruzada estratificada (K=5)                        │
│ • Separação por pessoa (controle de data leakage)              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ AVALIAÇÃO (src/metrics.py)                                     │
│ • Accuracy, Sensitivity, Specificity, Precision, F1           │
│ • Matriz de confusão                                            │
│ • Desvio padrão por fold                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Esquemas de Particionamento
| Dataset | Descrição |
|---------|-----------|
| `all` | Todos os dados sem separação |
| `L_split` | Apenas olhos esquerdos |
| `R_split` | Apenas olhos direitos |
| `personBase` | **Separação por pessoa** (controle principal) |
| `personBase_invert` | Separação invertida |

### Classificadores
- **LR**: Regressão Logística
- **SVM**: Support Vector Machine
- **RF**: Random Forest
- **MLP**: Multi-Layer Perceptron
- **AdaBoost**: Adaptive Boosting

### Métricas
- Accuracy (acurácia global)
- Sensitivity (sensibilidade/recall)
- Specificity (especificidade)
- Precision (precisão)
- F1-score

### Transformações Fotométricas
- Original (sem transformação)
- Histogram Equalization
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gaussian Blur

### Realce de Vascularização (algoritmos consolidados)

Além de brilho/contraste e equalização, o módulo
[`src/preprocessing.py`](src/preprocessing.py) traz a classe
`VascularizationEnhancer` com três algoritmos estabelecidos em visão
computacional e imagem médica para evidenciar estruturas tubulares/fibrosas
do estroma da íris (vasos, fibras radiais, criptas):

- **Frangi**: realce de vascularização por *vesselness* hessiano multiescala
  (Frangi et al., MICCAI 1998). A resposta vem dos autovalores da matriz
  hessiana e é seletiva para estruturas tubulares.
- **Gabor**: resposta máxima de um banco de filtros de Gabor orientados;
  realça fibras e vasos direcionais (base do reconhecimento de íris de Daugman).
- **Black-hat**: top-hat/black-hat morfológico multiescala; extrai estruturas
  finas claras ou escuras independentemente da iluminação de fundo.
- **Vessel**: combinação ponderada padrão dos mapas (0.6 Frangi + 0.4 black-hat).

Pré-processamento comum: canal verde (padrão na literatura de retina) e CLAHE.
Os realces estão integrados ao `PhotometricTransform`, então entram no pipeline
e nos testes de robustez como qualquer outra transformação.

**Resultados (anel da íris das duas fotografias públicas).** A visibilidade da
vascularização é medida pela razão contraste-ruído (*structure_contrast*, CNR)
e pela energia de cristas (*ridge_energy*). Os valores abaixo são as razões em
relação à melhor linha de base (brilho, contraste ou CLAHE) na Tabela 1 do
relatório do PGC II:

| Método | CNR (vs melhor baseline) | Energia de cristas (vs baseline) |
|--------|--------------------------|----------------------------------|
| Melhor linha de base | 1.0x | 1.0x |
| **Frangi** | **2.02-2.36x** | **5.25-5.33x** |
| Gabor | 1.19-1.69x | 0.28-1.33x |
| Black-hat | 0.98-1.25x | 0.14-0.48x |

O Frangi separa as fibras do fundo cerca de duas vezes melhor que brilho e
contraste e superou a linha de base nos 24 setores angulares avaliados. O banco
de Gabor ganha menos nas fotografias, mas foi o mais estável sob ruído no
fantoma sintético. A avaliação completa está em [`docs/PGC_2/`](docs/PGC_2/).

### Validação de Pupila

O módulo [`src/segmentation.py`](src/segmentation.py) traz `PupilValidator`,
que classifica a geometria pupila-íris a partir de descritores interpretáveis:
diâmetro da pupila, diâmetro da íris, espessura do anel da íris
(`r_iris - r_pupila`), razão pupila/íris e excentricidade (concentricidade dos
centros). A classificação é baseada em faixas fisiológicas (razão pupila/íris
~0.2-0.7) e devolve rótulo (`valid`/`borderline`/`invalid`), pontuação de
qualidade e motivos. Também expõe um vetor de *features* pronto para um
classificador supervisionado.

### Demonstração e experimentos reprodutíveis

[`experiments/demo_vascularization_pupil.py`](experiments/demo_vascularization_pupil.py)
roda os realces e a validação de pupila sobre as imagens reais de íris em
[`docs/imagens/`](docs/imagens/), mede CNR, energia de cristas e entropia
**dentro do anel da íris** e grava em `results/` os painéis comparativos, a
imagem anotada da validação e as tabelas (CSV/Markdown).
[`experiments/report_experiments.py`](experiments/report_experiments.py) roda
os onze experimentos do relatório do PGC II (E0 a E10), regenera as figuras,
as tabelas e as séries de [`docs/PGC_2/`](docs/PGC_2/) e grava os CSVs
completos e um `resumo.json` em `results/report/`. Os números e figuras são
medidos na execução, não fixados no código. Os testes em [`tests/`](tests/)
cobrem forma/intervalo das saídas, a vantagem do Frangi sobre o baseline e as
fronteiras do classificador.

```bash
python experiments/demo_vascularization_pupil.py
python experiments/report_experiments.py   # cerca de 3 minutos
python -m unittest discover -s tests
```

## 📈 Resultados Esperados

Conforme metodologia do artigo, o dataset `personBase` tende a apresentar os melhores resultados por reduzir vazamento de identidade:

| Dataset | Melhor Classificador | Acurácia |
|---------|---------------------|----------|
| personBase | MLP | ~92.36% |
| personBase | LR | ~90.81% |
| personBase | SVM | ~88.24% |

## Relatório e pôster da IC1

A pasta [`docs/`](docs/) tem o material acadêmico da IC1:

- [`relatorio_final_2026_IC.tex`](docs/relatorio_final_2026_IC.tex) e o
  [PDF](docs/relatorio_final_2026_IC.pdf): relatório final.
- [`Relatorio_parcial.tex`](docs/Relatorio_parcial.tex): relatório parcial.
- [`poster_IC_2026.tex`](docs/poster_IC_2026.tex) e o [PDF](docs/poster_IC_2026.pdf): pôster A0.
  A versão de 90 x 120 cm está em [`docs/versao_90x120/`](docs/versao_90x120/).
- [`poster_LEIA-ME.md`](docs/poster_LEIA-ME.md): compilação, validação, ressalvas científicas e
  licenças das imagens.

Os números do relatório são preliminares. A auditoria feita depois no PGC encontrou 73 linhas
duplicadas entre as 196 da base legada, e a avaliação antiga não demonstrava separação por pessoa.
Por isso os valores próximos de 92% precisam ser recalculados.

O trabalho continua em dois repositórios: o PGC em [`vsedrim/Iris`](https://github.com/vsedrim/Iris)
e a IC2 em [`vsedrim/Iris-IC2`](https://github.com/vsedrim/Iris-IC2). A etapa do PGC II sobre
vascularização e pupila foi desenvolvida sobre o pipeline deste repositório e está descrita na
seção seguinte.

## Relatório parcial do PGC II

A pasta [`docs/PGC_2/`](docs/PGC_2/) tem o relatório parcial do PGC II, sobre o realce da
vascularização da íris e a validação geométrica da pupila, no formato da ABNT (classe `abntex2`):

- [`metodologia_vascularizacao_pupila.tex`](docs/PGC_2/metodologia_vascularizacao_pupila.tex) e o
  [PDF](docs/PGC_2/metodologia_vascularizacao_pupila.pdf).
- `figuras/`, `tabelas/` e `dados/`: gerados por
  [`experiments/report_experiments.py`](experiments/report_experiments.py). Para mudar um número,
  reexecute o script em vez de editar as tabelas.

Para compilar, rode `pdflatex metodologia_vascularizacao_pupila.tex` duas vezes dentro de
`docs/PGC_2/`. Os Apêndices A e B trazem o ambiente, as sementes e a configuração completa dos
experimentos. Os tempos de execução (E7) dependem da máquina, então depois de uma nova execução do
script a tabela de tempos pode não bater com os valores citados no texto.

Os experimentos usam as duas fotografias públicas de [`docs/imagens/`](docs/imagens/), porque o
conjunto rotulado para DM2 não estava disponível nesta etapa. Os resultados verificam o
funcionamento dos métodos e não sustentam conclusões clínicas.

## 📁 Dataset

- **88 casos diabéticos** e **108 casos controle**
- Imagens coletadas sob supervisão de oftalmologistas do Hospital Farabi
- Download: [Google Drive](https://drive.google.com/file/d/1y7W84iMXkXcL7pnS-wkN2I5V5VIvZrci/view?usp=sharing)

## 📚 Citação

```bibtex
@inproceedings{iridology-icbme2018,
  author    = {Parsa Moradi and Naghme Nazer and Amirhosein Khasahmadi 
               and Hoda Mohammadzadeh and Hasan Khojasteh Jafari},
  title     = {Discovering Informative Regions in Iris Images to Predict Diabetes},
  booktitle = {25th National and 3rd International Iranian Conference 
               on Biomedical Engineering (ICBME)},
  year      = {2018},
}
```

## 📬 Contato

Para questões sobre o código ou metodologia:
- [Naghme Nazer](mailto:naghme93@gmail.com)
- [Parsa Moradi](mailto:parsa.moradi73@gmail.com)
