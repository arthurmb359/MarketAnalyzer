# AGENTS.md

Guia de arquitetura e boas práticas do projeto MarketAnalyzer.

Este documento define:

- estrutura do projeto
- responsabilidades de cada módulo
- fluxo de dados e pesquisa
- padrões de código
- como agentes de IA devem modificar o código

Este arquivo deve ser seguido sempre que o projeto for modificado.


# 1. Filosofia do Projeto

O MarketAnalyzer é um sistema de pesquisa quantitativa macro focado em mercados brasileiros, com ênfase em:

- Tesouro IPCA+
- juros reais
- curva DI
- câmbio
- regimes macroeconômicos

O objetivo do projeto é:

pesquisar → validar hipótese → transformar em estratégia → executar backtest

Para isso, o projeto separa claramente:

research
signals
backtests
data loaders

Essa separação é fundamental e nunca deve ser quebrada.

# 2. Estrutura do Projeto

Estrutura principal:

market_analyzer/

app/
    main.py
    registry.py
    ui.py

core/
    metrics.py
    reporting.py
    utils.py
    features.py

data_updater/
    update_config.py
    tesouro_updater.py

markets/
    tesouro_ipca/
        loader.py
        series.py
        signals.py
        backtests.py
        research.py

    usdbrl/
        loader.py
        series.py
        signals.py
        backtests.py

    di/
        loader.py
        series.py
        signals.py
        backtests.py

data/
    tesouro_ipca.csv
    usdbrl.csv

# 3. Camadas do Sistema

O sistema é dividido em 3 camadas principais.

## 3.1 Market Modules

Cada mercado possui seu próprio módulo:

markets/<market_name>/

Exemplos:

markets/tesouro_ipca
markets/usdbrl
markets/di

Cada mercado é autônomo.

Cada módulo pode ter:

loader.py
series.py
signals.py
backtests.py
research.py

Nem todos são obrigatórios.

## 3.2 Core Engine

Contém componentes reutilizáveis:

core/

Inclui:

- métricas
- formatação de relatórios
- funções estatísticas
- utilidades comuns

core/features.py: É a biblioteca de features genéricas.
Aqui entram funções como:
- rolling mean
- rolling std
- z-score
- tendência
- volatilidade
- drawdown rolling

Regra:
- não depende de um mercado específico
- recebe séries/dataframes
- devolve séries/dataframes

Nenhuma lógica específica de mercado deve existir aqui.

core/metrics.py: É a biblioteca de métricas de avaliação.

Aqui entram funções como:
- win rate
- pnl médio
- pnl mediano
- score total
- holding médio
- drawdown máximo
- anos para recuperar drawdown
- média/mediana por trade

Regra: recebe uma lista de trades ou dataframe de trades e devolve números-resumo

core/reporting.py: É a biblioteca de formatação de saída. Não calcula estratégia, só transforma resultado em texto legível.

Aqui entram coisas como:
- formatar datas, moeda, percentuais
- montar blocos de resumo, detalhe de trades, relatório final textual

core/utils.py: É o módulo de utilidades genéricas.

Exemplos:
- validação de colunas
- normalização de nomes
- conversão de datas

core/types.py: Concentra tipos compartilhados, normalmente com dataclasses.

Exemplos de dataclasses úteis:

@dataclass
class TradeResult:
    entry_date: object
    exit_date: object
    entry_price: float
    exit_price: float
    pnl: float
    holding_days: int

@dataclass
class SignalResult:
    signal: bool
    regime: str
    score: float
    explanation: str

## 3.3 App Layer
Responsável pela execução.

app/

Inclui:

main.py
registry.py
ui.py

Essa camada:

- registra algoritmos
- executa backtests
- conecta UI

Não contém lógica quantitativa.



# 4. Responsabilidade de Cada Arquivo
## loader.py

Responsável apenas por:

- ler dados
- normalizar colunas
- converter tipos

Exemplo:

csv → dataframe

Loader não deve calcular indicadores.

## series.py

Constrói séries e features derivadas.

Exemplos:

- rolling mean
- zscore
- volatility
- trend
- duration

Essas funções devem ser determinísticas e reutilizáveis.

## signals.py

Contém interpretação do mercado.

Signals transformam dados em decisões.

Exemplos:

- enter_trade
- exit_trade
- regime classification
- stress detection

Signals podem retornar:

- True / False
- probabilidade
- regime categórico
- score

Signals não executam trades.


## backtests.py

Contém estratégias reproduzíveis.

Backtests simulam:

- entrada
- saída
- PnL
- drawdown

Regras devem ser claras e estáveis.

Backtests não devem conter código exploratório.

## research.py

Laboratório de pesquisa.

Contém:

- experimentos
- análises exploratórias
- estatísticas
- comparações

Research pode:

- usar prints
- gerar gráficos
- retornar dataframes
- testar hipóteses

Research não deve ser usado diretamente pela UI.

# 5. Pipeline de Dados

Fluxo padrão:

loader
↓
series
↓
signals
↓
backtest
↓
report

Research roda fora dessa pipeline.


# 6. Data Conventions

## Tesouro IPCA+

Arquivo principal:

data/tesouro_ipca.csv

Regras:

- contém apenas Tesouro IPCA+
- coluna Prazo_anos deve existir
- data deve ser ordenada

## USD/BRL

Arquivo:

data/usdbrl.csv

Formato original BC:

data;valor

Internamente deve ser normalizado para:

usdbrl


# 7. Padrões de Código

Preferir:

- funções puras
- dataclasses simples

Evitar:

- classes complexas
- estado global
- side effects


# 8. Resultados de Backtests

Backtests devem retornar dados estruturados primeiro.

Formato recomendado:

{
    "summary": {...},
    "trades": [...],
    "config": {...}
}

Depois podem ser convertidos para texto via core/reporting.


# 9. Registry

Algoritmos executáveis pela UI devem ser registrados em:

app/registry.py

Exemplo:

registry.register(
    "IPCA+ State of Art",
    backtest_realrate_state_of_art
)


# 10. Regras para Agentes de IA

Agentes devem:

- modificar código incrementalmente
- preservar compatibilidade
- evitar refatorações grandes de uma vez
- respeitar separação de camadas

Nunca:

- misturar research com backtests
- misturar loader com feature engineering
- duplicar lógica entre mercados


# 11. Fluxo de Pesquisa

Workflow recomendado:

1. research.py
   testar hipótese

2. signals.py
   transformar hipótese em regra

3. backtests.py
   validar estratégia

4. registry
   disponibilizar na UI


# 12. Versões do Sistema

## V1
Exploração inicial testando parâmetros e criando vários hipóteses

## V2
Algoritmo trade no NTN-B/IPCA + consolidado com entradas, saide, escalonamento bem definidos e validados.


## V3 (planejado)

Integração de:
- IPCA+
- DI curve
- USD/BRL
- Macro regimes


# 13. Regra de Ouro

Se surgir dúvida sobre onde colocar código, pergunte:

isso é pesquisa?

→ research.py

isso é decisão de mercado?

→ signals.py

isso é estratégia executável?

→ backtests.py