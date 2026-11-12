# Classificador Multi-tarefa de Comentários — DistilBERT

Um único modelo que resolve três tarefas de classificação ao mesmo tempo, sobre comentários de redes sociais a respeito de onças-pintadas.

## Resultado

Arquitetura com encoder compartilhado e três cabeças independentes, treinada sobre base anotada de 4.569 comentários em sete dimensões. F1 reportado por tarefa a cada época, com parada antecipada por paciência de 8 épocas.

## O problema

As três perguntas — o comentário fala de onça, é desinformação, é notícia — incidem sobre o mesmo texto. Treinar três modelos separados desperdiça computação e ignora que as tarefas compartilham o mesmo entendimento de linguagem. Um modelo multi-tarefa aprende a representação uma vez e especializa só a saída.

## Como funciona

- **Encoder compartilhado** — `distilbert-base-multilingual-cased`, com o token `[CLS]` do último estado oculto como resumo da sequência.
- **Três cabeças heterogêneas** — camadas lineares independentes, com número de rótulos diferente por tarefa: 3, 2 e 3 classes.
- **Perda somada** — entropia cruzada por tarefa, somada antes do passo de otimização.
- **Estratificação por rótulo combinado** — a divisão treino/validação/teste preserva a distribuição conjunta das três tarefas, não de cada uma isolada.
- **Pesos de classe suavizados** — inverso da raiz quadrada da frequência, em vez do inverso puro, para não penalizar a classe rara a ponto de desestabilizar o treino.
- **Recursos de estabilidade** — recorte de gradiente, agendador linear com aquecimento e parada antecipada.

## Stack

Python · PyTorch · Hugging Face Transformers · scikit-learn · pandas · Matplotlib

## Como rodar

```bash
pip install -r requirements.txt
# coloque oncas_comentarios.csv na raiz
python trabalho_ia2.py
```

## Limitações

- A base anotada não é versionada aqui por envolver conteúdo coletado de terceiros.
- Classes com menos de 7 exemplos na combinação de rótulos são descartadas, o que reduz o conjunto.
- A perda soma as três tarefas com peso igual, sem calibração entre elas.
- Não há comparação medida contra três modelos separados, que seria a linha de base natural.

## Autoria

Pedro Henrique Acioly Amaral — disciplina de Inteligência Artificial, FACOM/UFMS.
