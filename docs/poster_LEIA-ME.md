# Pôster de iniciação científica em A0

A versão principal é a **A0 vertical, 84,1 x 118,9 cm**. Ela mantém o logo oficial
vetorial e as verificações de impressão, dentro do limite de 90 x 120 cm da UFABC.
A versão de [90 x 120 cm](versao_90x120/poster_IC_2026.tex) foi preservada sem alterações.

- [poster_IC_2026.tex](poster_IC_2026.tex): fonte principal editável.
- [poster_IC_2026.pdf](poster_IC_2026.pdf): PDF atualizado, com ressalvas científicas explícitas.
- [poster_IC_2026_previa.png](poster_IC_2026_previa.png): prévia gerada a partir do PDF.
- [validar_poster.py](validar_poster.py): compilação opcional, verificações e geração da prévia.

## Revisão científica antes de imprimir

Os valores do [relatório final do IC1](relatorio_final_2026_IC.tex) foram mantidos como
**resultados preliminares reportados**, sem reexecutar os experimentos. A consulta aos dados locais
revelou uma divergência que precisa ser discutida com o orientador:

- A [auditoria posterior do PGC](https://github.com/vsedrim/Iris/blob/main/projeto_codigo/docs/BASELINE_RESULTS.md) afirma que os
  resultados preliminares próximos de 92% precisam ser recalculados após remoção de duplicatas.
- Os metadados da base legada (`projeto_codigo/data/legacy/metadata.json` do PGC, local e fora do Git) registram 196 entradas
  (108 controles e 88 rótulos de diabetes), 73 cópias exatas removidas e 123 imagens únicas
  (66 controles e 57 rótulos de diabetes). Essas contagens **não comprovam o tamanho amostral
  dos experimentos do IC1** nem o número de pessoas distintas.
- Não há identificadores pessoais de origem nem confirmação clínica do subtipo DM2.
  O nome `personBase` não comprova isolamento por pessoa.
- A base está vinculada ao trabalho retratado de Moradi et al. A referência [4] do pôster identifica
  o [aviso de retratação](https://doi.org/10.1109/ICBME45317.2018.10207763), confirmado nos
  [metadados do Crossref](https://api.crossref.org/works/10.1109/ICBME45317.2018.10207763).
  Isso não se refere ao artigo de Samant e Agarwal, citado como [2].

O pôster não apresenta os resultados novos do PGC/IC2 como se fossem do IC1, não publica imagens
da base de licença não esclarecida e não afirma que a validação por pessoa foi comprovada.
**Validação do layout não substitui revisão científica.** Antes da apresentação, conferir com o
orientador a base efetivamente usada, os registros dos experimentos e a redação destas ressalvas.
Também conferir título, centro e autores com o resumo submetido; o título completo do modelo foi preservado.

## Organização visual e gráficos

- Síntese no alto: resultado e limitação aparecem juntos, sem medalhão de acurácia duplicado.
- Cerca de 42% da altura da página é reservada aos resultados e à discussão.
- Fundo branco, verde institucional, texto grafite e poucos destaques em dourado e terracota.
- Introdução, objetivos e fluxo metodológico compactos; os cartões e círculos decorativos foram retirados.
- Comparação global com pontos e hastes de um desvio padrão, não intervalos de confiança.
  Os cinco modelos aparecem; RF tem média de 83,67%, mas seu DP não foi reportado e não foi inventado.
- Robustez com transformações rotuladas e diferenças em pontos percentuais. A queda de 0,62 pp
  corresponde a 92,36% menos 91,74%; não demonstra robustez externa.
- Mapa local ampliado, com índices, escala de cores e destaque apenas para a célula (3,3).
  O pico de 95% entre 144 células é exploratório, sem correção para múltiplas comparações.

Não foi incluído QR code: nenhum endereço público confirmado do IC1 foi identificado.
Um QR futuro deve apontar para material autorizado, sem redistribuir a base legada.

## Requisitos da UFABC

| Item | Verificação |
|---|---|
| Tamanho | Uma página de 841 x 1189 mm, inferior a 900 x 1200 mm |
| Identificação | Título, autores, centro e universidade na parte superior |
| Título completo | Maiúsculas, TeX Gyre Heros Bold de 68 pt; maiúscula de referência com 17,39 mm |
| Unidade | Maiúsculas, 62 pt; maiúscula de referência com 15,85 mm |
| Conteúdo | Introdução, objetivos, materiais e métodos, resultados e discussão, conclusões |
| Agradecimento | Financiamento pelo CNPq explicitado em seção própria |
| Referências | Quatro referências principais, citadas no texto |
| Imagens | Fonte, licença e indicação de adaptação da fotografia; logo oficial da UFABC |

O texto principal usa 29 pt; os gráficos e descrições metodológicas usam principalmente 22 a 30 pt.
Créditos e referências exigem leitura mais próxima. As fontes que imprimem texto estão incorporadas;
gráficos, diagramas e logo são vetoriais. Uma declaração de Times-Roman herdada do logo não imprime
nenhum glifo e não é usada no conteúdo.

## Imagens e autorização de uso

A [macrofotografia](imagens/iris_closeup.jpg) é de **Carlos Andrés Reyes**, via
[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Iris_eclipse_-_Explored_-_Flickr_-_creyesk.jpg),
sob [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/). A licença permite reutilização e
adaptação com atribuição. A [faixa polar](imagens/iris_strip_ilustrativa.png) deriva da mesma imagem,
com remapeamento e ajuste de contraste. Ambas são **ilustrativas**, não imagens de participantes
ou saídas experimentais. Fontes e alterações estão identificadas no pôster e em
[LICENCA_iris_closeup.md](imagens/LICENCA_iris_closeup.md).

O [logo vetorial](imagens/logo_ufabc_vertical.pdf) segue o
[Manual de Identidade Visual da UFABC](https://www.ufabc.edu.br/administracao/sucom/programacao-visual/manual-de-identidade-visual-da-ufabc),
com cores e proporções preservadas. A fotografia grande de Petr Novák não é mais usada;
seus arquivos e licença foram mantidos para preservar as versões anteriores.

## Compilação e validação

No workspace `C:\src`, a tarefa **Conferir e renderizar poster IC1** compila, verifica o PDF e
atualiza a prévia. O equivalente em PowerShell é:

```powershell
& 'C:\src\.venv\Scripts\python.exe' 'C:\src\Brazil\Iris-DM2\IC1\docs\validar_poster.py' --compile
```

O script usa o PyMuPDF instalado nesse ambiente. Confere página única A0, presença das seções e
ressalvas, altura das maiúsculas no log, fontes usadas, texto fora da página e possíveis sobreposições.
A inspeção visual da prévia continua necessária após alterações de layout.

Para compilar somente o PDF, sem Python:

```powershell
Set-Location 'C:\src\Brazil\Iris-DM2\IC1\docs'
$env:PATH = "$env:APPDATA\TinyTeX\bin\windows;$env:PATH"
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error poster_IC_2026.tex
```

O documento usa pdfLaTeX, TeX Gyre Heros, Latin Modern, TikZ e os pacotes já presentes na instalação
TinyTeX. O `latexmk` executa as passagens necessárias para estabilizar âncoras e links. No Overleaf,
incluir o fonte, o logo e as duas imagens do fluxo, mantendo a subpasta `imagens`.
Ao abrir somente a pasta IC1 no VS Code, a tarefa **Compilar poster IC 2026 (A0 vertical)**
continua compilando apenas o PDF, sem atualizar a PNG.

## Impressão

Após a revisão científica, enviar o **PDF em tamanho real, escala 100%**, sem ajustar ao papel.
Não imprimir a PNG. Reduzir o documento pode tornar a altura das letras inferior ao mínimo exigido.
O arquivo não tem sangria nem marcas de corte; uma eventual sangria deve estender apenas os fundos,
sem redimensionar ou cortar conteúdo. Papel comum atende ao uso único; lona é uma opção mais durável
para guardar ou reutilizar, conforme a orientação da universidade.
