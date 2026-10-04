# PyMHM: retomada com um clone leve, sem transferência de resultados

Estado revisado em **4 de outubro de 2026**. Roteiro operacional, separado dos
docs científicos e excluído dos pacotes distribuídos.

**A implementação integral e a reprodução integral da literatura ainda não estão
concluídas.** Faltam controles de resolução, comparações independentes completas
e alguns drivers de casos. A retomada usa as fontes versionadas e as referências
primárias preservadas em `.tmp`; campos e ambientes são adquiridos novamente.
As quedas interromperam execuções; sua causa não foi estabelecida.

### Imports canônicos da organização atual

A orientação atual do usuário é remover a preservação dos imports anteriores à
refatoração. Use exclusivamente os módulos e submódulos atuais; não reintroduza
`MODULE_ALIASES`, `_compat`, exports físicos ocultos na raiz nem redirecionamentos
de solvers em módulos de malhas/FEM. A API principal da DSL continua na raiz.
Consumidores, notebooks, exemplos, testes e documentação usam os donos atuais.
Pickles novos usam esses donos; caminhos antigos não são uma interface suportada.
As fórmulas, tolerâncias, bases e registros científicos aceitos são preservados.
A evidência desta limpeza está em `build/refactoring/canonical-imports-v1/`.
A etapa está concluída: 5.413 testes portáteis distintos e 250 integrações
DOLFINx aprovados; cobertura de 99,9305% de linhas e 99,6964% de branches.
As 19 comparações preservam os valores/bits significativos de 1.058 arrays.
Lint, formato, tipos, documentação/MathJax e artefatos instalados passaram.
Recibo final: `build/reports/completion/canonical-imports-final-readiness-v1.json`.
Os recibos abaixo descrevem a etapa anterior e não autorizam compatibilidade
de imports; seus hashes e aquisições históricas permanecem imutáveis.

### API variacional independente do modelo físico

**Etapa de engenharia concluída em 4 de outubro de 2026.** A API principal usa
`Equation`, `LocalEquations`, `MultiscaleProblem`, `assemble` e `solve`, com UFL
nativa opcional ou blocos numéricos explícitos. São as mesmas abstrações para
campos escalares, vetoriais e mistos; modelos e variantes são declarações da
aplicação. `NestedEquations` permite vários níveis, e `leaf_moment` compõe gauges
físicos das folhas. D/g também representam formas face-only, sem incógnitas de
volume nem fatorizações locais artificiais. Os operadores de momentos, curl e
Newmark têm donos numéricos compartilhados. Basix continua obrigatório.

Os solvers por física permanecem em `_legacy` como implementações privadas
para comparações e campanhas de aquisição. Os 85 notebooks, organizados por problema,
não têm chamadas físicas legadas nos caminhos primários de cálculo; 24 chamadas
em 15 notebooks são comparações explícitas. As fontes não guardam outputs.
As células de cálculo atuais foram executadas e comparadas com os mesmos inputs;
as execuções completas e os leitores de galerias têm escopos distintos nos recibos.
Arquivos/figuras preexistentes ausentes no clone leve continuam impedindo algumas
execuções integrais. Não readquirir campanhas pesadas só para esconder esse limite.

Os 19 controles finais preservam literalmente 1.058 arrays, incluindo operadores,
bases e campos; o NPZ tem o mesmo digest do baseline. Os 39 registros científicos
e 11 recibos anteriores preservam os bytes e a identidade de execução. Nos 28
históricos do lote maior, 1.692 arrays físicos foram comparados, com diferença
máxima de 1,09×10⁻¹³; outros lotes e os controles de fronteira/spawn têm recibos
próprios. Nenhuma tolerância de solver foi relaxada.

Validação: 5.403 testes portáteis distintos aprovados, 303 skips opcionais;
99,9097% de linhas e 99,6786% de branches. A suíte integral foi seguida pelos
46 testes híbridos/face-only afetados pela expectativa de rejeição de A vazia:
o mesmo input malformado continua rejeitado pela dimensão de B, agora que A 0×0
é permitido. A cobertura acumulada usa os mesmos 178 módulos, sem alteração de
implementação entre as etapas. As integrações nativas DOLFINx aprovaram 250 testes.
Lint, format-check, typecheck, test-cov e docs-check passaram.

O navegador validou 43 páginas/517 expressões, 85 fontes de notebooks/443
expressões e 25 cópias históricas executadas/98 células/180 expressões, sem erros
MathJax ou overflow matemático. As capturas e os painéis atuais foram inspecionados;
os recibos anteriores dos tutoriais e notebooks FEM permanecem associados às
suas fontes. Importar o helper de elasticidade preserva o backend gráfico ativo,
e sua execução CLI seleciona Agg.

Wheel, sdist e Conda 0.1.0 foram construídos e inspecionados. Os 180 arquivos
runtime/tipagem coincidem com a fonte validada. Nos artefatos instalados fora do
checkout, 16 controles compatíveis e 124 testes genéricos passaram, incluindo
hierarquia, face-only e spawn; fatores e workers foram encerrados. O payload
Conda noarch foi extraído em venv isolada com dependências do Pixi lock, sem uma
resolução Conda nova. Windows não foi executado; nenhum artefato foi publicado.

Recibo consolidado:
`build/reports/completion/variational-dsl-final-readiness-v1.json`.
A interface resolve sistemas lineares reais; a numeração de formas UFL globais
e os mapas do esqueleto são declarados explicitamente. Contratos para compiladores
e solvers externos estão implementados, mas integrações ML/preCICE de produção
não foram qualificadas. Reprodução integral da literatura e campanha Brinkman
2D/3D em regimes extremos permanecem etapas científicas separadas.

### Prioridade de engenharia definida pelo usuário

A campanha focada de Brinkman fica para uma etapa posterior. A etapa de engenharia
organiza a arquitetura do PyMHM segundo os padrões de `voids` e `torch-flash`:
operações em funções livres com delegações explícitas; objetos próprios para
problemas, configurações, bases, sistemas e soluções; definição variacional dos
problemas locais e globais aproveitando UFL; providers locais por contratos
pequenos, com adaptadores FEM e futuros métodos externos/ML; execução serial
por macroelemento ou paralela em lotes limitados.

Referências de engenharia fornecidas pelo usuário:
`https://github.com/geomech-project/voids` e
`https://github.com/ThermoPhase-FCSRG/torch-flash`.
A preferência por composição e funções não exige classes exclusivamente de dados:
objetos podem gerir recursos e estados quando isso define uma responsabilidade clara.

Preservar os resultados demonstrados é requisito explícito. Antes de qualquer
mudança no núcleo, foram arquivadas 730 fontes/configurações da árvore de trabalho,
os registros aceitos e o wheel portátil. Baseline:
`build/refactoring/baseline-20261003-v1/manifest.json` e
`executed-project-sources-and-evidence.zip`, no mesmo diretório. Os 117 módulos
do core coincidem com o recibo final da entrega inicial. Campos e seus digests
originais permanecem nos respectivos diretórios; essa captura não recalculou PDEs.

A extração deve preservar operadores, RHS, quadratura, bases executadas,
mapas orientados, gauges, modos retidos e critérios originais. Comparar a montagem,
a reconstrução e os campos por problema antes/depois. Manter os métodos atuais
como delegadores durante a migração; não retaguar as aquisições históricas nem
substituir fórmulas ou tolerâncias para obter concordância. Em paralelo, workers
retornam contribuições por célula e o coordenador reduz em ordem fixa; nenhuma
escrita concorrente nos coeficientes de faces compartilhadas. Recursos FEM/MPI/
PETSc/CUDA são criados e liberados no worker, preservando spawn multiplataforma.

A API principal é variacional e independente do modelo físico. `Equation(a, L)`
declara a equação global; `LocalEquations(a, L, b, c, dofs, d=..., g=...)` declara
os quatro blocos A u+B lambda=f e C u+D lambda=g, com mapas de teste/trial,
orientações, kernels e momentos explícitos. `MultiscaleProblem` reúne essas formas
com um provider; `assemble`, `solve`, `leaf_moment` e `with_global_equation` são
funções livres. Os objetos delegam as operações ao núcleo compartilhado.
`LocalForm`, `GlobalForm`, providers por física e os `solve_*` históricos permanecem
como formulações privadas predefinidas; não são a API introdutória.

As formas podem ser arrays, operadores esparsos, colunas/linhas de formas lineares
ou UFL montada pela integração DOLFINx nativa opcional. Isso inclui formas globais
bilineares/lineares nas coordenadas reduzidas explicitamente declaradas, com
verificação contra montagem monolítica independente. Não há mapeamento automático
entre uma malha UFL global e as bases do esqueleto. Outros compiladores e solvers
locais entram pelos mesmos contratos, sem um dispatcher de PDEs.

Um problema local pode ser outro `MultiscaleProblem`; `NestedEquations` declara
suas restrições e reações de fronteira. A reconstrução recupera todos os níveis,
usando as bases executadas e sem resolver novamente os globais filhos. Locais
sem incógnitas de volume declaram A 0×0 e contribuem diretamente por D/g, como
no MsHHO face-only. Qualificar modelos ML e acoplamentos preCICE permanece
separado do contrato implementado; preCICE não é dependência do core.

Basix é dependência de runtime e fornece as bases/tabulações das famílias
nodais, RT e BDM. O nome histórico `portable` é apenas uma grafia compatível
para a mesma execução Basix. Os mapas de coordenadas preservam os momentos,
restrições normais, bolhas, orientações e bases persistidas dos espaços MHM;
um BDM completo não substitui as restrições H(div) publicadas. Polinômios
Legendre, Bernstein e coordenadas monomiais também usam tabulação nativa.
As integrações DOLFINx/UFL, PETSc e MPI permanecem opcionais. FIAT/FInAT não
são integrações executadas nesta entrega.

Tutoriais introdutórios em `docs/tutorials/{scalar,vector,providers}.md` usam
28 patches escalares e 17 variantes vetoriais, mais providers primais/mistos
com execução serial, threads ou processos spawn em lotes limitados. São exemplos
analíticos pequenos; não substituem estudos de convergência ou reproduções.

Preservação da refatoração estrutural anterior: 19 casos, 1.056 arrays de operadores, bases, soluções,
campos e métricas coincidem literalmente com o baseline. Os 39 registros aceitos
e 11 recibos preservam seus bytes e proveniência. Comparação final:
`build/refactoring/equivalence-v3/equivalence.json`. Os recibos iniciais abaixo
continuam associados à fonte que executaram.

**Refatoração estrutural concluída em 4 de outubro de 2026, antes da migração
integral das tabulações para Basix.** Lint,
format-check, typecheck, test-cov e docs-check passaram na fonte congelada:
**4.540 testes**, 284 skips opcionais, **99,9274% de linhas** e **99,6741% de
branches**. As seleções Python 3.11/3.12 tiveram 208 testes cada; a seleção
DOLFINx/UFL nativa teve 52. O adaptador Basix passou 59 controles em cada
versão executada (0.11.0 e 0.9.0), incluindo ordenação, orientação e replay.

A inspeção de navegador validou 42 páginas, 486 expressões MathJax e 46 imagens,
sem erros ou overflow; seis capturas dos tutoriais e interfaces foram inspecionadas.
Wheel, sdist e Conda 0.1.0 foram construídos e inspecionados: os 120 módulos do
core coincidem com a fonte final. Os pacotes isolados reproduzem literalmente
as montagens serial/spawn em lotes 1 e 2, para providers primais e mistos, sem
importações FEM opcionais ou processos remanescentes. O runtime Windows não
foi executado nesta validação Linux.

Recibo consolidado: `build/reports/completion/engineering-final-readiness-v1.json`.
As limitações da DSL global e das
qualificações externas permanecem explícitas acima. A campanha Brinkman em
2D/3D e regimes extremos continua como a próxima etapa científica autorizada;
não foi reaberta durante esta refatoração.

### Migração das tabulações para Basix

**Substituição concluída em 4 de outubro de 2026.** Basix fornece as tabulações
nodais de intervalos, triângulos, tetraedros e tensores cartesianos, RT/BDM e
polinômios de momentos/traços. Os mapas MHM conservam os espaços restritos,
momentos físicos, bolhas e orientações declarados. As bases novas persistem suas
matrizes executadas e digests; os consumidores de representações históricas
preservam os dados originais. As entradas nodais literalmente coincidentes têm
valores de Kronecker; derivadas e pontos vizinhos usam as tabelas nativas.
A restrição tetraédrica usa o suporte topológico exato da face. A contração
difusiva compartilhada acumula na precisão real mais larga do NumPy antes de
retornar entradas binary64, com as quadraturas e tolerâncias originais.

Os 19 controles finais contêm 1.058 arrays e 54 operadores esparsos, com
diferença relativa máxima de 3,76×10⁻¹⁴ nos operadores. As mudanças mínimas do
suporte esparso e as duas matrizes opcionais de modos retidos estão discriminadas
no recibo. No controle Brinkman com resistência 10⁸, a diferença na norma do erro
de pressão é 1,96×10⁻⁹; isso é preservação do controle de engenharia, não uma
certificação de resolução desse regime. Os 39 registros científicos e 11 recibos
aceitos mantêm seus bytes e proveniência. Foram reproduzidos literalmente 36
campos históricos Three Layers/MsHHO, com uma e duas threads BLAS, sem gerar
bases novas nem resolver PDEs. O replay com matrizes H(div) antigas apresentou
diferença relativa máxima de 4,46×10⁻¹⁴.

Validação global: **4.692 testes distintos aprovados**, 284 skips opcionais;
**99,9386% de linhas** e **99,7274% de branches**. A suíte integral foi seguida
pela repetição dos 20 testes tetraédricos afetados por uma atualização exclusiva
das expectativas de interpolação, com cobertura acumulada e os 121 módulos
de produção idênticos entre as etapas. As seleções Python 3.11/3.12 aprovaram
535 testes distintos cada. Basix/DOLFINx 0.9 aprovaram a seleção nativa de 173
testes, 33 controles tetraédricos suplementares e quatro controles nodais de
graus elevados. Lint, format-check, typecheck e docs-check passaram.

O navegador validou 42 páginas, 486 expressões e 46 imagens, com inspeção das
capturas de interfaces e tutoriais. Wheel, sdist e Conda 0.1.0 foram construídos
e inspecionados: seus 121 módulos coincidem com a fonte validada, e o sdist
contém os 855 arquivos de produção, configuração, testes e documentação
congelados. Os 16 controles dos pacotes instalados fora do checkout aprovaram
providers primais/mistos, contornos Dirichlet/Neumann e execução serial/spawn;
os recursos e processos foram encerrados. O runtime Windows não foi executado.

Recibo consolidado:
`build/reports/completion/basix-migration-final-readiness-v1.json`.
Os recibos anteriores continuam associados às fontes que executaram. A campanha
Brinkman 2D/3D em regimes extremos permanece como próxima etapa científica.

### Organização do código-fonte por responsabilidades

**Organização concluída em 4 de outubro de 2026.** Os módulos estão distribuídos
entre `core`, `fem`, `meshes`, `materials`, `models`, `methods`, `recovery`,
`estimators`, `adaptivity`, `linalg`, `execution`, `backends`, `io` e
`postprocessing`. Geometria, espaços de traço, tabulações, montagem, condensação,
reconstrução e execução têm responsáveis próprios. As famílias Darcy reutilizam
a montagem mista e o saddle de fluxo normal, conservando seus espaços, momentos,
quadraturas e gauges. Os casos científicos e seus drivers permanecem em
`examples/`, notebooks e documentação.

O núcleo usa funções livres com delegações explícitas dos objetos. Os 278
exports públicos e os 120 caminhos históricos de módulos continuam disponíveis;
a camada de compatibilidade resolve os mesmos objetos sem outra implementação.
Basix continua obrigatório para tabulação; integrações FEM/MPI e visualização
continuam opcionais. Há 168 módulos Python e 1.311 definições documentadas e
anotadas, incluindo funções privadas e locais; `source-check` integra o lint.
O desmembramento em responsabilidades não reduz a contagem total de arquivos.
Arquitetura e convenções atuais: `docs/architecture.md`.

Contra o baseline desta organização, os 19 controles com 1.058 arrays e 54
operadores esparsos coincidem **bit a bit**. Nove controles adicionais de Darcy
preservam literalmente 207 arrays, incluindo contornos homogêneos,
não homogêneos e gauge Neumann. Dez objetos históricos preservam 438 arrays
de estado, orientação, condensação e reconstrução. Foram reproduzidos 36 campos
históricos com uma e duas threads BLAS, sem bases novas nem solves de PDEs.
Os 39 registros científicos e 11 recibos mantêm seus bytes originais.
Os 73 notebooks versionados preservam Markdown e metadados; 27 células mudam
apenas imports. Eles não contêm outputs ou attachments salvos, e esta conferência
não é uma reexecução dos notebooks.

Validação: **5.148 testes distintos aprovados**, 284 skips opcionais;
**99,9180% de linhas** e **99,7087% de branches**. A suíte integral foi seguida
pela repetição dos 41 testes do arquivo cuja simulação de alteração de fonte
foi atualizada para o responsável atual, com cobertura acumulada e todos os 170
arquivos de implementação/tipagem idênticos entre as etapas. Lint, format-check,
typecheck e o gate test-cov passaram. A suíte FEM nativa aprovou **237 testes**;
quatro skips de coleta referem-se a módulos de gráficos sem Matplotlib.
`docs-check` passou, e o navegador validou 42 páginas, 486 expressões MathJax e
46 imagens, com 16 capturas inspecionadas.

Wheel, sdist e Conda 0.1.0 foram construídos e inspecionados: os 170 arquivos
de implementação e tipagem têm nomes e bytes exatos. Os 16 controles instalados
fora do checkout preservam literalmente os campos serial/spawn, com recursos
e processos encerrados. A execução foi Linux/Python 3.13, com dependências
bloqueadas; o payload Conda foi instalado num ambiente isolado com essas
dependências, sem uma resolução Conda independente. Windows não foi executado.
Nenhum artefato foi publicado.

Recibo consolidado:
`build/reports/completion/package-layout-final-readiness-v1.json`.
As fontes executadas e os recibos anteriores preservam suas identidades.
Esta organização mantém as capacidades verificadas; a reprodução integral da
literatura e a campanha Brinkman em regimes extremos continuam como trabalho
científico posterior.

### Exemplos em notebooks organizados por problema

**Organização concluída em 4 de outubro de 2026.** Os 73 notebooks existentes
estão em subpastas de `darcy`, `flow`, `elasticity`, `transport`, `waves`,
`foundations` e `convergence`. Seus nomes e IDs históricos foram preservados;
as 322 células científicas mantêm o AST computacional, com imports e bootstrap
ajustados para os novos caminhos. O conteúdo científico em Markdown, metadados,
contagens de execução e entradas foram preservados; links relativos foram
atualizados. Os originais não tinham outputs salvos.

Dez exemplos introdutórios adicionais apresentam as 28 variantes escalares,
17 vetoriais e providers locais/globais, incluindo UFL. Os notebooks mostram
chamadas explícitas das APIs, parâmetros e convenções de cada método. Todos
os dez foram executados, com 34 células de código; as 45 variantes e oito
chamadas representativas têm normas físicas **bit a bit** iguais às rotinas
existentes. Três controles homogêneos/incompressíveis também coincidem.
Todas as células do provider UFL foram executadas no Pixi FEM, com erro nodal
máximo de pressão 1,78×10⁻¹⁵ e resíduo global original 1,11×10⁻¹⁶.

Os exemplos didáticos atuais e futuros são notebooks, conforme `AGENTS.md` e
`CONTRIBUTING.md`. Rotinas Python de dados, campanhas, leitura e workers
continuam reutilizáveis, com seus 224 entry points auxiliares compatíveis.
Os 305 módulos de apoio e os 170 arquivos
do runtime mantêm os mesmos bytes do pacote anterior, assim como os 39 registros
científicos e 11 recibos. A campanha Brinkman permanece separada desta entrega.

Índice dos 83 notebooks: `notebooks/README.md`; métodos e caminhos:
`notebooks/catalogue.json`. O executor descobre subpastas e aceita grupos,
caminhos e IDs históricos; as cópias executadas preservam a hierarquia em
`build/notebooks`. Nomes ambíguos requerem caminho qualificado. O inventário de
arquivos continua exigindo os campos e figuras efetivamente consumidos.

```bash
pixi run --locked -e notebooks notebooks-run darcy/primal_galerkin.ipynb
pixi run --locked -e notebooks notebooks-run flow/introductory_methods.ipynb
pixi run --locked -e notebooks notebooks-run 01
```

Validação: lint, format-check, typecheck, test-cov e docs-check passaram;
**5.176 testes aprovados**, 290 skips opcionais, **99,9180% de linhas** e
**99,7087% de branches**. O executor/inventário aprovou 42 testes no ambiente
de notebooks, incluindo kernel real, PNG e paths nested/externos. Oito notebooks
históricos pequenos foram executados; o 07 depende da figura local ausente
`docs/figures/elasticity/refinement.svg`. Os demais estudos com dados grandes
não foram reexecutados nesta organização.

O navegador verificou 42 páginas, 486 expressões MathJax e 46 imagens da
documentação; os dez notebooks executados também foram exportados e inspecionados.
Wheel, sdist e Conda 0.1.0 foram construídos e inspecionados: o sdist preserva
literalmente os 83 notebooks e seus dois índices; o wheel contém apenas o pacote.
Os 16 controles dos artefatos instalados preservam os campos serial/spawn e
encerram fatores e processos. Runtime Windows não foi executado; nada publicado.

Recibo consolidado:
`build/reports/completion/notebook-layout-final-readiness-v1.json`.

### Entrega reduzida autorizada em 3 de outubro

O escopo corrente é uma entrega inicial **até hoje**, conforme a solicitação do
usuário: estudos de convergência curtos para as famílias de casos, reutilizando
aquisições válidas e completando séries pequenas quando viável. Cada estudo
declara o parâmetro refinado, os campos e normas, os resíduos das equações
originais, a quadratura, as bases executadas e as limitações. Três níveis
constituem uma observação inicial; não certificam automaticamente a faixa
assintótica, a estabilidade uniforme ou uma referência numérica resolvida.

As campanhas extensas, a reprodução integral das tabelas, as referências ainda
não resolvidas, a galeria integral e os notebooks dependentes desses campos
continuam neste roteiro como trabalho posterior. Não são critérios de fechamento
da entrega reduzida. Nenhuma tolerância numérica ou cobertura foi relaxada.
Casos que excedem os orçamentos curtos têm sua limitação registrada, sem substituir
silenciosamente o domínio ou os espaços publicados.

A reconstrução unfitted r64 foi interrompida no próximo checkpoint por essa
mudança de escopo. Seus arquivos parciais foram preservados e não têm aceitação
física final. Recibo: `r64-user-requested-interruption-v1/receipt.json` em
`build/reports/resume/unfitted-source-generation-d5eb89fa1606-v2/`.

### Fechamento da entrega reduzida

**Entrega reduzida concluída em 3 de outubro de 2026.** O catálogo corrente
contém **31 grupos** de observações iniciais e o renderizador público gera
**84 painéis** apenas dos registros JSON, sem uma nova solução das PDEs.
Índice: `docs/cases/minimal-convergence.md`; dados:
`examples/results/minimal-convergence/catalogue.json`. As cinco comparações
detalhadas selecionadas continuam em `docs/cases/index.md`.

O notebook `notebooks/convergence/73_initial_convergence.ipynb` foi executado e validou os
31 registros e as figuras; saída em `build/notebooks/convergence/73_initial_convergence.ipynb`.
Reprodução somente dos gráficos:

```bash
pixi run --locked -e notebooks python examples/plot_initial_convergence.py \
  --output build/initial-convergence-figures
```

Gates finais: lint, format-check, typecheck, test-cov e docs-check passaram;
**4194 testes**, 279 skips opcionais; **99,9136% de linhas** e **99,6419% de
branches**, sem relaxar critérios. Integrações nativas adicionais: **30 testes**
DOLFINx/Basix. Documentação: 39 HTML, **370 expressões MathJax** verificadas no
navegador e 31 figuras do catálogo carregadas. Wheel, sdist e Conda 0.1.0 foram
construídos e inspecionados; os 117 módulos do core são idênticos à fonte
executada naquele fechamento, anterior à refatoração de engenharia.
Os códigos privados de comparação e arquivos grandes de campos permanecem
fora dos artefatos distribuídos.

Recibo consolidado: `build/reports/completion/initial-delivery-final-receipt.json`;
plano atualizado: `build/reports/completion/workplan.json`. As aquisições mantêm
seus UUIDs, fontes executadas e bases originais; consumidores e aliases públicos
têm identidades próprias. O H(div) 3D fecha 20 comparações nativas e os replays
na publicação atual (`hdiv3d-current-publication-final-receipt.json`).

Pendências científicas preservadas: fluxo SPE10 e controles H1 periódicos ainda
não resolvidos; divergência crescente na camada interna Oseen; mixed-well com
dois níveis; SPE10 Brinkman com um nível; HPC4e original adiado. Nanoguide é um
controle DG independente, Three layers é temporal em malha espacial fixa e
Marmousi usa o recorte explicitado. Reprodução integral das tabelas, referências
refinadas, todos os notebooks/figuras dependentes e campanhas de desempenho
continuam como trabalho posterior, fora do fechamento solicitado.

### Estado da retomada neste host

Pixi está instalado em `/prj/thermophase/volpatto/.pixi/bin/pixi`; se não estiver
no PATH, usar esse executável. Instalar e executar com o lockfile existente.
Os ambientes `test`, `test-py311`, `test-py312`, `notebooks`, `docs`, `packaging`,
`fem` e `intel` foram instalados com `--locked`.

- Os gates finais da entrega inicial estão no fechamento acima. As suítes
  completas anteriores Python 3.11 e 3.12 verificaram o mesmo núcleo de 117
  fontes com 3969 testes e 278 skips; regressões novas dos consumidores foram
  verificadas nos ambientes correspondentes quando introduzidas. A suíte final
  padrão contém os novos controles e registra 4194 testes e 279 skips.
- Integrações executadas: DOLFINx para os controles unfitted; PARDISO em serial
  e spawn; comparação periódica independente nos **64 macros**, Q1/r32 e P0
  em 1/2/4/8/16 segmentos. As diferenças relativas de pressão, gradiente bruto
  e fluxo físico são no máximo **5,07×10⁻¹³**, com normas nativas q8/q10.
  O controle completo r64/s32 também executou: 270400 pressões, 4608 traços;
  normas físicas dos campos em fases concordam em até **3,51×10⁻¹²** relativos.
- Controle periódico completo r128/s32 independente: **1.069.632 incógnitas**;
  todas as 64 matrizes de acoplamento coincidem byte a byte. O solve nativo
  passou o critério original após uma correção sobre as mesmas equações,
  de 7,04×10⁻¹⁰ para 2,70×10⁻¹⁴. A aquisição faseada atual dos seis traços,
  UUID `ad352997-c845-4301-b6e6-070414b3c38a`, passou com resíduos originais
  **2,714–2,800×10⁻¹⁴**, após uma correção com A/B/f inalterados. O confronto
  físico r128/s32 concorda em pressão, gradiente e fluxo até **1,43×10⁻¹²**.
  Replay com duas threads e bases recém-calculadas de orientação equivalente
  reproduziu exatamente os digests dos seis campos. Aquisição: 25m34s/643 MiB;
  replay: 5m47s/515 MiB. Isso verifica a aritmética e o contrato de replay;
  resolução local, estabilidade e curva publicada ainda exigem os controles
  r256/r512 e referências refinadas. O resumo consolidado está em
  `build/reports/resume/periodic-acceptance-stream-r128-verification.json`.
- Periódico r256: os seis traços passaram o limite original 10⁻¹⁰, com
  resíduos relativos **1,128–1,165×10⁻¹³** após uma correção das mesmas
  equações. UUID `5a6eeb7e-b8f8-4471-9d8b-7ac2dd3077e3`; aquisição
  **1h50m12s/2,520 GiB**. Replay com duas threads e 192 remontagens de bases
  equivalentes reproduziu exatamente campos/trace/coarse/kernel em
  **22m28s/1,93 GiB**. O incremento r128→r256 permanece **18,95–19,13% no
  H1 quebrado** e **3,63–3,71% em L2**, com r256 como denominador: a resolução
  local ainda não está resolvida. A árvore imutável das 121 fontes executadas
  foi restaurada e verificada para os controles independentes dessa geração;
  os recibos não recebem os hashes de código posterior. O confronto nativo
  r256/s32 completo, com **4.231.744 incógnitas/37.994.560 não zeros**, passou:
  resíduo original **1,13×10⁻¹³**, pressão/gradiente/fluxo físico relativos
  até **2,97×10⁻¹¹**, estáveis em q8/q10. As 64 matrizes B são byte a byte
  iguais. Execução **8m16s/13,59 GiB**, usando somente as fontes imutáveis
  da aquisição; nenhuma fórmula ou tolerância depende da curva publicada.
  r512, resolução local e referências refinadas permanecem pendentes.
  Evidências: `periodic-acceptance-stream-r256-verification.json`,
  `periodic-local-r128-r256.json` e
  `periodic-source-generation-r256-final-restore.json` e
  `periodic-acceptance-stream-r256-final-verification.json`, em
  `build/reports/resume/`.
- A janela coordenada posterior ao r256 integrou quatro owners: restauração
  das matrizes executadas de faces oscilatórias Helmholtz, tabulação cardinal
  somente de valores, protocolo de fonte triangular separável e preparação
  dos vetores espaciais originais no stepper elástico. Os 91 testes Helmholtz
  e 165 testes das fontes/callers passaram em cada ambiente Python; os owners
  isolados têm cobertura de linhas e ramos 100%, com duas integrações FEM reais.
  Lint, format-check, typecheck e o gate completo de cobertura passaram.
  Fonte espacial preparada conserva unidades e bases e
  avalia a amplitude nos tempos originais; não implementa restart. Registros
  em `build/reports/completion/core-integration-window-20261003/`.
- Referência periódica conforming Q1: os níveis atuais 256, 512, 1024, 2048
  e 4096 passaram o limite original de 10⁻¹⁰. O nível histórico 4096 tem
  16.777.216 células e 16.785.409 nós; execução **7m09s/11,58 GiB**, resíduo
  original **5,41×10⁻¹³**. Os incrementos H1 completo 512→1024, 1024→2048
  e 2048→4096 são **34,49%, 19,12% e 9,77%**, com o nível mais fino como
  denominador: 4096 ainda não demonstra resolução do erro espacial. A
  sensibilidade q4→q6 no nível 512 é 9,23×10⁻¹⁰ em H1. A coleção e as fontes
  realmente executadas estão em `periodic-literal-q1-current-verification.json`,
  em `build/reports/resume/`. O confronto Basix independente Q1/512 passou:
  pressão/gradiente/fluxo físico relativos até **2,03×10⁻¹²**, normas q8/q10
  estáveis. Evidência: `periodic-literal-q1-512-native.json` no mesmo diretório.
  Q1/4096 não é solução exata; referência Q5 refinada permanece pendente.
- Aquisição periódica em fases está implementada e validada: contribuições
  compactas, reconstrução em arquivos temporários, publicação atômica, bases
  executadas e guardas de configuração/fontes. Os pilotos de replay estão em
  `build/results/periodic-guarded-r32` e `periodic-guarded-r64`. Replay com
  threads 2/base recém-calculada rotacionada é exato em r32; em r64, mantém
  traço/coarse/kernel exatos e pressão até 1,89×10⁻¹⁷. O incremento local
  r32→r64 é **52,58–53,13% no H1 quebrado**, com r64 como denominador, nos
  mesmos traços s1…16: o refino local continua insuficiente. O resíduo original
  local r64 relativo só à carga volumétrica chega a 1,94×10⁻¹⁰; a escala com
  tração dá 3,95×10⁻¹². São métricas distintas, sem relaxar critérios.
- L10: `--assembly-order` e `--norm-orders` estão disponíveis. O controle P8/r16,
  P2/s32 em q11/q13 foi executado nos **16 macros**. A diferença de campos em
  norma de gradiente é 1,1015×10⁻¹⁰, estável entre integração q13/q15. P3/s32
  nessa mesma malha tem núcleo exato do multiplicador e é rejeitado em ambas
  quadraturas; a receita de varredura completa usa r32.
- Obstáculo quadrado: 16 aquisições públicas P1/RT0, oito confrontos Basix P1
  q6/q8 e oito NeoPZ RT0 executaram nos 200 macros. Referências clássicas
  r4/r8/r16/r32 passaram nas equações físicas livres, energia, gauge e replay
  da base nativa executada. O incremento r16→r32 é 0,512606% em pressão e
  1,72552% em fluxo; a sensibilidade ainda exige refino antes de uma alegação
  de precisão. Os registros e figuras do obstáculo usam os campos atuais.
  Os seis casos pontuais completos P2/RT0, com 2048 macros, foram adquiridos
  com campos/bases executadas e passaram as equações originais. As seis montagens
  Basix independentes passaram o mesmo critério de campos, sem alterar limites;
  diferenças relativas máximas de fluxo P2/RT0: 1,71×10⁻¹¹/7,27×10⁻¹².
  A referência usa uma política aritmética uniforme sobre o operador original,
  com controle adicional de estabilidade independente do candidato. Treze figuras
  atuais do caso foram inspecionadas. A série Green completa de seis níveis
  executou em 5m08s/387 MiB, com resíduo original máximo 3,53×10⁻¹⁴. O notebook
  22 executou seus quatro blocos de código, 35 arquivos/101,6 MB e três figuras,
  em 11,4s/386 MiB; relatório `quarter-notebook-current-verification.json`.
- Marmousi: os dois SEG-Y primários foram adquiridos, verificados por SHA e
  carregados no recorte declarado em unidades SI. O piloto de domínio reduzido
  não é reprodução da tabela nem referência refinada. As campanhas completas
  MHM e clássicas permanecem pendentes.
- O inventário de notebooks filtra a seleção antes de ler manifestos e verifica
  campos e figuras antes do kernel. Onze notebooks executaram com suas figuras;
  campos ausentes ainda impedem os notebooks de campanhas. O notebook 22 está
  restrito às três figuras centrais e verifica bases completas dos campos pontuais.
- MsHHO: a campanha atual de cinco níveis e quatro contrastes foi adquirida
  e renderizada, com bases/R/E/coeficientes executados arquivados. A política
  explícita usa armazenamento local estendido e duas correções nas mesmas
  equações MHM, mantendo 10⁻¹⁰ e o padrão público de mínimo zero. A diferença
  nodal relativa máxima é 4,36×10⁻¹³; os resíduos físicos originais próprios
  são no máximo 5,35×10⁻¹². A montagem Basix independente nos oito macros
  aprova pressão/gradiente/fluxo em 10⁻⁹, incluindo Dirichlet homogêneo em
  κ=10⁶. A inserção cruzada com dados afins em κ=10⁶ permanece reprovada:
  cerca de 5,19×10⁻⁹ nos dois sentidos, acima de 10⁻¹⁰. A ação da diferença
  dos operadores arredondados explica esse resultado; a certificação discreta
  conjunta permanece pendente. Não é reprodução dos espaços locais exatos
  de L06 nem prova de inf-sup uniforme. Registros públicos em
  `examples/results/mshho.json` e `examples/results/mshho/native-verification.json`.
  O notebook 24 executa os registros e as duas figuras atuais.
- Wheel/sdist, Conda noarch e metadados foram reconstruídos e inspecionados na
  geração MsHHO atual. As 117 fontes nos três artefatos coincidem com as verificadas;
  a instalação limpa executa os mesmos campos em modo direto e `spawn`, sem
  importar FEM/CAD/MPI/aceleradores opcionais. Evidência:
  `mshho-core-integration-window-20261003/closed.json` em `build/reports/completion/`.
  `docs-check` ainda falha por **71 assets selecionados ausentes**, de 175. A auditoria
  matemática e a renderização MathJax passaram separadamente; isso não substitui
  o gate estrito da galeria. Ainda não se pode declarar a entrega pronta.

Evidências desta execução: `build/reports/resume/` e `build/reports/completion/`. Nenhum commit, push ou
publicação de pacote foi executado. Continuar pelos controles de P1 abaixo,
preservando a distinção entre aquisição atual, registros históricos e reprodução
literal da literatura.

### Fechamentos e contratos de campos nesta retomada

- Nested Q2/P1: os dez casos homogêneos e afins foram adquiridos, comparados
  integralmente com Basix e publicados com bases executadas e replay bit a bit
  em 1/2 threads. Figuras atuais e notebook 36 executado; referência em
  `examples/results/nested-current/native-verification.json`.
- MsHHO3D: dez casos P2/P0 em cinco resoluções, tetraedros e cubos, com
  fontes projetadas e bases executadas. O confronto independente inteiro
  tem diferença física máxima de 1,327e-14 e inserção nas equações nativas
  de 2,559e-14. A caracterização ideal quadrática aplica-se apenas aos casos
  K=I/P0 selecionados. Figuras atuais e registro em
  `examples/results/core-extensions/mshho3d-current/native-verification.json`.
  O consumidor de seções verifica a mesma norma q12 publicada, mantendo q10
  como controle. Seus 28 testes passaram nos três ambientes Python.
- H(div) 3D: os 20 casos analíticos tetraédricos/prismáticos em n1…5 foram
  adquiridos com C/T/DOFs, coeficientes e equações originais efetivamente
  executados. O confronto independente Basix dos 20 casos passou: diferenças
  relativas máximas em pressão/fluxo/divergência de
  5,029e-14/3,952e-13/6,412e-13. Resíduos originais completos próprios/nativos
  são no máximo 5,435e-13/1,533e-12. Cinco aquisições nativas aceitas foram
  preservadas com suas gerações originais; apenas as 15 restantes foram
  calculadas. Replay nativo em 1/2 threads e rotações coerentes das bolhas
  passaram nos 20 casos. Recibo em
  `build/reports/completion/hdiv3d-native-whole20-v2-final-receipt.json`;
  os 2845 artefatos foram novamente verificados na revisão root.
  O replay adicional dos arquivos próprios e as seções/figuras atuais ainda
  estão em execução/preparação; o notebook 57 também depende da elasticidade.
  O consumidor de seções literal passou 48 testes nos três ambientes Python,
  preservando as saídas MsHHO3D. Erros físicos L2 por campo são separados dos
  resíduos algébricos por bloco; estes não são renomeados como normas L2 físicas.
- Unfitted: comparação dos 16 macros r32 nas duas bases efetivamente
  executadas passou em normas físicas, máximo 3,977e-11. A inserção nodal
  direta nas equações nativas continua reprovada: limite inferior global
  conservador de 1,402e-10, acima de 1e-10. Condensação r64 dos 16 macros
  fechou em 2h47m18,5s/7,217GiB. Os sistemas reduzidos de 2704/3600 incógnitas
  passaram em 10,49s/655,01MiB, com resíduos 1,518e-15/2,684e-15. A reconstrução
  física dos 16 macros está em execução na geração imutável de 145 fontes,
  sob guarda de 8 GiB/28800 s. O controle volumétrico inicial ficou em
  1,077e-10/1,079e-10 e acionou o refinamento compartilhado das mesmas
  equações, com mínimo zero/máximo dois e limite 1e-10 inalterados. A aceitação
  original completa, o confronto independente e a publicação permanecem pendentes.
- Três camadas: ambas as trajetórias de 301 estados passaram as equações
  originais. A comparação de volume arquivada usa uma interpretação comum
  Basix após bijeção nodal e não certifica avaliação nas duas bases próprias;
  os registros públicos explicitam esse limite. A referência clássica h8
  tem uma nova aquisição de 301 estados com a matriz Basix literal e DOFs
  executados, geração `afd43c6b731c`: 424,44 s/1,950 GiB, resíduo original
  máximo 1,185e-13; os coeficientes são byte a byte iguais à geração anterior.
  Replay próprio em 1/2 threads e q8/q10 passou. Isso ainda não é referência
  espacial refinada: as malhas h4/h2 têm apenas controles estruturais, sem
  trajetórias; a viabilidade do solve h2 continua pendente. O produtor MHM
  da receita Polynomial efetiva e o consumidor entre malhas estão preparados,
  mas os 341 macros/301 estados próprios ainda exigem aquisição atual. Os
  recibos antigos permanecem imutáveis, sem novos hashes.
- Seis rasters primários selecionados foram extraídos diretamente dos PDFs
  locais, com SHA/página/recorte/DPI e inspeção. Não substituem as 71 imagens
  científicas ausentes do perfil canônico; o gate estrito de docs continua
  reprovado. A inspeção suplementar MsHHO3D não altera esse perfil.

## 1. Git e preparação no destino

Versionar fontes, testes, infraestrutura, Markdown, notebooks fonte, registros
científicos JSON/CSV e entradas compactas. A seleção atual contém **1913 arquivos, cerca de 32 MiB**.
Os três layers SPE10 NPZ são entradas pequenas, comuns do Git, sem filtro LFS.
Resultados de campo, figuras renderizadas, respostas locais, checkpoints, logs,
caches, builds, ambientes e ferramentas externas ficam ignorados. Não usar
`git add -f` para incluí-los. Não há arquivo de resultados para download.

A limpeza removeu caches/bytecode regeneráveis e preservou os dados científicos
locais. Os registros leves preservam observações anteriores; **não demonstram que
o código atual foi reexecutado no novo host**. `.git` local contém capturas internas
e caches antigos: não copiar esse diretório nem usar `git push --mirror`. Um clone
normal da branch não precisa dessas referências.

Nenhum staging, commit, push ou transferência foi executado nesta preparação.
Antes dela, somente `LICENSE` estava versionado. Revisar na origem:

```bash
git status --short
git add --dry-run .
git check-ignore docs/figures/helmholtz/stability-errors.png
git check-ignore examples/results/marmousi/mhm-H20-ell1.npz
git check-attr filter examples/results/spe10/layer-36.npz
```

Campos/figuras devem estar ignorados; o layer deve ter filtro `unspecified`.
Após revisar e commitar, obter a branch com este roteiro no destino:

```bash
mkdir -p /prj/thermophase/volpatto/Work
cd /prj/thermophase/volpatto/Work
git clone https://github.com/volpatto/pymhm.git pymhm
cd pymhm
git status --short
git rev-parse HEAD
pixi install --locked -e test
pixi install --locked -e notebooks
pixi install --locked -e docs
pixi install --locked -e packaging
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
mkdir -p build/reports/resume build/results
pixi run -e test lint
pixi run -e test format-check
pixi run -e test typecheck
pixi run -e test test-cov > build/reports/resume/test-cov.log 2>&1
```

Se o checkout já existir, atualizar a branch preservando mudanças locais. O remoto
precisa conter o novo commit antes do clone. Não copiar `.pixi` nem atualizar o
lockfile implicitamente. Instalar `fem`, `intel`, `meshing` e `remeshing` com
`--locked` quando necessários; consultar plataformas em [pixi.toml](pixi.toml).
`fem` fornece DOLFINx/Basix/PETSc/MPI, `intel` PARDISO, `remeshing` FreeFem/BAMG.
GPU/AMG acelerado e escalabilidade permanecem pendentes e ficam para depois.

Usar `tmux` ou o gerenciador de jobs. Logs em `build/reports/resume/`, nunca na
raiz; acompanhar com `tail -F build/reports/resume/test-cov.log`. Começar com um
worker e uma thread nativa; medir memória/tempo em um piloto do mesmo operador
antes de aumentar resolução. Os caps da máquina anterior não valem como garantia
para o novo host. Não iniciar várias campanhas grandes ao mesmo tempo.

## 2. Galeria enxuta e controle de armazenamento

[docs/publication-core.json](docs/publication-core.json) seleciona **28 casos
centrais e o índice**, com até três figuras por caso. As fontes complementares
permanecem no Git, mas não são copiadas ao site padrão. [mkdocs.yml](mkdocs.yml)
exclui também os assets fora da seleção e a página de desempenho, cuja execução
fica adiada. Isso não remove testes nem capacidades implementadas.

Conservar por família: comparação de campos/componentes, convergência/controle
de refino, e perfil ou comparação publicada quando trouxer evidência distinta.
Tabelas pequenas registram normas, equilíbrio, quadratura, espaços e incremento
da referência. Não regenerar todos os painéis diagnósticos antigos.

Na nova máquina, reduzir o custo sem mudar o problema matemático:

1. Separar aquisição, condensação, reconstrução, normas e plots. Persistir
   atomicamente coeficientes finais, malha/material, base executada com hash,
   configuração e manifestos. Só usar checkpoints da mesma nova aquisição.
2. Calcular normas entre níveis antes de descartar intermediários volumosos.
   Conservar a referência final, resultados compactos e contratos de replay.
3. Não salvar todas as respostas locais automaticamente. Para Marmousi, implementar
   duas passagens: condensar/descartar respostas, resolver global, remontar locais
   para reconstruir. Testar igualdade de operador/RHS/base/campos em piloto.
4. Para transientes, salvar estados de restart e tempos de observação; acumular
   normas/energia por streaming. Não guardar todas as variantes completas em RAM.
5. Publicar somente registros leves e figuras selecionadas; campos volumosos
   continuam ignorados. Não anunciar restart/controle de RAM ainda inexistente.

O periódico já dispõe dessas fases. Os demais drivers monolíticos precisam
dessas melhorias antes de suas maiores execuções.

### Atenção aos resumos que fazem o driver pular cálculos

O clone contém JSONs históricos leves. `verify_tensor_rt`, `verify_polygons` e
alguns drivers RAD reutilizam linhas existentes; `solve_spe10` precisa de `--force`
para recalcular. **Antes de executar uma família**, identificar seu JSON de saída
no código, preservar uma cópia em `build/previous-summaries/<familia>/` e remover
somente esse registro gerado do caminho de aquisição, ou usar um novo `--output`
quando disponível. Depois substituir o registro público pelo resultado novo.
Não remover manifestos de datasets, entradas, digitalizações ou curvas publicadas.
Não fazer uma limpeza indiscriminada de todos os JSONs.

## 3. Entradas reconstruíveis sem transferência

### SPE10 Model 2

Model 2 tem 60×220×85 células; os casos usam lâminas 2D. Model 1 não fornece essas
entradas. Três layers compactos já estarão no clone. Para reconstruí-los com
revisão OPM e SHA verificados pelo pacote, sem renderizar o volume inteiro:

```bash
pixi run -e notebooks python - <<'PY'
from examples.plot_spe10_data import CACHE, save_layers
from pymhm.reservoir import download_spe10_model2, load_spe10_model2
download_spe10_model2(CACHE)
save_layers(load_spe10_model2(CACHE))
PY
```

`pixi run -e notebooks python examples/plot_spe10_data.py --download` também gera
plots PyVista. Conferir ordem Fortran, orientação, mD/ft, porosidade e componente
Kx/Ky/Kz usada em cada caso; não substituir material para ajustar uma figura.

### Marmousi II

URLs primárias e hashes em [examples/marmousi_data.py](examples/marmousi_data.py).
Download explícito, com streaming e verificação antes de renomear:

```bash
pixi run -e test python - <<'PY'
from pathlib import Path
from urllib.request import urlopen
import hashlib
import os
from examples.marmousi_data import FILES, load_marmousi_crop

directory = Path('build/datasets/marmousi')
directory.mkdir(parents=True, exist_ok=True)
for filename, url, expected in FILES.values():
    target = directory / filename
    if target.exists():
        with target.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
        continue
    temporary = target.with_suffix(target.suffix + '.download')
    digest = hashlib.sha256()
    with urlopen(url, timeout=120) as source, temporary.open('wb') as destination:
        while block := source.read(1024 * 1024):
            digest.update(block)
            destination.write(block)
        destination.flush()
        os.fsync(destination.fileno())
    assert digest.hexdigest() == expected
    temporary.replace(target)
print(load_marmousi_crop(directory).provenance)
PY
```

Formato IBM float, 13601×2801 amostras, spacing 1,25 m. O crop selecionado tem
origem (3395,515) m e 2048×512 pixels de 5 m. Os arrays históricos da publicação
não foram identificados; isso limita a afirmação de reprodução literal.

### HPC4E e ondas de 2017

O caso estático HPC4E usa download revisionado Labmec/MHM pelo helper
[examples/hpc4e_data.py](examples/hpc4e_data.py). Preservar no Git a entrada leve
`examples/results/hpc4e/dataset.json`. Não confundir esse caso com ondas de 2017.

Para as três camadas dinâmicas foram preservados apenas
[case.json](examples/data/three-layer-2017/case.json),
[macro-mesh.json](examples/data/three-layer-2017/macro-mesh.json) e
[horizons.csv](examples/data/three-layer-2017/horizons.csv): 341 macros, interfaces
digitalizadas, materiais, unidades, fonte selecionada e hashes. Não contêm código
externo. Permitem reconstruir o **caso selecionado**, mas não identificam os dados
históricos ausentes. Ainda faltam drivers de aquisição e comparação.

### Artigos e referências externas

[docs/literature.md](docs/literature.md) e [docs/roadmap.md](docs/roadmap.md)
registram fontes/DOIs, formulações e limites. **Consultar primeiro `.tmp`: ali
estão as referências principais de conhecimento**, não somente backups de
resultados. Os PDFs ficam em `.tmp/refs/mhm/literature/`; o estudo periódico
inclui `ParValVerf.pdf`, L08 é `chaumontfrelet_paredes_valentin_2022a.pdf` e L10
é `1-s2.0-S0898122126000192-main.pdf`. Conferir versão, SHA e páginas contra os
relatórios de auditoria em `build/reports/resume/`. Reobter pela fonte/DOI somente
o material ausente. O núcleo portátil não depende dessa biblioteca.

MSL (`msl_mhm`, `msl_cg`, `msl_core`), `msl_mfem`, `mhm-mfem` e os códigos de
Santiago não acompanham o clone versionado. Neste host, cópias de fontes MSL e
comparadores privados estão preservadas em `.tmp`, sem compilados. As cópias
MSL não têm Git próprio: seus commits upstream são registros históricos **não
verificados**; os hashes atuais das 728 fontes conferem com o manifesto de
migração. Não usar `git -C` que encontre o Git do PyMHM como revisão externa.
MSL exige reconstruir core→cg→mhm→driver; caso a fonte esteja ausente, obter acesso.
NeoPZ/MHM/iMRS têm fontes públicas Labmec. Obter código externo em árvore
ignorada, por exemplo `build/reference-sources/`, fixar revisão/URL e manter
comparadores fora da distribuição. Inspeção não conta como execução. Sem acesso
a MSL executável, usar uma montagem independente DOLFINx/UFL, Basix ou NeoPZ do
mesmo caso completo, declarando o limite de acesso. O comparador periódico
`.tmp/comparisons/periodic-ufl/compare.py` já executou neste host. O wrapper
`.tmp/validation/unfitted-msl-smooth/run.py` tem CLI portátil, mas depende do
binário ausente `reference_export`; seus locais P1 são controles independentes,
não a discretização P8 da campanha.

### Figuras extraídas das publicações

As figuras originais dos artigos também estão ignoradas, incluindo, por exemplo,
`docs/figures/mh2m-heterogeneous/crisscross/published-figure-7.png` e
`docs/figures/mh2m-heterogeneous/crisscross/published-figure-8.png`. Não haverá transferência de thumbnails, PNGs ou SVGs.
Os solvers e renderers numéricos não recriam esses recortes do artigo.

Reobter o PDF pela fonte/DOI registrado e refazer cada extração necessária ao
perfil. Registrar URL/DOI, versão e SHA-256 do PDF, página, número da figura,
coordenadas do recorte, resolução de rasterização e SHA-256 da imagem final.
Conservar as digitalizações JSON/CSV leves já presentes no Git. Evitar recortes
que removam eixos, legendas ou informações necessárias à comparação. O gate de
assets e cada plot que incorpora uma figura publicada dependem também dessa
aquisição; resolver essa entrada antes de declarar a galeria completa.

## 4. Ordem de execução e lacunas reais

| Prioridade | Trabalho | Critério de conclusão |
|---|---|---|
| P0 | Ambiente, testes atuais, entradas, perfil enxuto | Clone instala sem campos antigos; gates e manifestos atuais |
| P1 | Periódico: replay; L10: quadratura/resolução local | Operadores e campos reproduzidos; controles locais/clássicos resolvidos |
| P2 | Helmholtz angular/estabilidade; Marmousi | Estudos publicados e baseline refinado, limites históricos explícitos |
| P3 | Novos drivers: obstáculo, Maxwell DG, três camadas, L11 §5.4 | Mesmo caso completo em duas montagens + refino espaço/tempo/quadratura |
| P4 | SPE10, wells, 3D, HPC4E e demais famílias | Correções propagadas; comparação completa independente |
| Depois | MPI/GPU/AMG e escalabilidade | Campanhas de desempenho em problemas com custo mensurável |

A revisão estática conferiu 91 comandos de drivers, 21 chamadas de tasks e a
sintaxe dos três blocos Python: arquivos, flags, escolhas, argumentos, ambientes,
imports locais e nomes de tasks. Não houve inconsistências após as correções.
Isso não certifica todas as campanhas. Os controles executados neste host estão
listados no estado da retomada. Executar um grupo por vez. Valores pequenos são pilotos,
não substitutos do artigo. Comparações devem usar arquivos produzidos na nova
aquisição, não JSONs antigos sem seus campos.

## 5. Periódico e interfaces internas

### L04 periódico

```bash
for stage in condense solve reconstruct; do
  pixi run --locked -e test python examples/verify_periodic.py --macro 8 --local-refinement 32 --segments 1 2 4 8 16 --workers 1 --native-threads 1 --stage "$stage" --artifacts build/results/periodic-pilot-r32
done
pixi run --locked -e test python examples/verify_periodic.py --macro 8 --local-refinement 64 --segments 1 2 4 8 16 32 --workers 1 --native-threads 1 --stage all --reference-levels 32 --artifacts build/results/periodic-pilot-r64 --output build/reports/resume/periodic-pilot-r64.json
pixi run -e test python examples/periodic_reference.py --sizes 32 64 --order 10 --native-threads 1
```

Fases e replay estão implementados; repetir o controle em r128/r256/r512 antes
da campanha final: Q1 local/P0 por segmento, 64 macros. Q1/r32 com 32 segmentos
admite um traço alternado que anula todas as integrações locais sob Dirichlet
fraco; não é par admissível. Para testar os seis traços em piloto, usar r64.
O replay deve usar a base/constraints arquivadas mesmo após uma rotação equivalente
da base recém-calculada. Conferir invariância por threads, campos e orientação.
Referência 32 só evita uma montagem grande simultânea; não é baseline final.
O incremento Q5/32→64 medido neste host é **57,03% em H1**: está sub-resolvido,
sem certificado de precisão. Os campos clássicos finais continuam pendentes.

Clássico Q5: executar separadamente `periodic_reference.py --sizes 512`, depois
1024 e, conforme incremento, 2048, mantendo `--order 10 --native-threads 1`.
Q5/1024 tem 26 224 641 DOFs; Q5/2048 tem 104 878 081: preflight obrigatório.
Após o refino da referência:

```bash
pixi run -e test python examples/compare_periodic.py --reference 5:2048:10:lor --compare-references 5:512:10:lor 5:1024:10:lor --refinements 128 256 512 --macro 8 --segments 1 2 4 8 16 32 --refresh --primary
pixi run -e notebooks python examples/plot_periodic.py --reference 5:2048:10:lor
```

Se o nível suficiente for outro, usar identificadores realmente resolvidos e
declarar isso. O artigo usa Q1/4096²: reprodução literal separada com
`verify_periodic.py --reference-only --reference-levels 512 1024 2048 4096
--reference-degree 1 --reference-assembly lor --reference-solver pyamg`. Q5 não
é a referência histórica. Verificar weak form, equilíbrio e normas físicas/H1
na partição comum, além do residual CSR.

### L08/L10 interfaces, h/p e contraste

```bash
pixi run -e notebooks python -m examples.unfitted_campaign --collect --refinement 16
pixi run -e notebooks python -m examples.unfitted_campaign --collect --refinement 16 --fit-locals
pixi run -e intel python -m examples.unfitted_convergence --study smooth --refinement 32 --degree 8 --maximum-segments 32 --workers 1 --local-solver pypardiso --assembly-order 13 --norm-orders 13 15
pixi run -e intel python -m examples.unfitted_convergence --study contrast --refinement 16 --workers 1 --local-solver pypardiso --contrasts 10 100 1000 10000 100000 1000000
```

Repetir local r24/r32/r64 mantendo dados e traços. Comparar r32→r64 com
`python -m examples.unfitted_local_resolution first second --order 13 --output ...`.
O CLI mantém o padrão `degree+3` (P8 monta q11); a flag `--assembly-order 13`
controla a montagem e `--norm-orders 13 15` integra os erros independentemente.
As convenções e validações de quadratura são centralizadas no assembler Darcy.
O índice q de regularidade em L10 satisfaz 0≤q≤ell e **não** é a ordem de Gauss.
O par P8/r16 com P3/s32 tem posto local 383/384 e modo global exato do
multiplicador; q11/q13 não o removem. r32 com esse traço tem posto completo.
O controle q11/q13 de ell2-s32 em r16 já foi adquirido; ainda falta o incremento
local r32→r64 e a comparação independente completa de L10. Integração exata de K não
cria salto de gradiente no triângulo polinomial. Recriar montagem independente
do caso completo. A discrepância da figura S2 não foi reconciliada.

## 6. Helmholtz e Marmousi

```bash
pixi run -e test python -m examples.helmholtz_campaign --resolutions 8 12 --angles 17 --workers 1
pixi run -e test python -m examples.helmholtz_article --studies direction convergence local-control --workers 1
pixi run -e test python -m examples.helmholtz_stability --ell 1 --frequency 75 --sizes 32 --workers 1
```

A segunda chamada contém os 256 ângulos; a primeira é piloto. Estabilidade:
refazer (ell,frequência)=(0,10),(0,20),(1,15),(1,75), pontos admissíveis publicados.
Para f75: `--sizes 32 48 64 96 128 192 256 384 512`. Preservar ressonâncias e
rejeições matemáticas. Recriar oráculos nativos completos angular/convergência/
Hankel. `plot_helmholtz_stability` lê os quatro novos JSONs, exigindo níveis finais
pelo menos 128, 256, 128 e 512, respectivamente. `plot_helmholtz_article` também
exige `published-convergence.json`, `local-refinement-eight.json` e
`native-convergence-verification.json`, além de `article.json`. Atualizá-los após
comparar; o plot não faz solve. Depois das quatro sequências completas:

```bash
pixi run -e notebooks python -m examples.plot_helmholtz_stability
```

Marmousi: criar piloto nativo menor, mesmo operador e crop explicitamente declarado.
P1 abaixo já contém mais de um milhão de pixels, não é piloto leve. Depois medir
fatoração/memória e executar P1/P2/P3/P4 em chamadas separadas:

```bash
pixi run -e fem mpiexec -n 4 python -m examples.marmousi_reference --data build/datasets/marmousi --degree 1
pixi run -e fem mpiexec -n 4 python -m examples.marmousi_reference --data build/datasets/marmousi --degree 2
pixi run -e fem mpiexec -n 4 python -m examples.marmousi_reference --data build/datasets/marmousi --degree 3
pixi run -e fem mpiexec -n 4 python -m examples.marmousi_reference --data build/datasets/marmousi --degree 4
```

Reorganizar armazenamento/fases da família MHM antes da execução grande. CLI atual:

```bash
pixi run -e intel python -m examples.marmousi_trace_family --data build/datasets/marmousi --H 20 --workers 1 --solver pypardiso --local-solver pypardiso --local-refinement-precision double --response-store build/results/marmousi/responses-H20-double --response-batch-size 128
```

H20/H40/H80 cobrem Q3, local r8/r16/r32 e ell0…4. H40/H80 precisam do controle
`extended`, com novo store identificado se ainda persistir respostas. Não
reutilizar double quando falhar o critério local. Medir baseline P3→P4, normas e
quadratura. Excluir raio 50 m da Dirac em ambos numeradores/denominadores declarados.
O bloco real é equivalente à equação complexa; fatoração simétrica indefinida
não implica SPD.

Após aquisição: `marmousi_fields.py records... --data ...`,
`marmousi_comparison.py candidate reference --data ... --workers 1 --output ...`;
depois `marmousi_sample_archives.py`, `plot_marmousi_family.py` e
`plot_marmousi_mhm.py`. Consultar `--help`, usando novos records. Tabela 6.1
continua não literal enquanto arrays/diagnóstico históricos faltarem.

## 7. SPE10, PGMHM e Unusual

### Darcy layer 36

```bash
pixi run -e test python -m examples.solve_spe10 --force --refinements 40 60 80 100 120 --segments 32 --degree 1 --order 5 --backend serial --workers 1
pixi run -e intel python -m examples.solve_spe10_reference --shape 120 220 --order 5 --solver pypardiso --threads 1
pixi run -e intel python -m examples.solve_spe10_reference --shape 240 440 --order 6 --solver pypardiso --threads 1
pixi run -e intel python -m examples.solve_spe10_reference --shape 480 880 --order 6 --solver pypardiso --threads 1
```

Continuar Q3 960×1760 se incremento exigir e orçamento permitir. Manter Q1 MHM/
r120/32 segmentos C0-P1 publicado. Recriar confronto MSL/NeoPZ completo e integração
das diferenças Q3 se renderer só ler records antigos. Publicar qx/qy/vetores/
perfis além da magnitude; manter material e unidades do artigo.

### Brinkman layer 1 / Taylor–Hood

```bash
pixi run -e test python examples/solve_spe10_brinkman.py --nx 6 --ny 11 --refinement 10 --segments 10 --degree 3 --order 12 --stabilization pointwise-2017 --backend serial --workers 1
pixi run -e fem python -m examples.solve_spe10_taylor_hood --patch --shape 4 4 --threads 1
pixi run -e fem python -m examples.solve_spe10_taylor_hood --shape 60 220 --threads 1
pixi run -e fem python -m examples.solve_spe10_taylor_hood --shape 120 440 --threads 1 --previous build/results/spe10/taylor-hood/taylor-hood-60x220.npz
```

Continuar 240×880 e maiores conforme incremento/memória; comparar `--mhm` com NPZ
novo e `--comparison-orders 24 32`. `--previous` recebe o arquivo de coeficientes, não as amostras da galeria. Manter
USFEM P3/P3 do artigo; TH é baseline clássico, não solução exata. Integração e
constante inversa históricas parcialmente ausentes precisam ser qualificadas.

### Adaptativo L09

```bash
pixi run -e intel python -m examples.spe10_adaptive_aligned --resolutions 15 30 60 120 240 --solver pypardiso-symmetric-matching --native-threads 1
pixi run -e remeshing python -m examples.solve_spe10_published --levels 7 --target-cells 3786 --workers 1 --executable FreeFem++
```

Conferir P2/r2/P0, RT2/Oswald, quatro componentes do indicador publicado, métrica
BAMG e `k>=ell+d`. Nova remalhagem pode mudar conectividade. Clássico 480×1760,
se necessário, em chamada separada `spe10_adaptive_aligned --resolutions 480
--previous-archive examples/results/spe10-adaptive/reference-rt2-240x880.npz`
com flags de solver acima. `complete_spe10_reference.py` completa normas de campo
novo; `solve_spe10_resolution`/`compare_spe10_resolution` são controles separados.
Recriar sistema UFL independente completo inicial/final antes de `plot_spe10_native`: plot não monta
referência. Não chamar controles enriquecidos de espaço publicado.

### PGMHM / Unusual

```bash
pixi run -e test python -m examples.pgmhm_campaign --resolutions 2 4 8 16 32 --segments 1 2 4 8 16 --workers 1
pixi run -e test python -m examples.solve_pgmhm_inclusions --factors 1 2 4 --segments 2 --workers 1
pixi run -e fem python -m examples.solve_pgmhm_inclusions_reference --patch
pixi run -e fem python -m examples.solve_pgmhm_inclusions_reference --graded --factors 1 2 3
pixi run -e intel python -m examples.solve_pgmhm_spe10 reference --component kx --nx 60 120 240 480 --solver pypardiso-symmetric-matching --native-threads 1
pixi run -e test python -m examples.solve_pgmhm_spe10 mhm --component kx --refinement 64 --segments 64 128 --material-fitted --alpha 0.1 --order 5 --workers 1
pixi run -e test python examples/solve_unusual.py --smoke
pixi run -e test python examples/solve_unusual.py
pixi run -e fem python -m examples.solve_unusual_spe10_reference --levels 1 2 4 --graded --threads 1
pixi run -e test python -m examples.solve_unusual_spe10 --refinement 64 --segments 32 64 --layer-resolution 0.25 --trace-fitted --max-local-cells 400000 --workers 1 --order 5
```

Rodar refino clássico em etapas conforme RAM. `solve_pgmhm_spe10 compare --mhm
... --reference ... --workers 1`, `compare_pgmhm_inclusions` e
`compare_inclusion_references` usam campos novos. Conferir weak form física,
energia, fonte/interfaces e q3/q4 das normas. Limite Unusual é por local, não RAM
total; não elevá-lo automaticamente. O resíduo forte exige resolução das interfaces de K.
Recriar confronto completo FreeFem; as chamadas acima não o geram implicitamente.

## 8. Casos que precisam de novos drivers

### Quarter-five-spot / obstáculo quadrado

[examples/quarter_spot_problem.py](examples/quarter_spot_problem.py): domínio
unitário, quadrado `[.25,.75]²`, área 25%, K=1e-4 dentro/1 fora, macrogrid n10;
fontes integradas −1/+1 em `[0,.1]²`/`[.9,1]²`. O material corta o interior dos
macros. Essa seleção reduziu o quadrado; não voltar ao retângulo anterior.

Criar aquisição pública original `solve_darcy`, mesh/material fitted quando a
formulação requerer, gauge/traços admissíveis. Criar comparador MSL/NeoPZ e
clássico refinado dos mesmos poços de área finita. Medir normas em overlay e
refino próprio. `plot_quarter_reference`, `plot_quarter_classical` e
`plot_quarter_elevation` só leem arrays ausentes, não fazem aquisição. A figura
19 pontual é outro caso: reproduzir sua Dirac/interpretação separadamente, com
elevação/fluxo próximos à publicação. `plot_quarter_spot` gera série homogênea/
layered que também não substitui o obstáculo.

### Transporte L11 §5.4

As entradas selecionadas estão em `examples/transport_random_problem.py` e
`examples/data/transport-random-2015/permeability.json`: array real 64×16,
seed PCG64 20261003, logaritmos iid uniformes, domínio [0,3]×[0,1], pressão
3/0 nas faces verticais e fluxo nulo nas horizontais. O artigo não fornece
o array, a lei probabilística, o seed ou esse acionamento Darcy; a seleção
não é uma identificação retrospectiva. Treze testes de identidade, eixos,
geometria, interfaces e BC passaram nos três ambientes Python. Um piloto do
domínio inteiro com 32 macros, P3/r8 e os traços 2/8, realizou cinco passos
dt=.001 em 32,53 s/212 MiB: equações originais ≤4,73×10⁻¹⁷ e Gram dos traços
livres de posto completo nesse piloto. Não é a campanha 512/T7 nem comparação
independente. Usar as entradas arquivadas, sem redesenhar o campo por seed.
`OfflineHybridSystem.solve` e `solve_transient_transport` já oferecem
`check_original=True`: verificação compartilhada com `refine_hybrid(max_steps=0)`,
sem mudar campos ou fatorizações, inclusive nos passos não retidos. Normas
originais e da carga física ficam disponíveis por passo. O controle passou
89 testes nos três ambientes e três casos nativos PARDISO. A campanha aleatória
completa e suas referências ainda precisam ser executadas. O driver público
`examples/transport_random_campaign.py` está implementado com as configurações
512/T7 declaradas e entrada arquivada. Salva campos físicos portáteis, bases
executadas, mapas/orientações e reações nodais; não arquiva coarse sem sua base
nem implementa restart. O contrato distingue Robin interior, fluxo difusivo
natural exterior e traço removido no inflow forte. A velocidade normal numérica
é o multiplicador Darcy, distinta da normal do vetor bruto volumétrico.

O piloto público atual q8, UUID `e02bdb85-7a39-4b2b-8738-da69c11d027a`, executou
32 macros/cinco passos em 21,17 s/177 MiB. Seu replay data-only com threads 1/2
reproduziu exatamente os digests portáteis de pressão, gradiente, fluxo Darcy,
concentração e gradiente da concentração. Integração independente dos campos
arquivados reproduziu as três massas observadas, inclusive IC zero. Preserva
os mínimos nodais negativos: -0,3679839 no primeiro passo e -0,3390840 no quinto;
não há corte nem alegação de positividade. Os 37 testes de arquivos/entradas/
checkpoints/normas passaram nos três ambientes; o contrato final por face passou
cinco regressões nos três ambientes. Lint e typecheck dos novos owners passaram.

Pilotos q8/q10/q12 do mesmo material, BC, espaços e cinco passos mediram
sensibilidade de quadratura, sem trocar a realização. Em t=.005, os incrementos
q8→q10/q10→q12 são 0,0241232%/0,0117330% em L2 da concentração e
0,0280562%/0,0134792% no H1 quebrado. Os denominadores são os campos computados
de quadratura maior; não constituem uma solução exata nem controle da campanha
512/T7. Evidências: `build/reports/completion/transport-random-current-pilot-q8-verification.json`
e `transport-random-pilot-quadrature-control.json`. Comparação independente
Basix P3/P2 executou nos 32 macros: 10.652 incógnitas/174.944 nnz,
resíduo original 1,24×10⁻¹⁶ e concordância dos campos físicos ≤1,79×10⁻¹².
Os dois campos foram avaliados em suas bases executadas sobre a mesma regra
positiva q6, adequada aos produtos polinomiais e K fitted. O candidato nas
equações originais nativas deu 5,51×10⁻¹³. Regressões independentes com pressão
3-x/fluxo (1,0), q3 admissível e q2 imediatamente excluído passaram; orientação,
pressões fracas e N0 foram revisadas separadamente. Isso é controle Darcy do
piloto, sem norma H(div), conservação fina ou referência refinada.

O transporte também foi montado de forma independente nos 32 macros, com
todos os cinco estados nativos, massa consistente, Robin interior e reações
nodais dos dados fortes. O confronto nos tempos observados .001/.005 passou:
concentração L2/gradiente/H1 quebrado ≤7,44×10⁻¹³ relativos, Robin interior
≤3,53×10⁻¹², reação dual nodal ≤2,88×10⁻¹². O candidato inserido no sistema
original aumentado deu ≤1,29×10⁻¹³, contra o estado anterior nativo declarado;
os limites 10⁻¹⁰/10⁻⁹ foram fixados antes da leitura e mantidos. A condição
forte deriva das faces exteriores incidentes de medida positiva; um macro que
toca a entrada apenas por um vértice não recebe uma condição nodal adicional.
O controle homogêneo preservou u=1 por cinco passos em ≤1,21×10⁻¹³. Posto
finito: B local 71/72, B global após eliminação essencial 912/912; isso não
fornece uma cota uniforme inf-sup. Relatório:
`build/reports/completion/transport-random-native-transport-comparison-face-current.json`.
O piloto nativo levou cerca de 7,7s/140 MiB; o confronto sem solve, 3,22s/135 MiB.
A trajetória completa 512/T7 e referências espaço/tempo permanecem pendentes.
O controle adicional com o Darcy heterogêneo arquivado e IC u=1 preservou
u=1 por cinco passos em ≤3,37×10⁻¹²: confirma o acoplamento fraco macro,
sem transformar a velocidade bruta em H(div). A escolha de IC é um parâmetro
separado do controle homogêneo; os padrões IC0/K arquivado são mantidos.

O novo adaptador clássico privado usa RT0/P0 global e concentração contínua
P3, no domínio inteiro e com os mesmos dados físicos salvos. Pilotos de cinco
passos executaram em 2048 e 8192 triângulos, com todas as equações originais
abaixo de 10⁻¹⁰. Matrizes/bases/orientações/campos efetivos estão arquivados;
o contrato v2 inclui também a matriz BE efetivamente executada em binary64
e os coeficientes predecessores dos checkpoints. Replay com threads 1/2 é
idêntico, sem novo solve; dez controles nativos analíticos e de normas passaram.
O incremento 2048→8192 permanece **36,15% no fluxo Darcy L2** e **52,72% no
H1 da concentração**, em t=.005, com o campo mais fino no denominador. A
sensibilidade q8→q10 é da ordem de 10⁻¹⁵ nesse piloto. Portanto, ainda não
é baseline resolvido nem substitui T7. Resultados compactos:
`examples/results/transport-random-classical-pilot-verification.json`.
Custos observados com o contrato v2: 2048/q8 6,59s/135 MiB;
8192/q8 13,93s/269 MiB, durante a campanha unfitted, sem alegação de
desempenho controlado. Ambos verificaram a matriz executada e os RHS dos
checkpoints pelos coeficientes predecessores. Próximo nível
32768 precisa de lease pesada; corresponde a uma malha selecionada, distinta
da conectividade histórica indisponível de 32000 triângulos.
O controle temporal clássico adicional manteve 2048 triângulos e adquiriu
dt=.0005/10 passos e dt=.00025/20 passos, ambos até t=.005. Os incrementos
H1 dt=.001→.0005→.00025 são **0,5955%/0,3141%**, estáveis em normas q6/q8.
Todos os checkpoints passaram o replay BLAS1/2 e seus RHS predecessores;
os owners da aquisição e as 117 fontes do core permaneceram inalterados.
O comparador compartilhado exige tempo físico comum, dados completos iguais
e índices explícitos entre trajetórias com dt diferente. Os 21 controles
nativos passaram. Custos dos dois novos controles: 5,53s/129,5 MiB e
6,23s/134,3 MiB, sob guarda de 60s/512 MiB durante a lease unfitted.
Esses incrementos iniciais não resolvem o baseline espacial ou T7. Recibo:
`build/reports/completion/transport-classical-current-temporal-verification.json`.
Ferramentas permanecem em
`.tmp/comparisons/transport-random-basix`, fora dos artefatos distribuídos.

```bash
pixi run --locked -e test python -m examples.transport_random_campaign
```

```bash
pixi run -e test python -m examples.transport_mixed_campaign --workers 1
pixi run -e test python -m examples.transport_coefficient_controls --epsilon 0.1 --resolutions 8 16 32 64 128 --local-refinement 16 --workers 1 --local-workers 1
pixi run -e test python -m examples.transport_face_resolution --local-refinement 32 --workers 1
pixi run -e notebooks python -m examples.transport_campaign --collect
```

Esses comandos complementares não são o caso aleatório acoplado §5.4. Executar
o driver aleatório completo, controles de dt e quadratura e referências refinadas
com o mesmo material e condições de contorno. Validar fluxo normal/volumétrico físico,
conservação, BC não homogênea, trajetória contra conformante independente e
refinos espaço/tempo/quadratura. Realização selecionada não identifica seed antiga.

Controle epsilon=1/r1024 não é caso publicado nem prioridade da galeria. O CLI
público `transport_face_resolution` aceita até r512; extensão exige testes de
streaming e replay no módulo compartilhado, sem usar checkpoints anteriores.

### Maxwell nanoguide

```bash
pixi run -e test python -m examples.maxwell_campaign
pixi run -e test python -m examples.maxwell_nanoguide --macro 16 --fine 128 --order 12 --steps 1131 --dt 0.01
pixi run -e test python -m examples.maxwell_nanoguide --macro 8 --fine 128 --order 12 --steps 1131 --dt 0.01
pixi run -e test python -m examples.maxwell_nanoguide_time --dt 0.005 --order 12
```

Repetir q20/q28. Criar aquisição **DG Q2 independente**, ainda ausente do clone;
piloto 32, refinamentos 256/512/1024 conforme incremento, tempo e material/quadratura.
Comparar E/H nos tempos staggered físicos, trajetórias, energia, CFL e traço
tangencial. `maxwell_nanoguide_results` lê DG/control records: regenerar antes
de plot. Fase/turn-on selecionados não identificam dados históricos. GPU depois.

### Três camadas, elastodinâmica 2017

1. Criar driver público a partir das três entradas leves, validando hashes/units.
   Macro points são normalizados: coordenadas físicas = 1000 m × points; usar a
   nondimensionalização declarada no caso, não misturar unidades.
2. `ElastodynamicStepper`, P3/r8, traço P2/s8, dt=.001/T=.3, Newmark beta=.25/
   gamma=.5, IC=0/tração=0. Lamé físico `lambda=rho*(Vp**2-2*Vs**2)`. O contrato
   explicita diferença da equação impressa; não chamar isso de reprodução literal.
3. Montagem independente DOLFINx/Basix, sem importar matrizes/integração local
   PyMHM. Subcortes são regiões de integração; base avaliada no elemento original.
   Força radial em disco integrada independentemente; não inverter slivers auxiliares.
4. Comparar M/K/B/f nos macros 7, 10, 151 e 153 e depois nos **341 macros**, simetria,
   orientação e q5/q7. Critérios registrados: erro relativo dos operadores de 1e-10, força na
   norma dual da massa de 2e-8. Não relaxar tolerâncias.
5. Medir montagem e três passos; adquirir os 301 estados em ambas as montagens. Comparar
   deslocamento, velocidade, tensão, energia, trabalho, momento e médias de tração
   no intervalo temporal.
   Operadores locais concordantes não bastam para aceitar trajetória.
6. Criar clássico P3, malhas conformantes Gmsh com h = 8, 4 e 2 m, mesmas interfaces/fonte;
   medir refino espaço/tempo/quadratura e trajetória. Publicar caso selecionado
   aceito com seus limites históricos. O CLI analítico abaixo não o substitui.

## 9. Demais famílias: reconstrução e controles

### Wells tetra/prisma / mapeado

```bash
pixi run -e test python examples/solve_mapped_oscillatory_well.py --fine-factor 8 --macro-factor 1 --quadrature-xy 28 --quadrature-z 10 --workers 1 --global-refinement-precision extended
pixi run -e test python -m examples.solve_mapped_well_classical --fine-factor 16 --quadrature-xy 28 --quadrature-z 10 --workers 1 --global-refinement-precision extended
pixi run -e test python examples/solve_mixed_well_geometries.py --kind tetrahedron --pressure-degree 1 --fine-factor 2 --macro-factor 1 --workers 1
pixi run -e test python examples/solve_mixed_well_geometries.py --kind tetrahedron --pressure-degree 2 --fine-factor 2 --macro-factor 1 --workers 1
pixi run -e test python examples/solve_mixed_well_geometries.py --kind prism --pressure-degree 1 --fine-factor 2 --macro-factor 1 --workers 1
```

Prosseguir com F8/F16, fatores macro 1/2/4/8 e q28/q40, separando restrição do traço e refino
clássico. `compare_mapped_well_joint --reference ... --candidates ... --orders
6 8 10 --workers 1 --group-workers 1` integra novos campos. F256 clássico tem
cerca de 25.17 milhões de DOF: refatorar fases e medir memória primeiro. Refazer integrações
pendentes e o mesmo caso completo com NeoPZ `TPZMHMixedMeshControl`/Basix: Piola, geometria
facetada comum, div(V) = Q, traço, gauge e graus interiores/de face.

### Nested/MsHHO/polígonos/3D

```bash
pixi run -e notebooks python examples/verify_nested.py
pixi run -e notebooks python examples/verify_mshho.py
pixi run -e notebooks python examples/verify_tensor_rt.py --levels 2 4 8 16 32 64
pixi run -e notebooks python examples/verify_polygons.py
pixi run -e test python -m examples.mh3d_campaign --resolutions 1 2 3 4 5
pixi run -e test python examples/solve_flow3d.py --levels 1 2 3 4 5 --workers 1
pixi run -e test python examples/solve_gals3d.py --levels 1 2 3 4 5 --workers 1
pixi run -e test python examples/solve_gals3d.py --levels 1 2 3 4 5 --workers 1 --primal-only --primal-refinement-precision extended
pixi run -e test python examples/solve_elasticity3d.py --resolutions 1 2 3 4 5 --degree 2 --workers 1
pixi run -e test python examples/solve_elasticity3d.py --resolutions 1 2 3 4 5 --degree 3 --workers 1 --sweep
pixi run -e test python -m examples.solve_mixed_elasticity3d --workers 1
pixi run -e test python examples/solve_planar3d.py --levels 1 2 3
pixi run -e test python -m examples.solve_tetra_pk uniform --workers 1
pixi run -e test python -m examples.solve_tetra_pk fixed --workers 1
pixi run -e test python -m examples.solve_star_polyhedra --workers 1
pixi run -e test python -m examples.solve_reconstruction3d uniform
pixi run -e test python -m examples.solve_reconstruction3d adaptive
pixi run -e test python -m examples.reconstruction3d_resolution
pixi run -e test python -m examples.reconstruction3d_rt_order
```

Os quatro `verify_*` também geram plots; conferir reutilização de JSON antes.
`verify_analytic.py --levels 4 6 8 16 32 64`, no ambiente `notebooks`, também
adquire e renderiza o caso analítico; exige a digitalização leve
`examples/results/published/harder2013_figure5.csv` para os marcadores publicados. Scripts
sem seletor executam famílias fixas: acrescentar seleção de piloto se caro.
`examples/data/reconstruction3d-macro.json` é input leve para resolução.
Construção RT difere do teorema `k>=ell+d`: 3D P3/P0 é admissível na fronteira;
P4/P2 não satisfaz k>=5. Auditar todos os pontos de chamada, dimensões e casos relacionados.
Refazer BC homogênea/não homogênea, campos/precisão e oráculos nativos completos
de fluxo/GaLS/H(div). Padrão de erro visual e residual pequeno não concluem a auditoria.

### MH/MH²M, elasticidade mista e HPC4E

```bash
pixi run -e test python -m examples.mh_campaign --resolutions 4 8 16 32 64 --workers 1
pixi run -e test python -m examples.mh_boundary_campaign --resolutions 1 2 4 8 16
pixi run -e test python -m examples.mh2m_campaign --resolutions 2 4 8 16 32 --reference-levels 64 128 256
pixi run -e test python -m examples.mh2m_heterogeneous --references 32 64 128 256 512 1024 --resolutions 4 8 16 32 64 --norm-workers 1
pixi run -e test python -m examples.mh2m_crisscross_campaign --stage acquire --norm-workers 1
pixi run -e fem python -m examples.compare_mh2m_cg3 --stage reference --sizes 32 64 128 256 512 --assembly-order 16 --quadrature-control 12 --workers 1
pixi run -e fem python -m examples.compare_mh2m_cg3 --stage cases --orders 8 10 --workers 1
pixi run -e test python examples/solve_elasticity_families.py --degree 2 --enrichment 1
pixi run -e test python examples/solve_hpc4e_mhm.py --download --segments 1 2 4 8 --workers 1
pixi run -e fem python examples/solve_hpc4e_reference.py --patch --nx 4 --ny 2 --degree 2 --threads 1
```

Para completar a elasticidade 2D, repetir tensor constante/variável, famílias
BDM/RT e enriquecimentos, além do caso oscilatório da Tabela 3 de L18. Drivers
públicos correspondentes:

```bash
pixi run -e test python examples/solve_primal_elasticity.py --degree 3 --resolutions 1 2 4 8 16
pixi run -e test python examples/solve_elasticity_tensor_rt.py --degree 2 --enrichment 0 --resolutions 1 2 4 8 16
pixi run -e notebooks python examples/solve_elasticity_literature.py --trace-degree 1 --enrichment 0 --segments 1 2 4 8 --order 12
```

O último importa helpers de plot e precisa de Matplotlib. A Tabela 3 tem uma
inconsistência entre valores/ordens da coluna de rotação; preservar os valores
publicados e comparar normas calculadas independentemente, sem ajustá-las.

MH²M: epsilon = 1/14, gamma = 1.8 da dissertação, fonte específica; não ajustar pela
figura. Crisscross precisa do manifesto gerado pelo estudo heterogêneo. Separar
refinos se necessário; medir CG3 próprio, overlay q8/q10 e montagem independente.

Após o patch HPC4E, clássico RT2 (`--degree 2`) em 512×256 e, conforme incremento,
1024×512, via mpiexec/fem, `--mpi --threads 1 --refinement-precision extended` e
workspace explícito e `--download` para obter as entradas. Workspace não garante teto de memória. Comparar com
`compare_hpc4e.py`/`compare_hpc4e_fields.py`, depois `plot_hpc4e.py`. Conferir
stress/compliance/work/gauge/materiais. Locking-free exige sweep Lamé finito/
infinito no mesmo refino da formulação mista; patch afim primal não comprova isso.

### Stokes/Brinkman/Oseen

```bash
pixi run -e test python examples/solve_stokes_adaptive.py --strategy uniform --levels 2 4 8 16 32 --trace-degree 1
pixi run -e test python examples/solve_stokes_adaptive.py --strategy macro --cavity --lid constant --iterations 4 --save-history
pixi run -e fem python examples/solve_cavity_reference.py --lid constant --levels 32 64 128 256 512
pixi run -e test python examples/solve_oseen.py --case smooth --levels 2 4 8 16 32
pixi run -e test python examples/solve_oseen.py --case boundary --viscosity .01 --levels 2 --adaptive --iterations 4 --order 20
```

Refazer camadas internas/adaptação de faces conforme casos versionados. Cavity
lid constante: singularidades nos cantos, norma com cutout, mesmo BC no baseline,
refino próprio. Lid regularizado é outro problema. USFEM2017 usa
nu*grad u:grad v e pseudo-tração(-nu*grad u+pI)n; coincidir gauge físico global,
estabilização e RHS. Preservar perfis unilaterais sem suavizar saltos. Oseen não
é Navier–Stokes.

### Elastodinâmica analítica, Equação 53

```bash
pixi run -e test python -m examples.elastodynamics_campaign --n 1 --dt 0.005 --order 12 --workers 1
pixi run -e test python -m examples.elastodynamics_campaign --n 2 --dt 0.005 --order 8 --workers 1
```

Repetir n = 3, 4, 5, 6 e 8 espacialmente. No estudo temporal, usar n = 2 e
dt = 0.1/2^j, com j = 0,…,10. Conferir normas,
quadratura, energia/trabalho, tração e taxas. `elastodynamics_results` e
`plot_elastodynamics_native` exigem novos comparadores antes de renderizar.

## 10. Aceite científico, atualização e conclusão

Manter em `docs/roadmap.md` uma linha para cada L01–L20 e extras: formulação,
caso, referência, resultado e limite. L12 RAD requer também
`verify_rad_conditioning.py`, `verify_rad_layer.py` e star-polyhedra;
L02 requer `pixi run -e notebooks python -m examples.darcy_jump_campaign --collect`
e controles do estimador. Os controles RAD também importam Matplotlib quando
renderizam: usar `pixi run -e notebooks python examples/verify_rad_conditioning.py`
e `pixi run -e notebooks python examples/verify_rad_layer.py`. O mapa de comandos é trabalho a executar, não certificado concluído.

Reprodução exige geometria, K/fonte, BC/gauge, método, graus, partições,
quadratura e norma publicados. Se investigação exaustiva não resolver uma entrada
histórica, registrar exatamente o limite e comparar o **mesmo caso selecionado
completo** em montagem independente. Referência clássica deve ter refino próprio;
taxas só são exigíveis sob as hipóteses da literatura. Testes pequenos/cobertura/
matrizes concordantes não substituem essa comparação.

Corrigir no módulo compartilhado; propagar para solvers, pontos de chamada, exemplos,
referências, figuras/notebooks/registros. Centralizar condições dimensionais e
testar fronteira admissível/excluída. Persistir bases executadas e hashes junto
aos coeficientes; testar replay por threads e rotações equivalentes. Não usar
tolerância relaxada, exclusão de cobertura ou workaround de caso.

Cada aquisição: commit/lockfile/versões, hashes dos inputs, malha/base/dtype,
solver/threads/quadratura, normas físicas e incremento clássico. Conferir a forma
fraca física original, conservação fina/macro, energia/trabalho e fluxo exterior.
Separar gradiente bruto, fluxo H(div) e multiplicador. JSON histórico não é saída nova.

Após aceitar uma família:

1. Atualizar JSON/CSV leves e matriz de literatura; nomear código de referência,
   módulo, revisão/URL. Fontes/ferramentas privadas ficam fora de Git/pacotes.
2. Gerar só figuras retidas no perfil. Adaptar renderer que exige comparador
   ausente ao contrato novo, sem fabricar manifesto de aceitação para plotar.
3. Mostrar componentes/vetores, escala comum, erro próprio declarado, malha macro
   verdadeira em todos painéis e interseções nos perfis. Preservar lados das
   interfaces. Usar velocity/flux, nunca speed. Inspecionar PNG/SVG em tamanho
   final: sem sobreposição, clipping, labels ilegíveis ou cinza ocultando sinal.
4. Atualizar/executar notebook selecionado, não todos para duplicar outputs.

```bash
pixi run -e notebooks python scripts/notebook_data.py --notebook 23
pixi run -e notebooks python scripts/notebook_data.py --notebook 23 --check
```

Antes desses comandos, atualizar os seletores de arquivos do notebook e de
`scripts/notebook_data.py` para os resultados novos. O helper ainda contém nomes
históricos, como Q3 768×1408 e Brinkman q5; os solves acima não recriam esses nomes.
O inventário seleciona agora o notebook antes de ler os manifestos; ausências
de outra família não interrompem essa seleção. Manifestos, arrays e hashes da
família escolhida continuam obrigatórios. Os nomes históricos de arquivos não
foram alterados; atualizar seletores somente junto com os resultados adquiridos.

O primeiro comando lista dependências/ausências quando seus manifestos existem;
o segundo exige campos reais, sem download automático. Usar
`pixi run -e notebooks python scripts/run_notebooks.py notebooks/<nome-real>.ipynb`
após confirmar o nome em `notebooks/`. Um plot lendo JSON não executa a PDE nem
valida o record.

Verificar assets da galeria antes do build:

```bash
pixi run -e test python - <<'PY'
import json
from pathlib import Path
root = Path('docs')
profile = json.loads((root / 'publication-core.json').read_text())
missing = [name for name in profile['assets'] if not (root / name).is_file()]
print('\n'.join(missing) if missing else 'Todos os assets selecionados existem.')
raise SystemExit(bool(missing))
PY
pixi run -e docs docs-check
```

No clone limpo faltam figuras por definição; gerar os campos/plots numéricos e
reextrair as figuras publicadas necessárias antes desse gate. O checker
valida delimitadores de todas fontes e matemática de todo HTML gerado. Build
não compila TeX: verificar MathJax no navegador; displays em $$ isolados e
equações longas alinhadas. Não inserir este diário operacional nos docs públicos.

```bash
pixi run -e docs docs-serve --dev-addr 127.0.0.1:8000
```

Encaminhar porta no cliente SSH para acessar servidor remoto; isso é instrução
operacional deste roteiro, não conteúdo científico da galeria.

Gates finais:

```bash
pixi run -e test lint
pixi run -e test format-check
pixi run -e test typecheck
pixi run -e test test-cov
pixi run -e test-py311 test
pixi run -e test-py312 test
pixi run -e fem test-fem
pixi run -e fem test-mpi
pixi run -e meshing test-meshing
pixi run -e visualization test-visualization
pixi run -e docs docs-check
pixi run -e packaging ci-check
pixi run -e packaging metadata-check
pixi run -e packaging build
pixi run -e packaging check-dist
```

Executar integrações PARDISO/GPU quando alvo da entrega; mocks não são integração.
Gates de linhas e branches ambos >=99%, sem esconder falhas. Inspecionar wheel/sdist,
instalação limpa e receita Conda noarch. PyPI/conda-forge não foram publicados.

Os gates e controles executados neste host estão descritos no estado da
retomada. Todas as integrações opcionais e campanhas restantes continuam
pendentes; não confundir esses resultados atuais com a aceitação integral.
A conclusão depende da matriz científica, das referências refinadas e dos
assets reais exigidos pelo gate da galeria.
