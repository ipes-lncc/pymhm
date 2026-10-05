# PyMHM: roadmap de implementação, verificação e validação

Atualizado em **5 de outubro de 2026**. Este é o plano canônico de implementação,
verificação e validação do projeto. A [matriz científica](#scientific-scope-and-acceptance-criteria) relaciona as formulações à literatura;
as [páginas dos casos](docs/cases/index.md) descrevem os resultados e seus limites.
Este roadmap não certifica a reprodução integral da literatura.

## Como usar este roadmap

Escolher uma entrega delimitada, conferir a evidência atual do caso e definir
seus critérios de aceite antes de implementar ou executar uma campanha. Uma
pendência de aquisição depende de dados, recursos e programas de referência;
não significa necessariamente que falta implementar o método.

Usar quatro estados nas próximas atualizações:

- **Implementado:** a capacidade existe, com seus contratos e restrições.
- **Verificado:** há execução identificada que verifica a capacidade no escopo declarado.
- **Pendente:** falta uma implementação, comparação ou controle especificado.
- **Dependência externa:** faltam dados históricos, acesso a uma referência ou recursos.

Uma entrega só passa a verificada com evidência reproduzível. Atualizar aqui o
estado e a próxima ação; resultados detalhados pertencem à página do caso.
Não acumular diários, comandos obsoletos ou resultados intermediários neste arquivo.
Não repetir uma campanha já aceita apenas porque uma anotação antiga a dizia pendente.

## Base atual que deve ser preservada

- [x] API variacional independente da física: `Equation`, `LocalEquations`,
  `MultiscaleProblem`, `assemble` e `solve`; formas UFL locais e globais ou blocos
  numéricos explícitos. Campos escalares, vetoriais e mistos usam esses contratos.
- [x] Recursão por `NestedEquations`, modos retidos, momentos físicos,
  reconstrução e problemas face-only. A recursão atual tem restrições de
  execução; sua disponibilidade não qualifica automaticamente hierarquias MPI.
- [x] Basix como dependência de runtime para elementos, polinômios e tabulações;
  mapas MHM próprios preservam as restrições, momentos e orientações publicados.
- [x] Imports canônicos, módulos organizados por responsabilidade, funções livres
  e delegações explícitas. Os imports anteriores à refatoração não são suportados.
- [x] Execução serial, threads e processos `spawn`, redução ordenada das
  contribuições de faces, integração MPI e solvers CPU/GPU nos escopos documentados.
- [x] Reúso de malhas, espaços e formas compiladas compatíveis por worker.
  Matrizes materiais são montadas por macroelemento; fatores e hierarquias só
  podem ser reutilizados quando a equivalência do operador estiver comprovada.
- [x] Notebooks organizados por problema, incluindo os dez
  [introdutórios](notebooks/introduction/README.md), em inglês, com formas,
  condições de contorno, referências clássicas e campos definidos passo a passo.
- [x] Estudos 2D/3D de desempenho; a
  [campanha de aceleradores 3D](docs/cases/darcy-3d-accelerators.md) já inclui
  PARDISO, escalabilidade forte/fraca CPU e uma/duas GPUs com cuDSS e AMGX.
  Os ganhos têm o escopo medido: não estabelecem vantagem universal nem precisão equivalente.
- [x] Ambiente `test` completo para Linux com duas GPUs; `test-core` portátil;
  preflight de dependências, pytest-xdist e fase exclusiva `serial`.

A base de engenharia em `c11a2d2`, verificada em Linux, contém **5.771 testes
aprovados**, **64 workers** e **32 testes exclusivos em série**, com cobertura de
**99,9327% de linhas** e **99,7098% de ramos**. O único skip do checkout é o
controle de propriedade do wheel instalado, executado separadamente: os oito
controles da instalação isolada passaram. Foram removidos 451 casos redundantes
sem perder linhas ou ramos cobertos. Essas contagens descrevem essa execução;
não são metas para manter artificialmente nas próximas versões.

As comparações de engenharia preservam resultados nos casos executados. Elas
não substituem testes de estabilidade, resolução ou reprodução de outros casos.
A execução nativa da revisão em Windows e a qualificação de providers ML/preCICE
continuam pendentes. As implementações privadas em `_legacy` ainda dão suporte
a comparações; não devem voltar a definir a API principal.

## Prioridades e dependências

| Marco | Prioridade | Resultado esperado | Dependência |
| --- | --- | --- | --- |
| R1 — Brinkman | P0 | Matriz 2D/3D de regimes e espaços com limites de precisão demonstrados | Base variacional e referências independentes existentes |
| R2 — Desempenho geral | P1 | Escalabilidade e custo a precisão equivalente, além dos benchmarks atuais | Controles físicos e profiling do fluxo completo |
| R3 — Aceite científico | P1 | Fechamento progressivo dos controles de resolução e alvos da literatura | Dados identificados e referência refinada de cada caso |
| R4 — API e arquitetura | P1 | Menos infraestrutura manual e duplicação, preservando a expressividade | Equivalência numérica antes/depois |
| R5 — Windows | P1 | Execução nativa identificada do core, PARDISO e artefatos | Runner Windows e dependências daquele perfil |
| R6 — Providers externos | P2 | Integrações substituíveis pelos contratos locais existentes | Contratos e controles físicos de R4 |
| R7 — Distribuição | P2 | Release com escopo científico e plataformas efetivamente verificados | Gates e evidências das capacidades anunciadas |

Começar por R1. R4 e R5 podem avançar em paralelo quando não alterarem a fonte
de uma aquisição em andamento. Executar R3 por família, com entregas pequenas.
Campanhas de desempenho precisam de recursos exclusivos; não misturar seus
tempos com testes ou outras campanhas concorrentes.

## R1 — Qualificar Stokes–Brinkman em 2D/3D

Já existem casos analíticos, estudos de camadas para famílias polinomiais,
referências Taylor–Hood refinadas e comparações nativas. O objetivo agora é
ampliar a qualificação de robustez e resolução, usando
[camadas](docs/cases/introduction-layers.md), [fluxo 3D](docs/cases/flow3d.md),
[reprodução Stokes](docs/cases/reproduction.md) e [SPE10](docs/cases/spe10.md).

- [ ] Definir uma matriz enxuta com Stokes, Brinkman intermediário, resistência
  pequena/grande, viscosidade pequena, resistência tensorial admissível e alto
  contraste, em ambas as dimensões. Identificar soluções exatas e referências numéricas.
- [ ] Tratar MHM–Taylor–Hood e MHM-USFEM como discretizações distintas. Conferir
  graus, malhas locais, kernels, traços, gauge de pressão, estabilização e fonte
  contra a formulação adotada. Declarar se o operador usa gradiente ou deformação simétrica.
- [ ] Separar pseudo-tração, multiplicador e tensão física. Verificar o limite
  de resistência zero, dados homogêneos/não homogêneos e as condições de compatibilidade.
- [ ] Refinar macro malha, malha local e espaço de traço separadamente. Usar
  pelo menos três níveis quando a viabilidade permitir estimar uma taxa;
  investigar platôs e camadas não resolvidas antes de atribuir uma ordem ao método.
- [ ] Medir erro de velocidade, pressão e gradiente/tensão separadamente,
  divergência e balanço macro. Controlar quadratura e resolução da referência
  independente do mesmo problema. Resíduos físicos por campo acompanham esses erros.
- [ ] Conferir os perfis de camada com valores unilaterais nas macrofaces;
  medir oscilações sem assumir que a estabilização fornece um princípio do máximo.
- [ ] Atualizar e executar o notebook introdutório e as figuras a partir dos
  registros aceitos. Mostrar campos, erros, macro malha e taxas efetivamente observadas.

**Aceite:** cada configuração tem espaços admissíveis, unicidade/gauge e
controles do operador; normas e incrementos identificam sua resolução. As taxas
são compatíveis com as hipóteses aplicáveis quando o regime assintótico é atingido.
Regimes pré-assintóticos ou não resolvidos permanecem explicitamente limitados.
Uma comparação analítica não recebe o rótulo de reprodução de uma figura histórica.

## R2 — Generalizar desempenho e preparar a execução HPC

A primeira campanha CPU/multi-GPU está concluída no escopo declarado. O próximo
passo parte dessas medições, das [estratégias de execução](docs/execution.md) e
não pressupõe que MHM será mais rápido que o melhor baseline clássico.

- [ ] Comparar custo para alvos comuns de erro de pressão e fluxo físico,
  refinando também macro malha e traços. Manter separada a comparação já existente
  com o mesmo número de elementos: espaços e precisão globais são diferentes.
- [ ] Ampliar a diversidade de materiais, incluindo coeficientes não periódicos,
  SPE10 e cargas locais desbalanceadas. Reusar kernels e infraestrutura por
  compatibilidade; montar e resolver cada operador material necessário.
- [ ] Medir montagem, fatores/hierarquias, todos os RHS, comunicação,
  serialização, redução, solve global e reconstrução. Identificar o limite da
  parte local, o gargalo global, a memória e o ponto de vantagem por tamanho/precisão.
- [ ] Comparar LU SciPy/PARDISO/MUMPS e AMG CPU/GPU sob condições de uso
  admissíveis e orçamento declarado. Justificar tolerâncias iterativas antes
  dos ensaios por erro de campo e discretização; manter os critérios existentes
  nos resultados atuais. Não alterar um limiar depois para aprovar uma amostra.
- [ ] Qualificar scheduling de lotes heterogêneos, workspaces residentes e
  transferência CPU/GPU com os mesmos controles do operador original. Reúso de
  fatores exige identidade comprovada; sem equivalência, apenas infraestrutura é compartilhada.
- [ ] Expandir o esqueleto global e avaliar pré-condicionamento/distribuição
  sem reunir todas as matrizes locais num único processo. Separar AMG de blocos
  positivos do tratamento de saddles e núcleos físicos.
- [ ] Planejar campanhas multi-node e com mais de duas GPUs quando houver
  recursos. Preservar posse dos recursos por worker/rank, `spawn`, limites de
  threads, encerramento de fatores e semântica das faces compartilhadas.

**Aceite:** curvas de tempo, speed-up, eficiência forte/fraca, memória e custo
versus erro usam aquisições reais, repetições e recursos identificados. Aquecer
JIT separadamente e declarar o escopo do aquecimento. O tempo completo inclui
setup, transferências, sincronização e encerramento; tempos de fases pertencem à
mesma aquisição. Ganhos ausentes e regressões são publicados. Resultados de
Gomes et al. e Penna et al., citados na página de aceleradores, orientam o estudo;
escalabilidade histórica de clusters não é uma capacidade já reproduzida aqui.

## R3 — Fechar a validação científica por família

A [matriz de alvos L01–L20 e adicionais](#acceptance-by-literature-target) deve continuar sendo o
índice de aceite. Antes de abrir uma tarefa, conferir a página atual do caso:
sistemas completos e referências já adquiridos não voltam a ser classificados
como implementações ausentes. Priorizar controles cujo incremento ainda possa
alterar a conclusão física.

### Darcy, interfaces, estimadores e SPE10

- [ ] Periódico: concluir controles local/traço e refinamento da referência
  clássica, mantendo material, macro malha e espaços publicados. Referências
  de graus diferentes são controles distintos, sem substituir a histórica Q1.
- [ ] Interfaces: estender os incrementos locais aos traços superiores
  admissíveis. Os endpoints r32→r64 de ℓ0/ℓ1 e os doze sistemas UFL S0/S2 já
  existem. P8/r16–P3/s32 tem kernel exato do multiplicador; solver ou quadratura
  não corrigem essa incompatibilidade. Identificar os dados históricos de S2.
- [ ] SPE10: medir separadamente resolução local, de face e de material,
  incrementos dos fluxos vetoriais e da referência; preservar camadas, unidades,
  orientação e macro partições. Fechar o controle energético Q3/RT0 ainda fora
  do critério declarado, sem modificar o limiar para obter aprovação.
- [ ] Adaptatividade: aproveitar as comparações independentes inicial/final
  já existentes e reduzir a sensibilidade restante da referência. Conferir
  conectividade/marking quando a reprodução depender da malha histórica.
- [ ] Poços e geometrias mapeadas: controlar singularidade, Piola, geometria
  comum, `div(V)=Q`, pressão, graus interiores/de face e refinamento da referência.
- [ ] Estimadores: conservar a distinção entre reconstrução algébrica e hipóteses
  do teorema. Centralizar e testar `k >= ell + d` na fronteira admissível e nos
  casos imediatamente excluídos de cada dimensão.

Evidência: [periódico](docs/cases/periodic.md), [interfaces](docs/cases/unfitted.md),
[fluxo SPE10](docs/cases/spe10-flux.md), [SPE10 adaptativo](docs/cases/spe10-adaptive.md),
[poço oscilatório](docs/cases/mapped-well-oscillatory.md) e
[reconstrução 3D](docs/cases/reconstruction3d.md).

### Variantes multiescala, RAD e fluxo

- [ ] MsHHO, MH, MH²M e recursão: completar controles afetados de convergência
  e fonte, respeitando as hipóteses de equivalência e os espaços de cada método.
  Sistemas independentes e referências MH²M já existem; discrepâncias gráficas
  históricas não autorizam ajustar coeficiente, fonte ou parâmetros pela figura.
- [ ] PGMHM e USFEM: resolver comprimentos reativos e interfaces materiais;
  separar espaços publicados dos controles enriquecidos. As comparações nativas
  completas já existentes são a base para o refinamento.
- [ ] RAD: complementar condicionamento, camadas e polytopes no mesmo caso físico;
  preservar o residual completo, derivadas de coeficientes e convenções de fronteira.
- [ ] Stokes adaptativo/Oseen: fechar controles de camada e baseline com o mesmo
  lid, gauge e cutout dos cantos. Lid regularizado é outro problema; Oseen não
  qualifica uma implementação não linear de Navier–Stokes.

Evidência: [MsHHO](docs/cases/mshho.md), [MH²M](docs/cases/mh2m-heterogeneous.md),
[PGMHM SPE10](docs/cases/pgmhm-spe10.md), [USFEM SPE10](docs/cases/unusual-spe10.md),
[Stokes adaptativo](docs/cases/stokes-adaptive.md) e [Oseen](docs/cases/oseen.md).

### Elasticidade

- [ ] Completar os sweeps de Lamé finito/infinito e refinos para GaLS,
  Taylor–Hood e formulações mistas. Um patch afim primal não prova ausência de locking.
- [ ] HPC4E: refinar ou quantificar a sensibilidade restante em rotação e
  compliance. Referências independentes RT1/RT2, equilíbrio/trabalho e
  comparações completas já existem; preservar traços e normas de tensão publicados.
- [ ] Elasticidade mista: complementar tensores, enriquecimentos e geometrias
  com modos rígidos físicos e simetria fraca. A inconsistência impressa de
  rotação na Tabela 3 e a conectividade desconhecida limitam a reprodução literal.

Evidência: [GaLS](docs/cases/elasticity.md), [HPC4E](docs/cases/hpc4e.md),
[famílias mistas](docs/cases/mixed-families.md) e
[elasticidade mista 3D](docs/cases/mixed-elasticity3d.md).

### Transporte e ondas

- [ ] Transporte: completar o caso de 512 macros até T=7 com a mesma realização
  material, comparando referências refinadas em espaço/tempo/quadratura.
  Distinguir velocidade/fluxo de volume da transferência normal numérica.
- [ ] Elastodinâmica: avaliar cada base executada separadamente nos estados
  arquivados; comparação em coordenadas comuns não substitui esse controle.
  Completar referência conformante e refinos de espaço, tempo e quadratura.
- [ ] Helmholtz/Marmousi: completar as sequências e a família de quinze casos
  com memória/armazenamento por fases, ressonâncias admissíveis e normas no
  mesmo domínio físico. As referências P1–P4 existentes são a base do controle.
  Para ponto-fonte 2D, manter o cutout físico fixo nas normas derivativas;
  não afirmar que a norma H1 global é finita.
- [ ] Maxwell: controlar a sensibilidade temporal/espacial da trajetória usando
  a referência central-DG Q2 e as comparações CPU/GPU já executadas. Manter
  tempos escalonados, CFL e balanço; fase/amplitude/turn-on históricos precisam
  estar identificados antes de afirmar reprodução literal.

Evidência: [transporte](docs/cases/transient-transport.md),
[elastodinâmica](docs/cases/elastodynamics.md), [Helmholtz](docs/cases/helmholtz.md),
[Marmousi](docs/cases/marmousi.md) e [Maxwell](docs/cases/maxwell-nanoguide.md).

**Aceite de R3:** cada alvo fecha sua condição específica, ou registra a entrada
não identificada e a comparação independente do mesmo caso completo disponível.
Dados históricos desconhecidos permanecem como dependência externa. Solução
manufaturada, um patch pequeno ou cobertura não substituem esse aceite.

## R4 — Tornar a API mais simples sem especializá-la por física

- [ ] Reduzir a infraestrutura manual para mapas de entidades/DOFs locais e
  globais, orientação, fronteiras e composição de gauges. Formas UFL globais já
  são montadas; a numeração do esqueleto continua explicitamente declarada.
- [ ] Consolidar operações repetidas nos donos de geometria, tabulação,
  montagem, condensação, reconstrução e execução. Retirar código privado legado
  só após migrar consumidores e provar preservação das variantes e dos campos.
- [ ] Qualificar a composição de hierarquias mais gerais e suas restrições de
  fronteira, recursos e paralelismo. O MPI atual não suporta problemas locais
  recursivos; não ocultar essa limitação sob a mesma opção de execução.
- [ ] Manter operadores de conveniência como exemplos de composição, apresentados
  depois da definição explícita. Atualizar os tutoriais escalares, vetoriais,
  primais, H(div), MH²M, MsHHO e USFEM sempre que os contratos mudarem.

**Aceite:** o usuário declara formas próximas da matemática sem precisar de um
dispatcher de PDEs nem de imports por modelo. Novas capacidades reutilizam
Basix e os contratos gerais; funções e objetos têm responsabilidades, docstrings,
tipos e convenções explícitas. Comparar blocos, RHS, bases, gauges e campos antes/depois;
a concordância numérica deve ser justificada, preservando identidades exatas de
armazenamento/replay. [Arquitetura](docs/architecture.md) e
[API variacional](docs/variational.md) permanecem sincronizadas.

## R5 — Qualificar Windows nativamente

- [ ] Executar a revisão identificada em Windows x86-64: `test-core`, Basix,
  SciPy/PyAMG, PARDISO, faces compartilhadas e serial/threads/processos `spawn`.
- [ ] Construir e instalar wheel fora do checkout; verificar origem dos imports,
  runtime MKL, liberação de fatores e os campos contra montagem independente.
- [ ] Executar notebooks portáteis selecionados e documentar a matriz real de
  plataformas/backends. DOLFINx/PETSc atuais continuam fora do perfil Windows nativo.
- [ ] Avaliar backend FEM Windows sem PETSc apenas se couber nos contratos
  gerais e houver necessidade concreta; WSL2 e Windows nativo são alvos distintos.

**Aceite:** recibo de execução Windows da revisão, core sem imports opcionais
obrigatórios, dependências resolvidas pelo lockfile e erro explícito para opções
não suportadas. Precisão estendida e limites de workers respeitam a plataforma.
Lockfile e configuração de CI isoladamente não certificam execução.
Ver [Windows](docs/windows.md).

## R6 — Qualificar providers externos

- [ ] Demonstrar um provider independente com os mesmos mapas, momentos,
  operadores e respostas fonte/traço exigidos pelo problema global.
- [ ] Qualificar um provider aprendido contra um FEM independente, com conjunto
  de teste separado, erro de campo e resíduos físicos; definir rejeição/fallback
  para respostas fora do contrato. Não restringir a interface a uma arquitetura ML.
- [ ] Avaliar preCICE como adaptador opcional se houver acoplamento que o justifique,
  considerando transferência, sincronização, licenciamento e plataformas.
  Não torná-lo dependência do core nem presumir suporte Windows.

**Aceite:** substituição do provider sem alterar a formulação global, dados
transferíveis entre workers e gestão explícita de recursos. Precisão,
estabilidade e custo são demonstrados por integração real; contratos ou mocks
não qualificam um pacote externo. Ver [providers](docs/tutorials/providers.md).

## R7 — Preparar a distribuição com escopo demonstrado

- [ ] Selecionar quais capacidades e casos compõem a release; anunciar apenas
  os regimes, plataformas e backends com evidência válida para a fonte entregue.
- [ ] Validar wheel/sdist, instalação isolada e receita Conda noarch; sincronizar
  versão, dependências, metadados, documentação e notas da release.
- [x] Manter o wheel com todo o runtime, tipagem, licença e metadados; limitar o
  sdist a `src/pymhm`, `pyproject.toml`, `README.md`, `LICENSE`, `.gitignore`
  exigido pelo backend e metadados gerados. Verificar caminhos permitidos, fontes
  idênticas e construção fora do checkout a partir do sdist. Docs, scripts,
  exemplos, testes, benchmarks, notebooks, recipe, roadmap
  e ambientes Pixi permanecem no checkout, fora dos arquivos de instalação.
- [ ] Conferir notebooks, catálogo, assets e marcações da literatura. Publicar
  só registros/figuras atuais; versionar somente as figuras selecionadas para
  a documentação publicada, com allowlist explícita. Campos grandes e saídas
  intermediárias ficam fora do Git; toda a documentação fica fora dos artefatos
  Python/Conda. Fontes externas e ferramentas privadas de
  comparação ficam fora do repositório versionado e dos artefatos distribuídos.
- [ ] Configurar o trusted publisher PyPI para `publish-pypi.yml` e o ambiente
  `pypi`; selecionar GitHub Actions como origem do Pages e permitir tags de release no ambiente
  `github-pages`. Tags `v*` devem publicar automaticamente o pacote e o site
  validados depois dos gates obrigatórios. Submissão ao conda-forge permanece
  uma etapa própria; construção de artefatos não implica publicação.

## Protocolo de aceite de qualquer entrega

1. **Definir o experimento:** formulação, hipótese, geometria, material, fonte,
   fronteira, gauge, espaços, graus, refinamentos, quadraturas, tolerâncias e
   alvos de precisão. Conferir hipóteses dimensionais e estabilidade/injetividade.
2. **Verificar o núcleo:** operador original, kernels esquerdo/direito, traços,
   momentos e condensação contra sistema completo independente. Um resíduo
   reduzido pequeno não estabelece unicidade, estabilidade ou erro de campo.
3. **Validar os campos:** normas por campo, residual por bloco físico, conservação
   macro/fina conforme a formulação, energia/trabalho quando aplicável e refinos
   separados. Referência numérica tem seu próprio controle de refino e quadratura.
4. **Preservar a proveniência:** revisão e fonte executada, lockfile, versões,
   hashes de inputs, malhas, coeficientes, bases executadas e digests, dtype,
   solver, recursos e tempos. Nova fonte/input inicia aquisição identificada;
   replotar um arquivo não reexecuta sua PDE ou um código externo.
5. **Verificar o replay:** usar a matriz de base arquivada em avaliações e
   orientações; repetir com threads BLAS distintas e rotações equivalentes do
   núcleo. Dimensões e hashes de fontes não substituem esse contrato.
6. **Propagar correções:** corrigir o dono compartilhado, testar a invariante,
   auditar os consumidores e readquirir os casos afetados. Não usar workaround
   de notebook, exclusão de cobertura ou tolerância relaxada para esconder falhas.
7. **Entregar a evidência:** página atual do caso, dados leves, figuras e notebook
   executado. Mostrar macro malha nos campos/erros e interseções nos perfis,
   preservar valores unilaterais e inspecionar legibilidade no tamanho final.
   Declarar projeto/módulo/revisão/URL da referência e distinguir inspeção de execução.

## Ambientes e gates de qualidade

Usar Pixi 0.76.2 e os lockfiles versionados do workspace e da integração AmgX.
Conferir `pixi --version` e executar `pixi list --locked --no-install -e test-core`
antes de preparar um ambiente; mudanças de dependências exigem atualizar manifesto
e lockfile juntos. Para preparar pela primeira vez o ambiente
completo Linux com duas GPUs:

```bash
pixi install --locked -e test
pixi run --locked -e test test-setup-amgx
pixi run --locked -e test test-dependencies
```

Antes de aceitar alterações:

```bash
pixi run --locked -e test lint
pixi run --locked -e test format-check
pixi run --locked -e test typecheck
pixi run --locked -e test test-cov
pixi run --locked -e docs docs-check
pixi run --locked -e packaging lock-check
pixi run --locked -e packaging ci-check
pixi run --locked -e packaging metadata-check
pixi run --locked -e test build
pixi run --locked -e test check-dist
```

O runner usa todos os CPUs disponíveis, uma thread numérica por worker e uma
segunda fase para `@pytest.mark.serial`. Marcar como serial apenas testes que
precisam de exclusividade; custo elevado sozinho não justifica a marcação.
Linhas e ramos mantêm gates independentes de **99%**. Conservar testes de
invariantes e integrações reais; reduzir combinações redundantes sem esconder
casos numéricos distintos. O ambiente completo não deve aprovar ausência de
dependências como validação de backend.

Os workflows [Tests](https://github.com/volpatto/pymhm/actions/workflows/tests.yml),
[Lint and Quality](https://github.com/volpatto/pymhm/actions/workflows/lint-and-quality.yml)
e [Docs](https://github.com/volpatto/pymhm/actions/workflows/docs.yml) têm
responsabilidades próprias e rodam independentemente em PRs e pushes de main.
As verificações usam os dois workspaces travados e são reutilizáveis na release.
A suíte completa com duas GPUs exige dispatch manual de Tests com `full_native`
habilitado. O [workflow de release](https://github.com/volpatto/pymhm/actions/workflows/publish-pypi.yml)
executa Tests e Quality em paralelo; depois Docs valida e publica o site; por
fim, PyPI publica os artefatos verificados. Configurar o trusted publisher para
`publish-pypi.yml`, a origem Pages como GitHub Actions e o ambiente `github-pages`
para aceitar tags `v*`. A galeria usa figuras de
publicação selecionadas e versionadas; campos grandes e intermediários continuam
fora do Git. A aquisição científica, execução de notebooks, aceite de campos e
figuras e inspeção MathJax permanecem verificações separadas.

Nos alvos portáteis, executar `test-core`, `test-py311` e `test-py312` conforme a
matriz de plataformas; essas execuções não substituem as integrações nativas.
A CI cobre Linux x86-64, Windows x86-64 e macOS Apple Silicon (ARM64). A resolução
macOS x86-64 permanece disponível no lockfile para uso local, sem validação
automática nesse alvo.

Para notebooks UFL, usar `introduction`; para as demais campanhas, o perfil
indicado na página do caso. Executar os notebooks afetados com `notebooks-run`
e atualizar seus catálogos. Não exigir campanhas pesadas não afetadas em cada mudança.

Inspecionar MathJax no navegador: `docs-check` valida markup, mas o build não
compila TeX. Usar displays em blocos `$$` isolados e equações longas alinhadas.
Ver [desenvolvimento](docs/development.md) e
[verificação](docs/verification.md) para os procedimentos mantidos.

## Primeira entrega a executar

- [ ] Selecionar a matriz mínima Brinkman de R1, com critérios por campo,
  espaços admissíveis e custo estimado por configuração.
- [ ] Fechar primeiro as camadas analíticas 2D de MHM e MHM-USFEM; ampliar para
  3D e extremos após os controles do operador e da referência.
- [ ] Registrar tabela de resolução/erro, campos e limites nas páginas dos
  casos e no notebook correspondente.
- [ ] Atualizar R1 com a evidência aceita e a próxima configuração pendente;
  executar os gates sem alterar os critérios para obter aprovação.

## Scientific scope and acceptance criteria

PyMHM separates local finite element problems from an explicit skeletal coupling.
A supported family needs a concrete operator, compatible spaces and numerical
evidence. An optional dependency or a generic matrix adapter alone does not
supply a finite element discretization.

The [initial convergence catalogue](docs/cases/minimal-convergence.md) reports the
current short studies, including unresolved numerical increments and incomplete
levels. The extended historical campaigns and performance sweeps are separate
from that evidence. The [case evidence guide](docs/cases/index.md) maps representative calculations to the
[literature](docs/literature.md). Its labels distinguish published comparisons,
matched data, independent references and analytical verification. The selected
cases do not establish complete reproduction of the MHM literature.

### Implemented constructions

The table identifies implemented paths; applicable degrees, boundaries, material
assumptions and executed evidence are specified on their linked pages.

| Construction | Concrete scope and limitations |
| --- | --- |
| Local/global hybrid algebra | Independent trial/test couplings and adjoint kernels, retained physical modes, constrained local solves, source lifts, offline/online reuse and [recursive MHM](docs/cases/nested.md). Supplied operators require their own kernel and stability checks. |
| Primal Darcy | Triangular Pk, rectangular Qk and tetrahedral Pk local pressure, physical pressure means and scalar/SPD permeability; [2D](docs/api/darcy.md), [3D](docs/api/darcy3d.md) and [Cartesian](docs/quadrilateral.md) conventions are explicit. |
| Mixed Darcy | Triangular RT/BDM, enriched rectangular RT, affine tetrahedral/prismatic families and mapped hexahedral RT. [Normal traces, pressure moments and interior enrichment](docs/api/darcy3d.md) remain independent choices. Nonaffine prisms and pyramidal mixed elements are outside this path. |
| Material interfaces | Exact integration over Cartesian/planar intersections, material-fitted local meshes and explicit macroface partitions. [Unfitted](https://github.com/volpatto/pymhm/blob/main/docs/cases/unfitted.md) integration alone does not resolve a gradient jump in an uncut polynomial cell. |
| Reconstruction and estimation | RT moment recovery, Oswald potentials and distinct published, energy-weighted and face-jump indicators. [Estimator hypotheses](https://github.com/volpatto/pymhm/blob/main/docs/cases/reconstruction3d.md) require `k >= ell + d`; algebraic RT construction is weaker. Continuous-test equilibrium and fine-cell DG balance are different conditions. |
| Alternative multiscale formulations | [MsHHO](docs/cases/mshho.md), [Robin MH](https://github.com/volpatto/pymhm/blob/main/docs/cases/mh.md), [MH²M](https://github.com/volpatto/pymhm/blob/main/docs/cases/mh2m.md) and [PGMHM](https://github.com/volpatto/pymhm/blob/main/docs/cases/pgmhm.md), with their own face/cell/source conventions. Equivalence and injectivity depend on the stated spaces and source assumptions. |
| RAD and transport | Conservative Pk Galerkin/SUPG, tensor diffusion, reaction and explicit coefficient derivatives; [MHM-USFEM](https://github.com/volpatto/pymhm/blob/main/docs/cases/unusual.md) uses the full unusual residual form. Stabilization does not imply a maximum principle. |
| Transient scalar problems | Backward Euler, positive capacity, changing loads/boundaries and prepared spatial operators. [Darcy coupling](https://github.com/volpatto/pymhm/blob/main/docs/cases/transient-transport.md) distinguishes volume flux from the numerical normal trace. Manufactured time convergence does not reproduce an unavailable random realization. |
| Stokes–Brinkman/Oseen | Taylor–Hood and full-residual equal-order USFEM in 2D/3D, tensor resistance, prescribed convection, component slip and physical pressure gauges. [Flow adaptation](https://github.com/volpatto/pymhm/blob/main/docs/cases/stokes-adaptive.md) assumes constant viscosity, full Dirichlet data, uniform trace degree and resolved jumps. No nonlinear Navier–Stokes iteration is claimed. |
| Primal and displacement–pressure elasticity | General material tensors, physical traction and three/six rigid modes; GaLS/Taylor–Hood and finite/infinite bulk limits. [Primal displacement](https://github.com/volpatto/pymhm/blob/main/docs/cases/primal-elasticity.md) alone is not uniformly locking-free. |
| Mixed elasticity | Row-wise BDM/enriched/rectangular RT stress, weak rotation, anisotropic compliance and [tetrahedral AFW](https://github.com/volpatto/pymhm/blob/main/docs/cases/mixed-elasticity3d.md). Displacement must represent rigid modes; weak symmetry is a moment condition. Classical AFW stability is not a theorem for arbitrary MHM traces. |
| Polygonal/polyhedral geometry | Straight-sided simple polygons, including nonconvex cells, and [certified star-shaped polyhedra](https://github.com/volpatto/pymhm/blob/main/docs/cases/star-polyhedra.md) with original polygonal face spaces. Empty-kernel cells, cavity shells and curved faces are outside the polyhedral path. |
| Helmholtz | Complex triangular/polygonal Pk and Cartesian Qk locals, polynomial/oscillatory traces, absorbing boundaries, diagonal PML and local resonance checks. [Wave](https://github.com/volpatto/pymhm/blob/main/docs/cases/helmholtz.md) and [Marmousi](https://github.com/volpatto/pymhm/blob/main/docs/cases/marmousi.md) comparisons retain their distinct data contracts. |
| Maxwell and elastodynamics | Tangentially coupled central-DG dynamics with mass-scaled CFL, and Newmark local responses with slabwise traction/substeps. [Maxwell](https://github.com/volpatto/pymhm/blob/main/docs/cases/maxwell.md) and [elastodynamics](https://github.com/volpatto/pymhm/blob/main/docs/cases/elastodynamics.md) state their analytical and heterogeneous comparison scopes separately. |

See the [API](docs/api.md), [FEniCS local forms](docs/fenics.md), [meshing](docs/meshing.md)
and [linear solvers](docs/solvers.md) pages for construction and backend details.

### Historical targets still requiring evidence

Scientific completion is tied to a particular experiment, including its input
fields, mesh, spaces, quadrature, boundary convention, gauge and norm. The following
limits remain material to claims about the literature:

- The [periodic](https://github.com/volpatto/pymhm/blob/main/docs/cases/periodic.md), [oscillatory well](https://github.com/volpatto/pymhm/blob/main/docs/cases/mapped-well-oscillatory.md)
  and [SPE10](https://github.com/volpatto/pymhm/blob/main/docs/cases/spe10-adaptive.md) comparisons retain nonzero classical-reference
  refinement increments and historical input/mesh qualifications. Local or face
  enrichment controls are different discretizations from the published setup.
- The literal L11 mixed-wall curve remains quantitatively different; its ε=1
  control is separate. The [transient random-field example](https://github.com/volpatto/pymhm/blob/main/docs/cases/transient-transport.md)
  requires a complete same-case acquisition and independent reference.
- The [Stokes stress diagnostic](https://github.com/volpatto/pymhm/blob/main/docs/cases/reproduction.md), [GaLS elasticity
  amplitudes/stabilization](https://github.com/volpatto/pymhm/blob/main/docs/cases/elasticity.md) and historical exterior traces in
  [mixed elasticity](https://github.com/volpatto/pymhm/blob/main/docs/cases/mixed-families.md) prevent unqualified reproduction of
  every printed ordinate or table.
- The [Helmholtz angular ordinates](https://github.com/volpatto/pymhm/blob/main/docs/cases/helmholtz.md), [Marmousi material crop
  and 15-case family](https://github.com/volpatto/pymhm/blob/main/docs/cases/marmousi.md), [historical Maxwell incident-wave phase,
  amplitude and turn-on](https://github.com/volpatto/pymhm/blob/main/docs/cases/maxwell-nanoguide.md), and heterogeneous [2017
  elastodynamic source case](https://github.com/volpatto/pymhm/blob/main/docs/cases/elastodynamics.md) have their own unresolved
  data or comparison requirements. Analytical wave tests do not replace them.
- [Historical cluster scaling](docs/execution.md) on 24–768 cores and the largest 3D
  meshes is not reproduced. Available MPI/GPU measurements concern their declared
  hardware and workloads; they do not establish an unmeasured speedup.

The bibliographic catalog and detailed case pages retain the theorem hypotheses
and numerical evidence behind each statement. Historical discrepancies are
reported with independently assembled same-case comparisons where available;
missing reference acquisition programs must be supplied before a clean execution
can reproduce those comparisons.

### Acceptance by literature target

Each row identifies a formulation, its physical case, the available comparison
and the remaining acceptance condition. The [catalog](docs/literature.md) identifies
the publications and reference-code revisions. Numerical summaries retain their
acquisition provenance; reproducing their results requires the corresponding
fields and reference programs. A checked-in summary alone does not verify a new
execution. Analytical tests and local operator comparisons remain separate from
complete published-case comparisons.

| Target and formulation | Case | Comparison and recorded evidence | Remaining acceptance condition |
| --- | --- | --- | --- |
| L01: primal Darcy MHM | Cosine, quarter five-spot, square obstacle and rough coefficients | Analytical fields, published curves, complete square-obstacle Basix/P1 and native NeoPZ/RT0 comparisons, and six full point-well Basix/P2 and RT0 comparisons on stated spaces | Resolve classical obstacle-reference refinement; identify historical random data before literal reproduction. |
| L02: elliptic error estimation | Cosine, inclusion and Dirac wells | [Reconstruction and estimators](https://github.com/volpatto/pymhm/blob/main/docs/cases/reconstruction3d.md), analytical errors and indicator controls | Repeat the same-case estimator comparison and reference refinement; retain the Dirac regularity limitation. |
| L03: abstract hybrid algebra | Local/global and recursive systems | Independent full Petrov–Galerkin algebra, retained kernels and source lifts | User-defined operators still require kernel, compatibility and stability hypotheses; no published numerical table is supplied. |
| L04: periodic Darcy robustness | [Periodic permeability](https://github.com/volpatto/pymhm/blob/main/docs/cases/periodic.md), fixed macrogrid and face enrichment | Phased basis replay, complete 64-macro independent Q1/P0 assembly and current five-level Q1 controls; Q5 accuracy is outside the current acceptance | Resolve local and classical-reference refinement; finite local/trace pairs require injectivity. The historical reference is Q1 on 4096² elements, with unidentified historical local refinement. |
| L05: mixed local Darcy | Rectangular, tetrahedral and prismatic wells | [Mixed wells](https://github.com/volpatto/pymhm/blob/main/docs/cases/mixed-well-geometries.md), analytical fields and native NeoPZ spaces | Complete matched whole-case reference acquisition and interior/face/quadrature controls with compatible divergence spaces. |
| L06: MHM–MsHHO connection | Elliptic macro and skeletal refinement | Analytical MsHHO/MHM comparisons with declared source spaces | Repeat convergence and same-case independent assembly under the equivalence hypotheses. |
| L07: face-based robustness | Periodic medium and SPE10 layer 36 with continuous face interpolation | [Darcy flux](https://github.com/volpatto/pymhm/blob/main/docs/cases/spe10-flux.md), Q1/C0-P1 MHM and Q3/MSL/NeoPZ references | Resolve local/face refinement and the classical-reference increment while retaining the published 66-square macro partition and material. |
| L08: unfitted preprint | Two-layer interface problem | [Unfitted study](https://github.com/volpatto/pymhm/blob/main/docs/cases/unfitted.md), analytical series and independent UFL | Identify preprint-specific data and hypotheses separately from final L10; the final regularity endpoint differs. |
| L09: H(div) recovery and adaptivity | [Adaptive SPE10](https://github.com/volpatto/pymhm/blob/main/docs/cases/spe10-adaptive.md) | Published P2/r2/P0 spaces, RT2/Oswald recovery, full independent initial/final UFL systems and refined RT2 classical references | Reduce remaining reference sensitivity and qualify historical BAMG connectivity; estimator estimates require `k >= ell + d`. |
| L10: unfitted flux approximation | Smooth h/p sweeps and two-layer contrast | [Unfitted study](https://github.com/volpatto/pymhm/blob/main/docs/cases/unfitted.md), analytical norms, native assemblies, P8 r32→r64 endpoints for ell0/ell1 and complete twelve-case independent UFL S0/S2 contrast systems | Resolve local error for higher enriched traces on admissible spaces; P8/r16–P3/s32 has an exact multiplier kernel. Historical Figure-7 S2 parameters remain unspecified despite same-discretization agreement for the declared cases. |
| L11: advective/reactive MHM | Mixed walls and random Darcy–transport | [Transport](https://github.com/volpatto/pymhm/blob/main/docs/cases/transient-transport.md), analytical and coefficient controls | Acquire the complete §5.4 realization, coupled trajectory and independently refined space/time/quadrature reference. |
| L12: generalized RAD | Polygonal/polyhedral diffusion and reaction layers | [Polytopes](https://github.com/volpatto/pymhm/blob/main/docs/cases/polygons.md), conditioning, boundary and layer controls | Repeat conditioning/layer/star-polyhedron studies and independent fields for the matched cases. |
| L13: equal-order Stokes–Brinkman | Smooth flow and SPE10 layer one | [SPE10](https://github.com/volpatto/pymhm/blob/main/docs/cases/spe10.md), published-space fields, analytical polynomial-family layer convergence and a five-level independent Taylor–Hood baseline with physical field integration | Assess remaining local/trace and classical-reference sensitivity, qualify missing historical mesh/stabilization choices, and complete 2D/3D extreme-regime validation. |
| L14: multilevel flow estimator | [Stokes adaptation and cavity](https://github.com/volpatto/pymhm/blob/main/docs/cases/stokes-adaptive.md) | Analytical indicators and classical cavity comparisons | Repeat adaptation and reference refinement with the same constant lid, physical pressure gauge and corner cutout. |
| L15: adaptive Oseen | Smooth and boundary-layer flow | [Oseen](https://github.com/volpatto/pymhm/blob/main/docs/cases/oseen.md), analytical refinement and independent operators | Repeat layer/adaptive whole-case controls, quadrature and physical norms. |
| L16: flow a priori analysis | Admissible Stokes–Brinkman spaces | Analytical/native flow verification and stated degree conditions | Verify the theorem's discretization and regularity conditions for each rate claim. |
| L17: primal elasticity | Analytical displacement and HPC4E | [HPC4E](https://github.com/volpatto/pymhm/blob/main/docs/cases/hpc4e.md), analytical fields, independent DOLFINx/UFL RT1/RT2 stress–displacement–rotation references, physical equilibrium/work checks and RT2 spatial refinement | Reduce or quantify remaining reference sensitivity, especially rotation and compliance, while preserving the published skeletal spaces and full-stress norms. |
| L18: weakly symmetric mixed elasticity | Tensor families and oscillatory Table 3 case | [Mixed elasticity](https://github.com/volpatto/pymhm/blob/main/docs/cases/mixed-families.md), native fields and analytical norms | Repeat BDM/RT/enrichment and tensor controls; retain the printed rotation-column inconsistency explicitly. |
| L19: locking-free elasticity | Finite/infinite bulk limits | [GaLS elasticity](https://github.com/volpatto/pymhm/blob/main/docs/cases/elasticity.md), mixed-field and native comparisons | Execute matched Lamé sweeps and refinement; an affine primal patch does not establish uniform locking freedom. |
| L20: scalable implementation | Distributed local/global execution | [Execution](docs/execution.md), spawn/MPI contracts and workload-specific measurements | Reproduce setup, transfers, synchronization and scaling at matched accuracy; historical cluster scaling remains unverified. |
| Additional MH | Robin hybrid diffusion | Analytical boundary/parameter sweeps and independent UFL | Repeat complete refinement; distinguish the Robin multiplier from physical flux. |
| Additional MH²M | [Oscillatory medium](https://github.com/volpatto/pymhm/blob/main/docs/cases/mh2m-heterogeneous.md) | Crisscross/diagonal cases and refined independent CG3 fields | Repeat same-case acquisition, overlay quadrature and baseline refinement; historical curve differences remain. |
| Additional PGMHM | Inclusions and SPE10 | Analytical/native equations and separately refined classical fields | Acquire matched whole-case comparisons with published enrichment and heterogeneous stabilization. |
| Additional Unusual | [Reaction–diffusion](https://github.com/volpatto/pymhm/blob/main/docs/cases/unusual.md) and SPE10 layers | Analytical/native fields and recorded FreeFem comparisons | Reacquire the full external comparison and verify material/reaction-length/trace resolution. |
| Additional Helmholtz | [Angular and stability studies](https://github.com/volpatto/pymhm/blob/main/docs/cases/helmholtz.md), [Marmousi](https://github.com/volpatto/pymhm/blob/main/docs/cases/marmousi.md) | Plane/Hankel fields, native saddle systems and classical material-crop references | Repeat published angular/stability sequences; validate Marmousi storage phases and reference refinement while retaining historical-input limits. |
| Additional Maxwell | [Nanoguide](https://github.com/volpatto/pymhm/blob/main/docs/cases/maxwell-nanoguide.md) | Analytical dynamics, native operators, independent central-DG Q2 reference through 1024², matched staggered-time fields, CPU/GPU equivalence and space/time/material controls | Quantify remaining reference and MHM temporal sensitivity over the declared trajectory; identify historical phase/amplitude/turn-on before claiming literal image reproduction. |
| Additional elastodynamics | [Equation (53) and three layers](https://github.com/volpatto/pymhm/blob/main/docs/cases/elastodynamics.md) | Analytical Newmark trajectories, complete independent 341-macro original equations and 301-state common-basis coordinate comparisons for the selected three-layer case | Evaluate both executed field bases separately, verify the conforming reference on several finer meshes and complete space, time and quadrature refinement. Historical heterogeneous inputs remain unresolved. |

### Numerical acceptance

1. Verify geometry, quadrature, basis orientation, local/adjoint kernels,
   constrained solves, trace signs, physical gauges and admissible spaces.
2. Derive manufactured sources and boundary data from the actual operator.
   Verify original physical equations and saved-field replay, rather than only
   a scaled CSR residual or a patch solution.
3. Refine macro, local and skeletal spaces independently. Measure physical
   pressure/displacement and flux/stress separately, preserving broken traces.
4. Compare matched fields and norms with an independent assembly. Without an
   exact solution, also refine a classical reference and report its own increments.
5. Claim a published reproduction only with matching data and discretization,
   stated extraction uncertainty and compatible rate hypotheses. Singular forcing
   or heterogeneous materials do not inherit smooth-problem rates automatically.

Line and branch coverage of at least 99% is a software quality gate, not a proof of
accuracy, inf-sup stability or literature completion. Optional API contract tests
are accompanied by native integrations on available platforms.

### Execution and distribution

[Execution](docs/execution.md) and [performance](https://github.com/volpatto/pymhm/blob/main/docs/performance.md) record setup, assembly,
factorization, repeated loads, global solve, reconstruction, transfers,
synchronization and peak memory. Compare backends at equal accuracy and reproducible
thread settings. Portable spawn semantics, MPI partitions, operator reuse and
resident GPU batches retain their explicit numerical and platform restrictions.
AMG on admissible positive systems and block preconditioning of saddle systems are
distinct paths; backend availability does not establish acceleration.

The portable core imports without optional FEM, CAD, MPI or accelerator runtimes.
Platform claims require native tests. PyPI readiness requires checked wheel/sdist,
clean-install verification and synchronized metadata; release automation does not
mean an upload has occurred. Conda-forge additionally requires an accepted recipe
and available dependencies on its target platforms.

Validated release tags call the dedicated Tests and Quality workflows in parallel,
then build and deploy Docs, then publish the checked package to PyPI.
Maintainers configure the PyPI trusted publisher for `publish-pypi.yml`, GitHub
Actions as the Pages source and the `github-pages` environment's release-tag
deployment rules. The
published gallery uses selected versioned figures; large field and intermediate
archives remain outside Git, and documentation is excluded from Python and Conda
installation artifacts.

Both release formats include the complete runtime and typing files. The source
archive also contains only the build configuration, README, license and generated
metadata required for installation. Tests, scientific examples, notebooks,
documentation and environment definitions are maintained in the repository.
Optional backend adapters ship with the runtime; their native dependencies are
installed separately for the supported target platform.
