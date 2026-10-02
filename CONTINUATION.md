# PyMHM: retomada com um clone leve, sem transferência de resultados

Estado revisado em **2 de outubro de 2026**. Roteiro operacional, separado dos
docs científicos e excluído dos pacotes distribuídos.

**A implementação integral e a reprodução integral da literatura ainda não estão
concluídas.** Faltam controles de resolução, comparações independentes completas
e alguns drivers de casos. A retomada parte exclusivamente das fontes versionadas:
nenhum campo, checkpoint, ambiente ou comparador da máquina anterior será enviado.
As quedas interromperam execuções; sua causa não foi estabelecida.

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

Os drivers monolíticos precisam dessas melhorias antes de suas maiores execuções.

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
registram fontes/DOIs, formulações e limites. Reobter os artigos pelas fontes
citadas quando for necessário conferir uma hipótese; o núcleo não precisa da
biblioteca de PDFs para funcionar.

MSL (`msl_mhm`, `msl_cg`, `msl_core`), `msl_mfem`, `mhm-mfem` e os códigos de
Santiago não acompanham o clone. MSL exige obter acesso privado novamente.
NeoPZ/MHM/iMRS têm fontes públicas Labmec. Obter código externo em árvore
ignorada, por exemplo `build/reference-sources/`, fixar revisão/URL e manter
comparadores fora da distribuição. Inspeção não conta como execução. Sem acesso
a MSL, construir uma montagem independente DOLFINx/UFL, Basix ou NeoPZ do mesmo
caso completo, declarando o limite de acesso.

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
Isso não é uma execução das campanhas: **solves grandes não foram executados
nesta preparação**. Executar um grupo por vez. Valores pequenos são pilotos,
não substitutos do artigo. Comparações devem usar arquivos produzidos na nova
aquisição, não JSONs antigos sem seus campos.

## 5. Periódico e interfaces internas

### L04 periódico

```bash
pixi run -e test python examples/verify_periodic.py --reference-levels 32 --macro 8 --local-refinement 32 --segments 1 2 4 8 16 32 --workers 1 --native-threads 1
pixi run -e test python examples/periodic_reference.py --sizes 32 64 --order 10 --native-threads 1
```

Antes de r512, implementar fases e replay públicos validados em r32. Persistência
anterior não está disponível e sua aceitação não estava concluída. Repetir a
primeira chamada com r128/r256/r512: Q1 local/P0 por segmento, 64 macros. Referência
32 só evita uma montagem grande simultânea; não é baseline final.

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
pixi run -e intel python -m examples.unfitted_convergence --study smooth --refinement 16 --degree 8 --maximum-segments 32 --workers 1 --local-solver pypardiso
pixi run -e intel python -m examples.unfitted_convergence --study contrast --refinement 16 --workers 1 --local-solver pypardiso --contrasts 10 100 1000 10000 100000 1000000
```

Repetir local r24/r32/r64 mantendo dados e traços. Comparar r32→r64 com
`python -m examples.unfitted_local_resolution first second --order 13 --output ...`.
Implementar seleção de quadratura no módulo compartilhado: o CLI atual usa `degree+3`, logo P8
monta q11, sem flag q13. Resolver o incremento local de ell2 e a dependência
q9/q11 de ell3; conferir q13 sem afrouxar tolerância. Integração exata de K não
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

```bash
pixi run -e test python -m examples.transport_mixed_campaign --workers 1
pixi run -e test python -m examples.transport_coefficient_controls --epsilon 0.1 --resolutions 8 16 32 64 128 --local-refinement 16 --workers 1 --local-workers 1
pixi run -e test python -m examples.transport_face_resolution --local-refinement 32 --workers 1
pixi run -e notebooks python -m examples.transport_campaign --collect
```

Não são o caso aleatório acoplado §5.4. Criar driver: material exponencial 64×16,
realização/seed salvos; Darcy primal P3/traço P2 em 2 segmentos; transporte
Galerkin P3/traço P2 em 8 segmentos; inflow=1/IC=0/fonte=0/T=7 e controle de dt.
APIs `darcy_transport`/`scalar_transient` fornecem `output_steps`/`on_step`; callback
não restaura integrador sozinho. Validar fluxo normal/volumétrico físico,
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
Além disso, ele lê os manifestos de todas as famílias antes de filtrar o notebook
escolhido: um manifesto ausente de outra família pode interromper o inventário.
Ajustar o inventário para seleção por notebook quando necessário, sem contornar
as conferências dos hashes.

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

Nesta preparação foram verificados scripts/infraestrutura, seleção leve e
artefatos. Não foram repetidas cobertura completa, campanhas científicas nem
todas integrações nativas. Não usar percentuais históricos como resultado atual.
A conclusão depende da matriz científica e das comparações acima.
