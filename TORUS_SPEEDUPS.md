# Torus engine speed-ups (branch `torus-landing-speedup`)

Marc Suchard's optimisation plan for `TorusFourierTreeLikelihood` (steps 0 to 4 plus the store/restore fix), and the
savings specific to the landing jump law, implemented on the landing engine.

**Wherever the paper engine was correct, every log-likelihood and every ancestral draw is bit-identical to it.** The
paper engine is the one behind the paper's production runs. The two deliberate changes below alter no number in an
ordinary MCMC run. Three pre-existing defects are fixed, and those fixes do change values, but only in configurations
the paper's runs never used. The one other exception is the sign bit of NaN on degenerate inputs, which the paper
engine itself does not reproduce from one JIT history to the next.

Relative to the paper engine (`838e4ebfb`), the speed-up commits change only `TorusFourierKernel.java` and
`TorusFourierTreeLikelihood.java`, plus tests. Against `torus-split-tip` the branch also carries the landing law
(`2822dd114`, `838e4ebfb`). It changes those same two files, adds `TorusLandingPruning`, and changes
`TorusLogSpacePruning` and the parser. So review `838e4ebfb..torus-landing-speedup` for the speed-ups, and
`torus-split-tip..838e4ebfb` for what the landing law changed in the code the plan was written against.

## Branch

| Commit | What it is |
|---|---|
| `0728de732` | `torus-split-tip`, the code the plan was written against |
| `2822dd114` | Landing jump law (pure-Fourier tips, stationary landing root, `TorusLandingPruning` reference, FFT landing ancestral sampler). A work-in-progress checkpoint: its known ancestral-sampler defect is fixed in the next commit |
| `838e4ebfb` | **Paper engine.** The exact source of the jar behind the paper's production runs. Compiled with OpenJDK 11's javac (no `-target`), it gives 17 classes byte-identical to that jar. Every comparison below is against this commit |
| `b088c3128` | Store/restore fix: a node's buffer flips at most once per store/restore cycle, as in `BufferIndexHelper`. A branch-rate event carrying a node index flags only that node. `TorusCacheFuzzTest` (step 0) |
| `8a47e0076` | Step 4: planned FFT |
| `036210dbb` | Landing: per-(Sigma, lambda) mode tables, exponentials once per mirror pair, cached tip coefficients, pooled scratch |
| `d51f17e86` | Step 4: butterfly stages applied two at a time |
| `b258ab376` | Steps 1, 1b and 2: screened image sum, Brownian-tip shortcut, characteristic function tabulated once, clean residues skipped |
| `017b52a9b` | Step 3: per-branch message cache |
| `4e2396818`, `86d226a06`, `10824e266` | Fixes from an automated adversarial review (memory budget, exactness at extreme inputs, unsignalled branch-length changes, shared precision, exception handling in the parallel sweep) |
| this commit | These notes, two benchmark harnesses and a golden-output driver, test wording, and a platform guard on the regression test. One comment changes in the engine, on the same lines, so the compiled classes are unchanged |

## How each step was kept exact

- **Step 4, planned FFT.** The plan's version changed results at rounding level. Here `fftN` tabulates the twiddle
  factors that `four1`'s own recurrence produces, and its bit-reversal swap list, once per kernel. Every butterfly is
  the same operation on the same operands in the same order. Contiguous lines are transformed in place. Strided lines
  are gathered eight at a time into contiguous scratch. Two butterfly stages are applied per pass, since stages h and
  2h map each set {q, q+h, q+2h, q+3h} onto itself. `TorusPlannedFftTest` compares raw bits against the old routine
  up to nGrid 2048 in 1-D, 256 in 2-D, 32 in 3-D and 8 in 4-D, on Gaussian, nonnegative, signed-zero, huge,
  subnormal and sparse inputs. The 2-D grids of 512 and 1024 are covered end to end, by the 402-tip comparison below.
- **Step 1, screened image sum.** Images are screened through q(b + s) = q(b) + s'(A + A')b + s'As. Only images
  whose term underflows to exactly 0.0 are skipped, with a margin that covers the rounding of the screen and of the
  direct form. The screen is not used where a term could overflow. `TorusAtomScreeningTest` checks it against the
  exhaustive loop, kept verbatim, over 1,093 cases (1 to 3 dimensions, correlations to 0.999, eigenvalues 1e-10 to 10,
  t from 1e-8, covariances near 1e-305, observations outside (-pi, pi]).
- **Step 1b, Brownian tips.** Without jumps a tip's spectral half is all signed zeros, so it is skipped. The shortcut
  is taken only when t, every Brownian multiplier, every phase m.y and every tip-damping factor is finite. Otherwise
  the split path runs and gives the paper engine's NaN. `TorusAtomScreeningTest` compares the two on 300 cases,
  invalid inputs included.
- **Step 2.** The jump characteristic function is tabulated once per likelihood, built on first use and checked
  against the contents of the arrays it came from. Residues with nothing flagged are skipped before their exponent is
  built.
- **Step 3, per-branch messages.** The message a node sends up its branch is double-buffered per node and residue,
  and the flags take `TreeDataLikelihood`'s meaning (a flagged node's branch changed). Partials are formed from the
  children's messages only where needed (a recomputed message, the root, ancestral sampling), with the arithmetic and
  order of the old per-node cache. A message is also recomputed when it is not valid for the node's current branch,
  which covers a former root after a topology move.
- **Landing.** `lambda + a_m` and `lambda / (lambda + a_m)` are tabulated per residue while Sigma and lambda are
  unchanged bit for bit. exp and expm1 are evaluated once per mirror pair m, -m, whose quadratic forms are
  bit-identical (the tables check this before relying on it). Tip Fourier coefficients depend only on the tip angles
  and Sigma_eps. They are computed once, checked against Sigma_eps at every evaluation, and checked against the tip's
  angles whenever the tip's message is rebuilt.

## Verification

All on x86_64 Linux with OpenJDK 11, except where stated.

| Check | Result |
|---|---|
| The 75 torus tests that existed, plus 17 new ones: `TorusCacheFuzzTest` (11), `TorusPlannedFftTest` (2), `TorusAtomScreeningTest` (3), `TorusPaperEngineRegressionTest` (1) | all 92 pass on the final build |
| `TorusCacheFuzzTest`: BEAST's own tree operators and parameter moves, with one or two moves between store and restore, each evaluated (except in one step in eight, which restores without evaluating). Cached likelihood and ancestral draws must equal a full recompute bit for bit. Both laws, 1 to 3 dimensions, forced root changes, shared precision, `ScaledTreeLengthRateModel`, Brownian shift with tip error | pass. On the paper engine 5 of the 11 fail: the two-evaluation cases (177 of 311 and 90 of 296 log-likelihoods wrong), the root-change test, which also evaluates twice (146 of 561), shared precision (102 of 221) and `ScaledTreeLengthRateModel` (149 of 217) |
| `TorusPaperEngineRegressionTest`: eight seeded chains (both laws, dimensions 1 to 3, root moves, parallel residues, branch rates, nGrid 256) | digests equal those of the paper engine's jar. On Java 8 the two engines also give identical digests, though different ones from Java 11 in 7 of 8 scenarios, so the test checks only on Java 11 x86_64 |
| `TorusGolden` (included): 2,668 lines, values as raw bits and arrays as SHA-256 digests of their bits. Kernel operations up to nGrid 512 in 1-D, 128 in 2-D and 16 in 3-D. Seven likelihood chains with per-branch landing audit counts | identical to the paper engine, except a recompute counter in the two branch-rate chains (deliberate) |
| Production trait XMLs (153 tips, nGrid 256, landing), 40 seeded moves with store and restore (`XmlLikelihoodBench`, included) | every value identical |
| Full BEAST chains on three production XMLs, 1,500 states, `fullEvaluation="1000"` | logs identical row by row, on the final build |
| Seeded ancestral sampling on a 224-tip tree (nGrid 128, five draws of every internal node) | identical, on the final build |
| An automated adversarial review: four review agents, each finding checked by a verifier agent. Then every probe was rerun on the fixed builds (kernel probes on `4e2396818`, cache and concurrency probes on `86d226a06`). `10824e266` only reorders the constructor | all findings fixed in `4e2396818`, `86d226a06` and `10824e266` |

## Timing

All on one Linux workstation (64-core AMD Threadripper PRO, CPU only, OpenJDK 11). The production and 402-tip pairs
ran side by side, so both engines saw the same load. The random-tree runs ran one after the other.

**The paper's production setting.** One residue, a fixed 153-tip tree, nGrid 256, landing law, one core. Every
proposal changes Sigma or lambda, so every evaluation is a full recompute. The cache cannot help, and the gain comes
from the FFT, the landing tables and the tip coefficients.

| | Paper engine | New engine | Speed-up |
|---|---|---|---|
| Residue 143, per evaluation (an MCMC state costs at most this, since out-of-range correlation proposals are rejected without one) | 2,112 ms | 738 ms | 2.86x |
| Residue 155 | 2,348 ms | 788 ms | 2.98x |
| Residue 100 | 2,355 ms | 795 ms | 2.96x |
| 300-state BEAST runs including start-up | 585-624 s | 208-210 s | 2.8-3.0x |

**The plan's problem size.** A random 32-tip tree, 100 residues, nGrid 64, 420 proposals (`RandomTreeBench`,
included), run here with `-Xmx12g -Djava.util.concurrent.ForkJoinPool.common.parallelism=8`, so 8 pool workers plus
the calling thread. Median ms per evaluation, with identical log-likelihood digests from both engines:

| Proposal | Landing: paper | Landing: new | Speed-up | Shift: paper | Shift: new | Speed-up |
|---|---|---|---|---|---|---|
| Node height | 70.9 | 17.1 | 4.1x | 245.1 | 65.1 | 3.8x |
| Subtree slide | 72.8 | 17.8 | 4.1x | 247.2 | 54.2 | 4.6x |
| Narrow exchange | 76.1 | 19.6 | 3.9x | 270.0 | 77.1 | 3.5x |
| Wide exchange | 136.1 | 34.2 | 4.0x | 456.3 | 111.5 | 4.1x |
| Wilson-Balding | 118.1 | 31.1 | 3.8x | 406.2 | 101.9 | 4.0x |
| One residue's precision | 26.8 | 9.9 | 2.7x | 118.0 | 80.3 | 1.5x |
| Full recompute | 342.0 | 121.8 | 2.8x | 1,376.2 | 1,050.7 | 1.3x |

Under the shift law, precision moves and full recomputes gain least, and every factor is below those of the plan's
prototypes (6.7x on tree proposals, 4.4x on a precision move and 3.5x on a full recompute, on a 10-core M1 Max with
JDK 18). The likely reason is that the screen here skips only images whose terms are exactly zero, where the plan's
screen also skipped those far below rounding.

**A larger tree and finer grids.** One residue on a fixed 402-tip tree, landing law, one core with SerialGC
(`XmlLikelihoodBench`). Values were bit-identical between the engines:

| nGrid | Paper engine, per evaluation | New engine | Speed-up | Peak RSS, paper / new |
|---|---|---|---|---|
| 512 (three residues, `-Xmx26g`) | 28.1 s | 10.1-10.3 s | 2.75x | 3.7 / 5.9 GiB |
| 1024 (paper `-Xmx48g`, new `-Xmx110g`) | 112.0 s | 41.8 s | 2.68x | 8.4 / 20.9 GiB |

At nGrid 1024 the tip-coefficient cache is granted only from `-Xmx101g`, because of the conservative accounting
below. Without it (`-Xmx48g`) an evaluation takes 48.3 s at 14.6 GiB.

## Behaviour changes, and defects fixed on the way

- **Deliberate.**
  - The flip-once guard: a second evaluation between store and restore now rewrites the proposal's buffer instead of
    the stored one. The fault it fixes was reachable in the paper's runs, but only in BEAST's full-evaluation phase
    (the first 1,000 states), where a full re-evaluation follows every restore. So it changed no logged value.
  - A branch-rate event with an index flags only that node, so its branch and the path to the root are recomputed,
    instead of every node. `AbstractModel` relays a variable change as (variable, parameter index), so the index is
    not always a node number. The per-evaluation branch-length check below catches the real change in that case.
- **Pre-existing defects, fixed.** None was reachable in the paper's runs.
  - Residues sharing one precision matrix: a change to it dirtied only the first of them.
  - Branch lengths that change without a matching event: `ScaledTreeLengthRateModel` rescales every branch after a
    tree move. Every branch length is now computed once per evaluation, and any branch whose length differs bit for
    bit from the cached one is recomputed, whatever events were fired.
  - With 8 or more residues (the parallel sweep), an exception in one residue could reach the caller while other
    residues were still writing. Now, after every residue has stopped, the lowest-numbered residue's exception is
    rethrown and any others are dropped. The serial sweep still stops at the first.
- **Diagnostics only.** The kernel's cumulative landing audit counters (`getLandingNonpositiveCount` and its siblings)
  count fewer grids under tree moves, because unchanged messages are reused. They count more after any exception
  inside a residue's sweep (a strict-mode audit, an invalid partial or a failed branch operation), because sibling
  messages are computed before the failure surfaces. Per-branch counts, values and the exception thrown are identical.

## Memory

The message cache keeps two grids per observed tip and per non-root internal node, where the old engine kept two per
internal node, root included. The landing tip-coefficient cache adds two grids per observed tip. That cache is limited by a process-wide budget: an
eighth of the maximum heap by default, or `-Dtorus.landing.tipCacheMB` (0 turns it off). It is granted only while
messages and cache together fit in half the heap. Grids of 256 KiB and more are charged double. From 512 KiB, G1 can
store a grid as a humongous object that takes up to twice its size, and the lower threshold adds margin. One residue at nGrid 256 on 153 tips peaks at 0.6-1.6 GiB RSS under `-Xmx3g`.
Multi-residue runs need about 2-3x the old engine's heap.

## Reproducing

```bash
ant compile-all
CP="build:src:$(ls lib/*.jar | tr '\n' ':')"     # src/ supplies beast.properties for BeastParser
for T in TorusFourierTest TorusLandingTest TorusLandingAncestralTest TorusAncestralSamplerTest \
         TorusCacheFuzzTest TorusPlannedFftTest TorusAtomScreeningTest TorusPaperEngineRegressionTest; do
  java -Xmx8g -cp "$CP" org.junit.runner.JUnitCore test.dr.evomodel.treedatalikelihood.continuous.$T
done
# timing in the plan's setting (shift as well as landing), and golden output for diffing two builds
java -Xmx12g -Djava.util.concurrent.ForkJoinPool.common.parallelism=8 -cp "$CP" \
     test.dr.evomodel.treedatalikelihood.continuous.RandomTreeBench landing 64 100 32 420
java -cp "$CP" test.dr.evomodel.treedatalikelihood.continuous.TorusGolden all > golden.txt
```

The suites take about 75 s in all. A hand-rolled javac build needs `-sourcepath src`, so that `beagle.basta` compiles
from `src/beagle/basta`.

To run the harnesses against the paper engine, check out `838e4ebfb` (for example `git worktree add ../paper-engine
838e4ebfb`) and copy in `RandomTreeBench.java`, `TorusGolden.java`, `XmlLikelihoodBench.java` and
`TorusCacheFuzzTest.java`, which the first two need, and `TorusPaperEngineRegressionTest.java`. The golden output
should then differ from this branch's only in the two `recomputes` lines. `TorusPaperEngineRegressionTest.main()`
prints the digests of whichever build it runs on.

`XmlLikelihoodBench file.xml` times a trait XML under the moves of a production chain. It expects the ids `tfl`,
`lambda`, `diag_0` and `off_0`. The production XMLs are not in the repository.

## Not done

These change results at rounding level, or are larger rewrites, so they were left out:
- a real-input FFT with half-spectrum passes
- fixed spectral bands (at the paper's tip error of 6.5 degrees, tip coefficients beyond about |m| = 78 are below 1e-17
  of mode 0)
- threading within one residue. Threading the FFT lines and the elementwise per-mode passes would be exact. The landing
  reset sum and the root integral are compensated sums over all modes in index order, so they must stay sequential
  to keep the bits. This matters most for one residue on a large tree or a fine grid, where the residue loop gives
  no parallelism.
