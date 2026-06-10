# DensityFPSI — Fuzzy Private Set Intersection under Density-Bounded Assumptions

Research artifact for the paper *"Fuzzy Private Set Intersection under
Density-Bounded Assumptions"* (anonymous IEEE S&P 2027 submission).

> **Anonymity notice.** This repository accompanies a double-blind submission.
> It contains no author, affiliation, or other identifying information. Please
> keep it that way until the camera-ready / artifact-evaluation stage.

## Overview

Existing fuzzy PSI (FPSI) protocols rely on **separation-based** assumptions —
*"items are at least 2δ apart"* (which costs `O(2^d)` and only works for tiny
`d`) or *disjoint projection* (which is only justified for high-dimensional
data). No single separation-based protocol handles both regimes.

This work replaces separation with a **density-bounded** assumption, the
**C-max collision property**: items may collide inside a δ-ball, but the
multiplicity of collisions *per axis* is bounded by `C`. Under this assumption
we build the first FPSI protocol for the ℓ∞ metric that:

- handles **low- and high-dimensional regimes simultaneously**,
- scales **linearly** in both set sizes and the dimension `d` (for fixed `C`),
- uses **only symmetric-key primitives** (no public-key crypto).

The construction reduces fuzzy matching to an **oblivious selected set
intersection (OSSI)** over per-axis membership tags, instantiated from a
shared-output OPPRF (so-OPPRF) and a secret-shared set intersection (S³I).

## Scope of this artifact

This is a **single-process benchmark prototype** measuring local compute time
plus a formula-based online communication transcript. The S³I OLE correlations
are produced by **actually running the Ring-LPN PCG backend** (`pcg/`); only the
**OT** correlations underlying the so-OPPRF (the AltMod-WPRF F4 mod-2 OT layer)
are modeled as an **ideal local tape**. There is no live network: the one-time
interactive PCG seed exchange, socket scheduling, serialization, and RTT latency
are not modeled.

What is **implemented and measured**:

- C-max collision verification (Algorithm 1) and the `cmax` of generated sets,
- membership encoding,
- so-OPPRF over RB-OKVS with the AltMod weak-PRF,
- PCG-backed secret-shared (r)OLE for S³I — the Ring-LPN PCG backend is
  **really executed and timed** (with tape refills on exhaustion),
- the S³I subproduct-tree / random-linear-combination / gcd nonempty test.

What is **abstracted or not yet included** (see *Limitations*):

- the **OT** layer of the so-OPPRF: the AltMod-WPRF F4 mod-2 OT correlations are
  a PRG-generated **ideal local tape**, standing in for real base OTs (the
  paper's §7.2 instantiates these with libOTe / the AltMod PRF from secure-join;
  this is the ~`256·dCN` 1-out-of-2 OTs in §E.2),
- the **live-network** layer: PCG runs in a single process, so the one-time
  interactive PCG seed exchange, socket scheduling, serialization, and RTT are
  not measured,
- the final `F_OT` retrieval of matched sender items (the benchmark stops at
  match bits).

Numbers from this artifact are meaningful for the **algorithmic online compute
and online-transcript** claims; they are not end-to-end networked protocol
timings.

## Repository layout

```text
gf64/    GF(2^64) field & polynomial arithmetic, batched Bernstein–Yang gcd,
         linear algebra kernels (AVX-512 / VPCLMUL)
pcg/     DMPF / Ring-LPN OLE PCG backend + additive-FFT evaluation domains
okvs/    RB-OKVS encoding / decoding
opprf/   AltMod weak-PRF + so-OPPRF over OKVS
fpsi/    Membership encoding, OSSI, S³I, and the integrated FPSI benchmark
```

Dependency direction: `gf64 → pcg`, `okvs → opprf → fpsi`, and `gf64, pcg → fpsi`.

### Code ↔ paper map

| Paper | Where |
| --- | --- |
| C-max collision property (Def. 1, §4); Alg. 1 verification; Table 1 / Fig. 13 | `fpsi::membership_cmax_collision`; `cmax` field of `bench_fpsi` |
| Membership encoding `M_j` (§5.1) | `fpsi/src/fpsi_membership.cpp`, `build_membership_pairs` |
| OSSI functionality `F_OSSI` (§5.2, §6) | `ossi_run_ideal`, batched OSSI in `fpsi/src/fpsi.cpp` |
| so-OPPRF = AltMod WPRF + RB-OKVS (§6.1) | `opprf/` (`altmod_prf`, `shared_oprf`) + `okvs/` |
| SSrOLE / rOLE from OLE via PCG (§6.1, App. A) | `pcg/` (`ole_pcg`, `ring`, `fft_additive`, `dmpf64`) |
| S³I: subproduct tree + RLC + gcd (§6.3; Kissner–Song, Yeo–Ying) | `s3i_intersect_shared[_nonempty]`; `gf64/` (`gf64_poly`, `gf64x4_gcd`) |
| Truncated 128→64-bit PRF packing (App. E.1) | `opprf/src/altmod_prf.cpp` |
| `F_OT` item retrieval (§5.3) | *not* in the released benchmark (match-bit only) |

## Requirements

- **CPU:** modern x86-64 with `AVX512F/BW/VL/DQ`, `VAES`, `VPCLMULQDQ`,
  `AES-NI`, `PCLMUL`. The default configure **fails** if these flags are
  unavailable.
- **Toolchain:** CMake ≥ 3.20, Ninja, and a C11 / C++20 compiler (GCC or Clang).

A portable fallback exists for debugging only — its numbers are **not**
comparable to the reference AVX-512 profile.

## Build

```bash
cmake --preset release
cmake --build --preset release -j
ctest --preset release --output-on-failure
```

Equivalent manual invocation:

```bash
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Debug-only portable build (no AVX-512, **not** for benchmarking):

```bash
cmake --preset release-portable
cmake --build --preset release-portable -j
```

## Reproducing the paper

All binaries land in `build/<module>/`. Reference experiments in the paper used
a single-threaded Intel Xeon Gold 6448H; `bench_fpsi` reports 40-bit statistical
security over `GF(2^64)`.

| Paper result | Command |
| --- | --- |
| Online latency & communication (Table 2) | `./build/fpsi/bench_fpsi <delta> <M> <N> <d> <C> <iters>` over the swept grid |
| Online breakdown (Tables 5, 6) | same, with `--verbose` |
| Offline breakdown (Table 4) | `--verbose`; `OLE Gen` corresponds to `bench_pcg` |
| C-max concrete analysis (Table 1, §7.1) | `cmax` output vs. the configured `C` |
| OLE correlation generation | `./build/pcg/bench_pcg` |
| OKVS encode/decode | `./build/okvs/bench_okvs_core` |

Representative single run (matches the paper's `M=N=2^16`, `d=16`, `C=4`,
`δ=30` cell):

```bash
./build/fpsi/bench_fpsi 30 65536 65536 16 4 1 --verbose
```

## Benchmark reference

### FPSI (main)

```bash
./build/fpsi/bench_fpsi [delta M N d C iters] [--verbose|-v]
```

| Arg | Meaning |
| --- | --- |
| `delta` | ℓ∞ matching radius δ |
| `M` | receiver set size |
| `N` | sender set size |
| `d` | item dimension |
| `C` | max membership-tag size for S³I (**must be a power of two**) |
| `iters` | repetitions |

Default output:

```text
FPSI benchmark (ideal OT/OLE tape)
params: delta=30 M=65536 N=65536 d=16 C=4 iters=1
correctness: forced_hits=4096 observed_hits=4096 cmax=4
offline_seconds=...
online_seconds=...
total_seconds=...
comm_r_to_s_MB=...   comm_s_to_r_MB=...   comm_total_MB=...
```

`--verbose` adds per-phase offline/online timings, per-`Y`-item latency, the
directional communication accounting, and operation counts.

### Component benchmarks

```bash
./build/gf64/bench_gf64_core [n_mul] [poly_reps] [poly_deg] [gcd_reps] [linalg_reps]
./build/pcg/bench_pcg        [logN] [compression] [weight] [reps] [batch]
./build/okvs/bench_okvs_core [n_items] [n_queries] [w] [epsilon]
./build/opprf/bench_opprf    [n_programmed] [n_queries] [reps]
```

Quick sanity sweep after building:

```bash
ctest --preset release --output-on-failure
./build/gf64/bench_gf64_core 262144 4096 16
./build/pcg/bench_pcg 20 4 3 3 8
./build/okvs/bench_okvs_core 65536 65536 218 0.10
./build/opprf/bench_opprf 4096 4096 3
./build/fpsi/bench_fpsi 30 4096 4096 16 4 1
```

Correctness expectations: `observed_hits == forced_hits`, and `cmax <= C`
(if `cmax > C` the generated workload exceeds the configured tag capacity and
the run should be discarded).

## Online vs. offline convention

- **offline** = input-independent local preprocessing: membership encoding,
  sender key generation, OSSI slot expansion, OPPRF PRF masking, RB-OKVS
  encoding, AltMod-WPRF tape generation, and PCG/OLE correlation generation.
- **online** = ideal-tape consumption and downstream compute: AltMod-WPRF tape
  materialization, receiver OKVS decode, and the S³I arithmetic + gcd nonempty
  test. PCG/OLE correlations are fully materialized *before* the S³I online
  timer starts, and `s3i_online_seconds` is measured directly (not derived by
  subtracting PCG time from an enclosing timer).

Party convention: `R` = FPSI receiver = OPPRF sender = S³I output receiver;
`S` = FPSI sender = OPPRF query holder = S³I sender.

`bench_fpsi` reports only the **directional online transcript** bytes, not the
expanded local preprocessing tapes. The accounting in `--verbose` follows:

```text
okvs_hint_r_to_s_bytes        = okvs_cols * 8
sooprf_key_mult_s_to_r_bytes  = key_mult_blocks * 2 * 16
sooprf_f4_choice_s_to_r_bytes = f4_blocks * 2 * 16
sooprf_f4_msg_r_to_s_bytes    = f4_blocks * 2 * 16
s3i_ole_correction_*_bytes    = scalar_ole_correlations * 8   (each direction)
s3i_hs_reveal_s_to_r_bytes    = N * 2 * (2*C) * 8
```

## Limitations

- `C` must be a power of two (the S³I subset tree follows the paper's
  power-of-two presentation).
- The field is `GF(2^64)`; failure probabilities that depend on `|F|` are at
  64-bit scale unless amplified.
- The benchmark uses the **weak** `F'_OSSI` nonempty predicate by default, not
  full intersection element recovery.
- Final item-retrieval OT is not included.
- The so-OPPRF OT correlations (AltMod-WPRF F4 mod-2 OT) are an ideal local
  tape. The S³I OLE is real PCG compute, but it runs single-process: live PCG
  seed exchange, socket scheduling, serialization, and RTT are out of scope.

## License & citation

License: *to be added at de-anonymization.*

```bibtex
@inproceedings{densityfpsi,
  title     = {Fuzzy Private Set Intersection under Density-Bounded Assumptions},
  author    = {Anonymous},
  booktitle = {IEEE Symposium on Security and Privacy (S\&P)},
  year      = {2027},
  note      = {Under submission}
}
```
