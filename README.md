# NewFPSI Chapter 6 Prototype

This repository is a cleaned benchmark prototype for the Chapter 6 FPSI design.
It targets the **ideal OT/OLE tape sanity-check model**:

- OT/OLE/PCG correlations are treated as preprocessed tapes.
- Offline time includes local preprocessing such as membership encoding, OKVS encoding, AltMod-WPRF tape generation, and PCG/OLE tape generation.
- Online time measures local tape consumption and downstream compute. S^3I PCG/OLE correlations are materialized before the S^3I online timer starts; the reported S^3I online time is measured directly, not derived by subtracting PCG time from an enclosing timer.
- Real socket/libOTe/coproto traffic is **not** measured yet. `bench_fpsi` reports formula-based **online network transcript** communication for the current ideal-tape protocol model. Expanded preprocessing tape/storage bytes are not reported as communication.

The default build assumes a modern x86-64 server with:

```text
AVX512F / AVX512BW / AVX512VL / VAES / VPCLMULQDQ / AES-NI / PCLMUL
```

Portable builds are kept only for debugging and should not be used as benchmark references. The S^3I nonempty test uses the AVX512/VPCLMUL 4-lane SIMD-batched gcd path by default, with scalar fallback for rare lower-degree cases.

---

## Directory layout

```text
gf64/   GF(2^64) arithmetic, polynomial arithmetic, gcd, and linear algebra kernels
pcg/    DMPF/OLE PCG backend and PCG throughput benchmark
okvs/   RB-OKVS encoding/decoding
opprf/  AltMod-WPRF ideal-tape OPPRF over OKVS
fpsi/   Membership encoding + weak F'_OSSI + S^3I FPSI benchmark
```

Dependency direction:

```text
gf64 -> pcg
okvs -> opprf -> fpsi
gf64 -> fpsi
pcg  -> fpsi
```

---

## Build

Reference build:

```bash
cmake --preset release
cmake --build --preset release -j
ctest --preset release --output-on-failure
```

Equivalent manual build:

```bash
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The default configure fails if AVX512+VAES+VPCLMUL flags are unavailable. For debug-only fallback:

```bash
cmake --preset release-portable
cmake --build --preset release-portable -j
ctest --preset release-portable --output-on-failure
```

Do not compare portable benchmark numbers against the reference AVX512 profile.

---

## Benchmarks

All paths below assume the default build directory `build/`.

### 1. GF64 core benchmark

```bash
./build/gf64/bench_gf64_core [n_mul] [poly_len] [reps]
```

Example:

```bash
./build/gf64/bench_gf64_core 262144 4096 16
```

Measures:

- GF(2^64) addition/xor throughput
- scalar multiplication throughput
- polynomial multiplication throughput
- AVX512 build: x4 gcd/coprime and linear algebra kernels

Use this to check the low-level arithmetic backend.

---

### 2. PCG throughput benchmark

```bash
./build/pcg/bench_pcg [logN] [compression] [weight] [reps] [batch]
```

Example:

```bash
./build/pcg/bench_pcg 20 4 3 3 8
```

Measures PCG key generation and expansion throughput using the forced BIG_STATE + eval-domain OLE backend.

Notes:

- `logN` controls the batch size `N = 2^logN`.
- `compression` and `weight` are the PCG sparse-polynomial parameters.
- The benchmark reports both serial same-process expansion throughput and the parallel-latency throughput estimate `N / max(party0, party1)`.
- This is an offline correlation-generation benchmark.
- FPSI/S^3I uses the generated OLE correlations as an ideal tape.

---

### 3. OKVS benchmark

```bash
./build/okvs/bench_okvs_core [n_items] [n_queries] [w] [epsilon]
```

Example:

```bash
./build/okvs/bench_okvs_core 65536 65536 218 0.10
```

Measures:

- OKVS encoding throughput
- OKVS decoding throughput

This benchmark measures the OKVS layer alone. It does not include OPPRF PRF masking unless explicitly performed by the OPPRF/FPSI benchmarks.

---

### 4. OPPRF benchmark

```bash
./build/opprf/bench_opprf [n_programmed] [n_queries] [reps]
```

Example:

```bash
./build/opprf/bench_opprf 4096 4096 3
```

Measures:

- OKVS encoding for programmed OPPRF points
- AltMod-WPRF ideal-tape generation
- AltMod-WPRF online tape consumption
- OKVS decoding for receiver queries

Model:

```text
AltMod-WPRF ideal tape + OKVS
```

This is not a real coproto/libOTe network benchmark. It is a local sanity check for the arithmetic path and cost split assuming preprocessed OT correlations.

---

### 5. FPSI benchmark

```bash
./build/fpsi/bench_fpsi [delta M N d C iters] [--verbose|-v]
```

Example:

```bash
./build/fpsi/bench_fpsi 30 4096 4096 16 4 1
```

Default output is intentionally short:

```text
FPSI benchmark (ideal OT/OLE tape)
params: delta=30 M=4096 N=4096 d=16 C=4 iters=1
correctness: forced_hits=256 observed_hits=256 cmax=4
offline_seconds=...
online_seconds=...
total_seconds=...
comm_r_to_s_MB=...
comm_s_to_r_MB=...
comm_total_MB=...
```

Meaning:

- `delta`: L_infinity matching radius.
- `M`: receiver set size.
- `N`: sender set size.
- `d`: item dimension.
- `C`: maximum membership tag size used by S^3I. Must currently be a power of two.
- `iters`: number of benchmark repetitions.
- `forced_hits`: synthetic matches inserted into the workload.
- `observed_hits`: matches recovered by the FPSI pipeline.
- `cmax`: measured C-max collision value of the generated receiver set.
- `offline_seconds`: local preprocessing time.
- `online_seconds`: ideal-tape online compute time.
- `total_seconds`: offline + online, excluding public setup/validation.
- `comm_r_to_s_MB`: online transcript size from FPSI receiver `R` to FPSI sender `S`, in decimal MB.
- `comm_s_to_r_MB`: online transcript size from FPSI sender `S` to FPSI receiver `R`, in decimal MB.
- `comm_total_MB`: total online transcript size, i.e., `R->S + S->R`, in decimal MB.

Per-item timing, raw network byte counts, and phase-level online communication formulas are printed only in verbose mode.

Verbose mode:

```bash
./build/fpsi/bench_fpsi 30 4096 4096 16 4 1 --verbose
```

Verbose output adds the detailed breakdown:

```text
[public setup / validation]
  public_setup_vandermonde_seconds
  cmax_verify_seconds

[offline/local-prep time: no protocol communication]
  membership_encoding_seconds
  sender_membership_keygen_seconds
  ossi_program_slot_expansion_seconds
  ossi_query_slot_expansion_seconds
  opprf_program_mask_prf_seconds
  okvs_linear_encode_seconds
  sooprf_altmod_wprf_correlation_tape_generation_seconds
  pcg_correlation_generation_seconds
  offline_local_prep_total_seconds

[online/local-hybrid compute time]
  sooprf_altmod_wprf_tape_online_seconds
  opprf_receiver_okvs_decode_seconds
  s3i_online_seconds
  online_local_hybrid_total_seconds

[network communication/accounting]
  network_comm_MB / network_comm_bytes
  online_comm_r_to_s_MB / online_comm_r_to_s_bytes
  online_comm_s_to_r_MB / online_comm_s_to_r_bytes
  okvs_hint_r_to_s_bytes
  sooprf_f4_msg_r_to_s_bytes
  s3i_ole_correction_r_to_s_bytes
  sooprf_key_mult_s_to_r_bytes
  sooprf_f4_choice_s_to_r_bytes
  s3i_ole_correction_s_to_r_bytes
  s3i_hs_reveal_s_to_r_bytes
  network_comm_per_Y_item_bytes
  formulas

[count/accounting]
  membership_pairs
  total_tag_values
  max_tag_size
  okvs_programmed_items
  okvs_query_items
  okvs_cols
  okvs_hint_table_bytes
  sooprf_packed_cols
  key_mult_blocks
  key_mult_lanes
  f4_blocks
  f4_lanes
  s3i_scalar_ole_correlations
  s3i_pcg_expanded_ole_correlations
  pcg_refills
  ssrole_polymul_calls
```

Model:

```text
membership encoding
+ secure-join-style AltMod-WPRF ideal tape / OKVS
+ PCG-backed SSrOLE ideal tape
+ weak F'_OSSI nonempty test
+ no OT retrieval of matched Y items
```

The FPSI benchmark currently outputs match bits. It does not include the final OT step that would retrieve matched sender items.

---

## Offline vs online convention

In this repository:

```text
offline = local preprocessing before online protocol execution
online  = local ideal-tape consumption and downstream compute
```

Offline includes:

- receiver membership encoding
- sender membership key generation
- OSSI slot expansion
- OPPRF programmed PRF masking
- OKVS linear encoding
- AltMod-WPRF correlation tape generation
- PCG/OLE correlation generation for S^3I

Online includes:

- AltMod-WPRF ideal-tape share materialization
- OPPRF receiver OKVS decode
- S^3I arithmetic and gcd/nonempty test using preprocessed OLE tape; PCG/OLE tape generation is completed before the measured S^3I online phase

Network communication accounting:

Party convention:

- `R` = FPSI receiver = OPPRF sender = S^3I output receiver.
- `S` = FPSI sender = OPPRF receiver/query holder = S^3I sender.

`bench_fpsi` intentionally does not report expanded ideal preprocessing tape material as communication. The local model still consumes preprocessed OPRF/SSrOLE correlations, but their expanded in-memory size is not real network traffic:

- AltMod-WPRF key-multiplication tape
- AltMod-WPRF F4 mod-2 OT tape
- S^3I OLE/PCG tape

`network_comm_bytes` / `online_comm_bytes` counts only the directional online transcript bytes implied by the local ideal-tape model:

- `R -> S`
  - OPPRF/OKVS hint table (`okvs_cols * 8`)
  - AltMod-WPRF F4 mod-2 sender masked messages
  - S^3I OLE correction openings from `R`
- `S -> R`
  - AltMod-WPRF key-multiplication derandomization messages
  - AltMod-WPRF F4 mod-2 receiver choice/correction messages
  - S^3I OLE correction openings from `S`
  - S^3I final `h_S` reveal for the weak nonempty test

The current byte formulas printed by `--verbose` are:

```text
okvs_hint_r_to_s_bytes = okvs_cols * 8
sooprf_key_mult_s_to_r_bytes = key_mult_blocks * 2 * 16
sooprf_f4_choice_s_to_r_bytes = f4_blocks * 2 * 16
sooprf_f4_msg_r_to_s_bytes = f4_blocks * 2 * 16
s3i_ole_correction_r_to_s_bytes = scalar_ole_correlations * 8
s3i_ole_correction_s_to_r_bytes = scalar_ole_correlations * 8
s3i_hs_reveal_s_to_r_bytes = N * 2 * (2*C) * 8
```

The preprocessing tape bytes are expanded local tape/accounting bytes, not measured network traffic for a concrete OT/PCG generator.

Not included yet:

- real OT/VOLE/PCG network protocol
- socket scheduling / serialization / latency
- final OT retrieval of matched `Y` items

So the current numbers are useful for sanity-checking the algorithmic pipeline and online communication formulas under ideal tapes, but they are not end-to-end network protocol timings.

---

## Recommended sanity checks

After building, run:

```bash
ctest --preset release --output-on-failure

./build/gf64/bench_gf64_core 262144 4096 16
./build/pcg/bench_pcg 20 4 3 3 8
./build/okvs/bench_okvs_core 65536 65536 218 0.10
./build/opprf/bench_opprf 4096 4096 3
./build/fpsi/bench_fpsi 30 4096 4096 16 4 1
./build/fpsi/bench_fpsi 30 4096 4096 16 4 1 --verbose
```

Correctness sanity for FPSI:

```text
observed_hits should equal forced_hits
cmax should be <= C
```

If `cmax > C`, the generated workload exceeds the configured S^3I tag capacity and the benchmark should be rejected.

---

## Current limitations

- `C` must be a power of two because the current S^3I subset tree assumes it.
- Field is currently GF(2^64), so statistical failure probabilities that depend on `|F|` are 64-bit scale unless repeated/amplified.
- The benchmark uses weak `F'_OSSI` nonempty testing, not full intersection recovery by default.
- Final item-retrieval OT is not included.
- Real coproto/libOTe communication is intentionally not part of this cleaned ideal-tape branch.

