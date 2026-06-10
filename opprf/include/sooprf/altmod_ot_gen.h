#pragma once

// Real silent-OT-backed generator for the AltMod-WPRF correlation tape.
//
// This replaces AltModWPrfCorrelationTape::make_ideal() (which fills the tape
// from a local PRG) with correlations produced by *actually running* libOTe's
// silent OT/VOLE in a single process. It exists to make the OFFLINE
// correlation-generation cost real and to back the paper's claim that the
// OT correlations are silently precomputed (paper Section 7.2 / E.2), in the
// same way the S^3I OLE side already runs the real Ring-LPN PCG backend.
//
// IMPORTANT — what this does NOT change:
//   * The online eval (SharedAltModOprf::eval_altmod_wprf_tape_into) is left
//     untouched. There, the four f4 strings are consumed as one-time pads and
//     cancel algebraically for every valid F3 value u in {0,1,2}; correctness
//     is independent of the pad VALUES (verified for u=0,1,2). So we only need
//     genuine, real-OT-derived randomness of the right SHAPE and COUNT, not a
//     bit-exact reproduction of secure-join's F4-VOLE compress.
//   * Online communication is still formula-estimated by the benchmark.
//
// Therefore this generator's purpose is: (1) measure the real silent-OT
// offline time, and (2) seed the tape from genuine OT correlations. If you
// later wire up the *networked* mod2OtF4 (derandomization + t1/t2 exchange),
// you must instead reproduce secure-join's exact compress
// (secure-join/CorGenerator/F4BitOtBatch.cpp: compressSender/compressRecver,
// mod2OtF4 in Prf/AltModWPrf.cpp). Until then, the pad-cancellation argument
// above keeps the in-process eval correct.

#include <cstddef>
#include <cstdint>
#include "sooprf/shared_oprf.h"

namespace sooprf {

// Cost model for the silent backend, returned alongside the tape so the
// benchmark can report a real (not formula) offline correlation-gen time.
struct SilentTapeCost {
    double silent_gen_seconds = 0.0;   // wall-clock for the silent OT/VOLE run
    std::size_t f4_correlations = 0;   // binary correlations consumed by mod2 F4 OT
    std::size_t key_correlations = 0;  // binary correlations consumed by key mult
    std::size_t silent_ot_instances = 0; // raw 1-out-of-2 silent OTs actually run
};

// Build the tape from real libOTe silent OT correlations.
//
// `input_count` is the number of so-OPRF query slots in the batch, identical
// to the argument of make_ideal(). `seed` only seeds the bookkeeping that
// shapes the tape; the correlation material itself comes from libOTe.
//
// On a build WITHOUT libOTe available (SOOPRF_WITH_LIBOTE undefined) this falls
// back to make_ideal() and reports silent_gen_seconds = 0, so the rest of the
// codebase still compiles and runs.
AltModWPrfCorrelationTape make_silent_ot(std::size_t input_count,
                                         std::uint64_t seed,
                                         SilentTapeCost* cost = nullptr);

} // namespace sooprf