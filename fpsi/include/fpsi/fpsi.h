#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "opprf/opprf.h"

namespace fpsi {

using opprf::Block128;

struct KeySetPair {
    Block128 key{};
    std::vector<std::uint64_t> values;
};

struct MembershipEncodingStats {
    std::size_t M = 0;
    std::size_t N = 0;
    std::size_t d = 0;
    std::uint64_t delta = 0;
    std::size_t programmed_pairs = 0;
    std::size_t total_tag_values = 0;
    std::size_t max_tag_size = 0;
};

struct OssiParams {
    // Upper bound on every selected set cardinality. The S3I subset tree currently
    // follows the paper's power-of-two presentation and requires C to be a power of two.
    std::size_t C = 8;

    opprf::Opprf64Params opprf_params{};
    std::uint64_t prf_seed = 0x465053495f4f5353ULL;
    std::uint64_t sooprf_tape_seed = 0x534f4f5052465f31ULL;
    std::uint64_t ssrole_seed = 0x5333495f5353524fULL;
    std::uint64_t gamma_seed = 0x47414d4d415f5231ULL;
    std::uint64_t root_seed = 0x524f4f545f46494eULL;
};

struct OssiStats {
    std::size_t programmed_pairs = 0;
    std::size_t query_pairs = 0;
    std::size_t okvs_cols = 0;
    std::size_t C = 0;
    std::size_t d = 0;
    std::size_t ssrole_polymul_calls = 0;
    // Paper accounting: two scalar OLEs per evaluation point for SSrOLE.
    std::size_t scalar_ole_equivalent = 0;
    // Concrete PCG-backed execution over additive FFT domains.  Because the
    // PCG backend expands powers-of-two evaluation domains, this can exceed
    // scalar_ole_equivalent when m+n+1 is not a power of two.
    std::size_t pcg_ole_correlations = 0;
    std::size_t pcg_refills = 0;
    // Wall-clock time spent generating/expanding PCG OLE correlations.
    // This is preprocessing/offline cost, not online protocol time.
    double pcg_offline_seconds = 0.0;
    // Public setup/precomputation time for Vandermonde/interpolation/fused extension matrices.
    // This depends only on public parameters (C and the fixed GF(2^64) domain).
    double vandermonde_precompute_seconds = 0.0;

    // Profiling-only timers for S^3I online path.
    double prof_ssrole_online_seconds = 0.0;
    double prof_vand_extend_seconds = 0.0;
    double prof_vand_shift_seconds = 0.0;
    double prof_vand_gamma_eval_seconds = 0.0;
    double prof_vand_interpolate_seconds = 0.0;
    double prof_subset_tree_seconds = 0.0;
    double prof_rlc_seconds = 0.0;
    double prof_gcd_seconds = 0.0;
    double prof_s3i_build_seconds = 0.0;
    // Direct wall-clock time for the S^3I online compute after all ideal OLE/PCG
    // correlations have been materialized. Benchmarks should prefer this over
    // subtracting pcg_offline_seconds from an enclosing timer.
    double prof_s3i_online_wall_seconds = 0.0;
};

struct OssiResult {
    std::vector<std::uint64_t> intersection;
    OssiStats stats;
};

struct OssiBatchBitResult {
    // matched[i] = 1 iff the i-th d-key OSSI query has non-empty intersection.
    // This is the weak F'_OSSI output used by the FPSI pipeline before OT retrieval.
    std::vector<std::uint8_t> matched;
    OssiStats stats;
    std::size_t N = 0;
};

// Domain-separation helper for turning an OSSI key k and slot ell in [1,C]
// into the fixed 128-bit key consumed by Opprf64. Both parties use this helper.
Block128 ossi_slot_key(Block128 key, std::size_t ell) noexcept;

// Domain-separated OPRF input key for half-packed so-OPRF slots.
// Pair index p is 1-based and corresponds to slots 2p-1 (.hi) and 2p (.lo).
Block128 ossi_pair_key(Block128 key, std::size_t pair_index) noexcept;

// Public setup: precompute all small Vandermonde/interpolation/fused extension
// matrices used by the C-way S^3I path.  This is independent of all inputs and
// OLE correlations, so benchmarks should report it separately from offline/online.
void s3i_precompute_vandermonde(std::size_t C, OssiStats* stats = nullptr);

// Chapter-6 S^3I: selected sets are given as additive shares over GF(2^64).
// receiver_share[j*C + ell] XOR sender_share[j*C + ell] is the ell-th element
// of the j-th noisy selected set.
std::vector<std::uint64_t> s3i_intersect_shared(std::span<const std::uint64_t> receiver_share,
                                                std::span<const std::uint64_t> sender_share,
                                                std::size_t d,
                                                std::size_t C,
                                                std::uint64_t ssrole_seed,
                                                std::uint64_t gamma_seed,
                                                std::uint64_t root_seed,
                                                OssiStats* stats = nullptr);

// Weak S^3I/F'_OSSI predicate: returns only whether the intersection is non-empty.
// It performs the same subset-tree and random-linear-combination computation as
// s3i_intersect_shared(), but skips root finding and only tests deg(gcd)>0.
bool s3i_intersect_shared_nonempty(std::span<const std::uint64_t> receiver_share,
                                    std::span<const std::uint64_t> sender_share,
                                    std::size_t d,
                                    std::size_t C,
                                    std::uint64_t ssrole_seed,
                                    std::uint64_t gamma_seed,
                                    OssiStats* stats = nullptr);

// Chapter-6 OSSI in the (ideal so-OPPRF, PCG-backed SSrOLE)-hybrid local model.
// This is executable/testable protocol code: it uses the existing Opprf64 OKVS
// path for the selection layer and the existing pcg/dmpf64 evaluation-domain
// OLE backend for S^3I polynomial multiplications.
OssiResult ossi_run_ideal(std::span<const KeySetPair> receiver_pairs,
                          std::span<const Block128> sender_keys,
                          const OssiParams& params = {});

// Batched Chapter-6 OSSI for the actual FPSI use case: N independent d-key
// query tuples.  The OPPRF/OKVS selection layer is executed once over all
// N*d*C slot queries, then the weak S^3I predicate is evaluated per tuple.
// sender_keys_flat is laid out as [query 0 d keys][query 1 d keys]...
OssiBatchBitResult ossi_run_ideal_batch_nonempty(std::span<const KeySetPair> receiver_pairs,
                                                 std::span<const Block128> sender_keys_flat,
                                                 std::size_t N,
                                                 std::size_t d,
                                                 const OssiParams& params = {});

// Return the exact number of scalar GF(2^64) OLE correlations consumed by
// the batched weak S^3I stage for the current subset-tree/RLC implementation.
std::size_t s3i_required_ole_correlations(std::size_t N, std::size_t d, std::size_t C);

// Run only the batched weak S^3I stage on already materialized OPPRF shares.
// receiver_share/sender_share are laid out as [query 0 d*C slots][query 1 d*C slots]...
OssiBatchBitResult s3i_batch_intersect_shared_nonempty(std::span<const std::uint64_t> receiver_share,
                                                       std::span<const std::uint64_t> sender_share,
                                                       std::size_t N,
                                                       std::size_t d,
                                                       const OssiParams& params = {});

// Domain-separation helper for the FPSI membership-encoding layer.
// It turns a universe point v on axis j into the OSSI base key (v || j).
Block128 membership_key(std::uint64_t v, std::size_t axis) noexcept;

// Locally verify the C-max collision parameter of X for the l_infty radius delta.
// X is laid out as M consecutive d-dimensional rows: X[i*d + j].
std::size_t membership_cmax_collision(std::span<const std::uint64_t> X_flat,
                                      std::size_t M,
                                      std::size_t d,
                                      std::uint64_t delta);

// Build the receiver's membership encoding { (v || j, M_j(v)) }.
// Each tag value is the 1-based receiver item index i+1.  If C_limit != 0,
// the function throws when any membership tag exceeds C_limit.
std::vector<KeySetPair> build_membership_pairs(std::span<const std::uint64_t> X_flat,
                                               std::size_t M,
                                               std::size_t d,
                                               std::uint64_t delta,
                                               std::size_t C_limit = 0,
                                               MembershipEncodingStats* stats = nullptr);

// Build the sender's flattened OSSI query keys y_{i,j} || j from Y.
// Y is laid out as N consecutive d-dimensional rows: Y[i*d + j].
std::vector<Block128> build_sender_membership_keys(std::span<const std::uint64_t> Y_flat,
                                                   std::size_t N,
                                                   std::size_t d);

// Executable FPSI membership-encoding front end in the current local hybrid model.
// This returns only the weak F'_OSSI match bits; OT retrieval of matched Y items
// remains outside this local benchmark path.
OssiBatchBitResult fpsi_match_ideal_batch_nonempty(std::span<const std::uint64_t> X_flat,
                                                   std::size_t M,
                                                   std::span<const std::uint64_t> Y_flat,
                                                   std::size_t N,
                                                   std::size_t d,
                                                   std::uint64_t delta,
                                                   const OssiParams& params = {},
                                                   MembershipEncodingStats* membership_stats = nullptr);

} // namespace fpsi
