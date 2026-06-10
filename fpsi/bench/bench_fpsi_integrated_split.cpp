#include "fpsi/fpsi.h"
#include "sooprf/prng.h"
#include "sooprf/shared_oprf.h"
#include "sooprf/altmod_ot_gen.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using clock_type = std::chrono::steady_clock;

static double sec_since(clock_type::time_point a, clock_type::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

static bool is_power_of_two(std::size_t x) noexcept {
    return x && ((x & (x - 1)) == 0);
}

static std::uint64_t group_base(std::size_t group, std::size_t axis, std::uint64_t delta) {
    // Keep different groups and axes far outside each other's delta windows.
    const std::uint64_t stride = 4 * std::max<std::uint64_t>(delta, 1) + 1024;
    return 0x100000ULL + static_cast<std::uint64_t>(group) * stride +
           static_cast<std::uint64_t>(axis) * 0x100000000ULL;
}

static void make_true_membership_workload(std::vector<std::uint64_t>& X,
                                          std::vector<std::uint64_t>& Y,
                                          std::size_t M,
                                          std::size_t N,
                                          std::size_t d,
                                          std::uint64_t delta,
                                          std::size_t C,
                                          std::size_t& forced_hits_out) {
    if (M == 0 || N == 0 || d == 0 || C == 0) {
        throw std::invalid_argument("M,N,d,C must be nonzero");
    }

    const std::size_t groups = (M + C - 1) / C;
    X.assign(M * d, 0);
    for (std::size_t i = 0; i < M; ++i) {
        const std::size_t g = i / C;
        for (std::size_t j = 0; j < d; ++j) {
            X[i * d + j] = group_base(g, j, delta);
        }
    }

    sooprf::SplitMix64 rng(0x6d656d6265725f31ULL);
    Y.assign(N * d, 0);
    forced_hits_out = 0;
    for (std::size_t n = 0; n < N; ++n) {
        const std::size_t g = static_cast<std::size_t>(rng.next_u64() % groups);
        for (std::size_t j = 0; j < d; ++j) {
            Y[n * d + j] = group_base(g, j, delta);
        }

        if ((n % 16) == 0) {
            ++forced_hits_out;
        } else {
            // Break one coordinate just outside the delta window.
            const std::size_t bad_axis = static_cast<std::size_t>(rng.next_u64() % d);
            Y[n * d + bad_axis] += delta + 1;
        }
    }
}

static std::size_t ossi_pair_index_from_slot(std::size_t ell) noexcept {
    return (ell + 1) / 2;
}

static std::uint8_t ossi_slot_uses_hi(std::size_t ell) noexcept {
    // Half-packing convention: slot 2p-1 uses .hi, slot 2p uses .lo.
    return static_cast<std::uint8_t>((ell & 1U) != 0U);
}

static void expand_ossi_program(std::span<const fpsi::KeySetPair> receiver_pairs,
                                std::size_t C,
                                std::vector<fpsi::Block128>& programmed_keys,
                                std::vector<fpsi::Block128>& programmed_pair_keys,
                                std::vector<std::uint8_t>& programmed_slot_hi,
                                std::vector<std::uint64_t>& programmed_values) {
    std::size_t reserve = 0;
    for (const auto& kv : receiver_pairs) {
        if (kv.values.size() > C) throw std::invalid_argument("receiver set exceeds C");
        reserve += kv.values.size();
    }
    programmed_keys.clear();
    programmed_pair_keys.clear();
    programmed_slot_hi.clear();
    programmed_values.clear();
    programmed_keys.reserve(reserve);
    programmed_pair_keys.reserve(reserve);
    programmed_slot_hi.reserve(reserve);
    programmed_values.reserve(reserve);
    for (const auto& kv : receiver_pairs) {
        for (std::size_t ell = 1; ell <= kv.values.size(); ++ell) {
            programmed_keys.push_back(fpsi::ossi_slot_key(kv.key, ell));
            programmed_pair_keys.push_back(fpsi::ossi_pair_key(kv.key, ossi_pair_index_from_slot(ell)));
            programmed_slot_hi.push_back(ossi_slot_uses_hi(ell));
            programmed_values.push_back(kv.values[ell - 1]);
        }
    }
}

static void expand_ossi_queries(std::span<const fpsi::Block128> sender_keys_flat,
                                std::size_t N,
                                std::size_t d,
                                std::size_t C,
                                std::vector<fpsi::Block128>& query_keys,
                                std::vector<fpsi::Block128>& query_pair_keys,
                                std::vector<std::size_t>& query_pair_index,
                                std::vector<std::uint8_t>& query_slot_hi) {
    if (sender_keys_flat.size() != N * d) throw std::invalid_argument("sender key size mismatch");
    query_keys.clear();
    query_pair_keys.clear();
    query_pair_index.clear();
    query_slot_hi.clear();
    query_keys.reserve(N * d * C);
    query_pair_keys.reserve(N * d * (C / 2));
    query_pair_index.reserve(N * d * C);
    query_slot_hi.reserve(N * d * C);
    for (std::size_t n = 0; n < N; ++n) {
        const std::size_t base = n * d;
        for (std::size_t j = 0; j < d; ++j) {
            const auto& k = sender_keys_flat[base + j];
            for (std::size_t pidx = 1; pidx <= C / 2; ++pidx) {
                const std::size_t out_pair_idx = query_pair_keys.size();
                query_pair_keys.push_back(fpsi::ossi_pair_key(k, pidx));
                for (std::size_t off = 0; off < 2; ++off) {
                    const std::size_t ell = 2 * pidx - 1 + off;
                    query_keys.push_back(fpsi::ossi_slot_key(k, ell));
                    query_pair_index.push_back(out_pair_idx);
                    query_slot_hi.push_back(ossi_slot_uses_hi(ell));
                }
            }
        }
    }
}

struct Timings {
    double public_setup_s = 0.0;
    double cmax_verify_s = 0.0;
    double membership_encoding_s = 0.0;
    double sender_membership_keys_s = 0.0;
    double ossi_program_expansion_s = 0.0;
    double ossi_query_expansion_s = 0.0;
    double opprf_mask_prf_s = 0.0;
    double okvs_encode_s = 0.0;
    double sooprf_ot_tape_generation_s = 0.0;
    double sooprf_ot_tape_share_s = 0.0;
    double opprf_receiver_decode_s = 0.0;
    double s3i_total_s = 0.0;
    double s3i_pcg_offline_s = 0.0;
    double s3i_online_s = 0.0;
};

struct CommAccounting {
    static constexpr std::size_t kBlockBytes = 16;
    static constexpr std::size_t kFieldBytes = sizeof(std::uint64_t);

    // Party convention for directional accounting:
    //   R = FPSI receiver = OPPRF sender = S^3I output receiver.
    //   S = FPSI sender   = OPPRF receiver/query holder = S^3I sender.
    // These are formula/accounting bytes under ideal preprocessed tapes, not socket traces.

    // Main online protocol sizes.
    std::size_t okvs_hint_bytes = 0;                    // R -> S

    // AltMod soOPRF online transcript after ideal OT correlations exist.
    std::size_t sooprf_key_mult_s_to_r_bytes = 0;       // S(input holder) -> R(key holder)
    std::size_t sooprf_f4_choice_s_to_r_bytes = 0;      // S(WPRF receiver) -> R(WPRF sender)
    std::size_t sooprf_f4_msg_r_to_s_bytes = 0;         // R(WPRF sender) -> S(WPRF receiver)

    // S^3I online transcript after ideal OLE correlations exist.
    std::size_t s3i_ole_correction_r_to_s_bytes = 0;    // one GF(2^64) correction per OLE
    std::size_t s3i_ole_correction_s_to_r_bytes = 0;    // one GF(2^64) correction per OLE
    std::size_t s3i_hs_reveal_s_to_r_bytes = 0;         // final h_S,1 and h_S,2 reveal

    // Ideal preprocessed tape accounting.  These are not socket-measured bytes;
    // they are the expanded correlation material consumed by the local ideal-tape model.
    std::size_t sooprf_key_mult_tape_bytes = 0;
    std::size_t sooprf_f4_tape_bytes = 0;
    std::size_t s3i_ole_tape_bytes = 0;

    // Count-level accounting.
    std::size_t okvs_cols = 0;
    std::size_t okvs_programmed_items = 0;
    std::size_t okvs_query_items = 0;
    std::size_t sooprf_query_pairs = 0;

    std::size_t sooprf_packed_cols = 0;
    std::size_t sooprf_key_mult_blocks = 0;
    std::size_t sooprf_key_mult_lanes = 0;
    std::size_t sooprf_f4_blocks = 0;
    std::size_t sooprf_f4_lanes = 0;

    std::size_t s3i_scalar_ole_correlations = 0;
    std::size_t s3i_pcg_expanded_ole_correlations = 0;

    std::size_t preprocessed_tape_bytes() const noexcept {
        return sooprf_key_mult_tape_bytes + sooprf_f4_tape_bytes + s3i_ole_tape_bytes;
    }
    std::size_t sooprf_online_r_to_s_bytes() const noexcept {
        return sooprf_f4_msg_r_to_s_bytes;
    }
    std::size_t sooprf_online_s_to_r_bytes() const noexcept {
        return sooprf_key_mult_s_to_r_bytes + sooprf_f4_choice_s_to_r_bytes;
    }
    std::size_t sooprf_online_bytes() const noexcept {
        return sooprf_online_r_to_s_bytes() + sooprf_online_s_to_r_bytes();
    }
    std::size_t s3i_online_r_to_s_bytes() const noexcept {
        return s3i_ole_correction_r_to_s_bytes;
    }
    std::size_t s3i_online_s_to_r_bytes() const noexcept {
        return s3i_ole_correction_s_to_r_bytes + s3i_hs_reveal_s_to_r_bytes;
    }
    std::size_t s3i_online_bytes() const noexcept {
        return s3i_online_r_to_s_bytes() + s3i_online_s_to_r_bytes();
    }
    std::size_t online_r_to_s_bytes() const noexcept {
        // OKVS hint/table is sent by R during the online OPPRF phase.
        return okvs_hint_bytes + sooprf_online_r_to_s_bytes() + s3i_online_r_to_s_bytes();
    }
    std::size_t online_s_to_r_bytes() const noexcept {
        return sooprf_online_s_to_r_bytes() + s3i_online_s_to_r_bytes();
    }
    std::size_t online_bytes() const noexcept {
        return online_r_to_s_bytes() + online_s_to_r_bytes();
    }
    std::size_t total_accounted_bytes() const noexcept {
        return preprocessed_tape_bytes() + online_bytes();
    }
};

static CommAccounting compute_comm_accounting(std::size_t N,
                                              std::size_t C,
                                              std::size_t programmed_pairs,
                                              std::size_t query_slots,
                                              std::size_t query_pairs,
                                              std::size_t okvs_cols,
                                              const fpsi::OssiStats& s3i_stats) {
    CommAccounting c;
    c.okvs_cols = okvs_cols;
    c.okvs_programmed_items = programmed_pairs;
    c.okvs_query_items = query_slots;
    c.sooprf_query_pairs = query_pairs;

    c.sooprf_packed_cols = (query_pairs + 127) / 128;
    c.sooprf_key_mult_blocks = sooprf::AltModPrfParams::KeyBits * c.sooprf_packed_cols;
    c.sooprf_key_mult_lanes = c.sooprf_key_mult_blocks * 128;
    c.sooprf_f4_blocks = sooprf::AltModPrfParams::MidBits * c.sooprf_packed_cols;
    c.sooprf_f4_lanes = c.sooprf_f4_blocks * 128;

    // OKVS hint/table: the sender needs the encoded OKVS table to answer its
    // query points, so this belongs to online OPPRF communication, not offline.
    // Formula: okvs_cols GF(2^64) elements.
    c.okvs_hint_bytes = okvs_cols * CommAccounting::kFieldBytes;

    // AltMod soOPRF online transcript under preprocessed OT tape.
    // Key multiplication: two 128-bit F3-encoding planes per key-row/packed-column block.
    c.sooprf_key_mult_s_to_r_bytes = c.sooprf_key_mult_blocks * 2 * CommAccounting::kBlockBytes;
    // F4 mod-2 conversion: receiver sends two 128-bit correction/choice blocks and
    // sender sends two 128-bit masked-message blocks per middle-row/packed-column block.
    c.sooprf_f4_choice_s_to_r_bytes = c.sooprf_f4_blocks * 2 * CommAccounting::kBlockBytes;
    c.sooprf_f4_msg_r_to_s_bytes = c.sooprf_f4_blocks * 2 * CommAccounting::kBlockBytes;

    // S^3I SSrOLE online transcript.
    // Each scalar OLE correlation opens one GF(2^64) correction from each side.
    c.s3i_scalar_ole_correlations = s3i_stats.scalar_ole_equivalent;
    c.s3i_pcg_expanded_ole_correlations = s3i_stats.pcg_ole_correlations;
    c.s3i_ole_correction_r_to_s_bytes = c.s3i_scalar_ole_correlations * CommAccounting::kFieldBytes;
    c.s3i_ole_correction_s_to_r_bytes = c.s3i_scalar_ole_correlations * CommAccounting::kFieldBytes;
    // Final weak F'_OSSI reveal: S sends h_S,1 and h_S,2.  The current representation
    // has 2C GF(2^64) words per polynomial per FPSI query.
    c.s3i_hs_reveal_s_to_r_bytes = N * 2 * (2 * C) * CommAccounting::kFieldBytes;

    // Expanded ideal preprocessing tape/accounting.
    // These are reported separately as the local ideal model's consumed correlation material.
    c.sooprf_key_mult_tape_bytes = c.sooprf_key_mult_blocks * 2 * CommAccounting::kBlockBytes;
    c.sooprf_f4_tape_bytes = c.sooprf_f4_blocks * 4 * CommAccounting::kBlockBytes;
    // PCG/OLE tape in this implementation stores enough material to locally realize an OLE
    // share pair; use the actually expanded PCG count, which can exceed scalar_ole_equivalent.
    c.s3i_ole_tape_bytes = c.s3i_pcg_expanded_ole_correlations * 4 * CommAccounting::kFieldBytes;
    return c;
}

static double mb(std::size_t bytes) noexcept {
    return static_cast<double>(bytes) / 1000000.0;
}

static double mib(std::size_t bytes) noexcept {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

static void print_usage(const char* prog) {
    std::cerr << "usage: " << prog << " [delta M N d C iters] [--verbose|-v]\n"
              << "  default output: offline/online latency summary only\n"
              << "  --verbose: print phase breakdown and size/accounting details\n";
}

int main(int argc, char** argv) {
    // Positional args: delta M N d C iters. Options may appear anywhere.
    bool verbose = false;
    std::vector<std::string> positional;
    positional.reserve(6);
    for (int ai = 1; ai < argc; ++ai) {
        const std::string arg(argv[ai]);
        if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 2;
        } else {
            positional.push_back(arg);
        }
    }
    if (positional.size() > 6) {
        std::cerr << "too many positional arguments\n";
        print_usage(argv[0]);
        return 2;
    }

    std::uint64_t delta = positional.size() > 0 ? std::strtoull(positional[0].c_str(), nullptr, 10) : 4;
    std::size_t M       = positional.size() > 1 ? std::strtoull(positional[1].c_str(), nullptr, 10) : 4096;
    std::size_t N       = positional.size() > 2 ? std::strtoull(positional[2].c_str(), nullptr, 10) : 1024;
    std::size_t d       = positional.size() > 3 ? std::strtoull(positional[3].c_str(), nullptr, 10) : 8;
    std::size_t C       = positional.size() > 4 ? std::strtoull(positional[4].c_str(), nullptr, 10) : 8;
    std::size_t iters   = positional.size() > 5 ? std::strtoull(positional[5].c_str(), nullptr, 10) : 3;
    // AltMod WPRF tape path has fixed accounting: reusable key OTs + F4-bit OTs.

    if (!is_power_of_two(C)) {
        std::cerr << "C must be a power of two for the current S3I subset tree\n";
        return 2;
    }
    if ((C & 1U) != 0U) {
        std::cerr << "C must be even for half-packed so-OPRF\n";
        return 2;
    }
    if (d < 2 || M == 0 || N == 0 || C == 0) {
        std::cerr << "M,N,C must be nonzero and d must be at least 2\n";
        return 2;
    }

    std::vector<std::uint64_t> X;
    std::vector<std::uint64_t> Y;
    std::size_t forced_hits = 0;
    make_true_membership_workload(X, Y, M, N, d, delta, C, forced_hits);

    fpsi::OssiParams params;
    params.C = C;
    params.opprf_params.epsilon = 0.08;
    params.opprf_params.okvs_seed[0] = 17;

    fpsi::OssiStats setup_stats;
    const auto setup0 = clock_type::now();
    fpsi::s3i_precompute_vandermonde(C, &setup_stats);
    const auto setup1 = clock_type::now();

    const auto verify0 = clock_type::now();
    const auto cmax = fpsi::membership_cmax_collision(X, M, d, delta);
    const auto verify1 = clock_type::now();

    Timings tavg;
    tavg.public_setup_s = sec_since(setup0, setup1);
    tavg.cmax_verify_s = sec_since(verify0, verify1);

    fpsi::MembershipEncodingStats last_mem{};
    fpsi::OssiStats last_s3i{};
    std::size_t observed_hits = 0;
    std::size_t okvs_cols = 0;
    std::size_t programmed_pairs = 0;
    std::size_t query_slots = 0;
    std::size_t query_pairs = 0;
    CommAccounting comm{};
    sooprf::SilentTapeCost ot_cost{};

    for (std::size_t it = 0; it < iters; ++it) {
        fpsi::OssiParams p = params;
        p.sooprf_tape_seed += 0x100000ULL + it;
        p.ssrole_seed += 0x200000ULL + it;
        p.gamma_seed += 0x300000ULL + it;
        p.root_seed += 0x400000ULL + it;
        p.prf_seed += 0x500000ULL + it;

        fpsi::MembershipEncodingStats mem_stats;
        const auto m0 = clock_type::now();
        auto receiver_pairs = fpsi::build_membership_pairs(X, M, d, delta, p.C, &mem_stats);
        const auto m1 = clock_type::now();
        auto sender_keys = fpsi::build_sender_membership_keys(Y, N, d);
        const auto m2 = clock_type::now();

        std::vector<fpsi::Block128> programmed_keys;
        std::vector<fpsi::Block128> programmed_pair_keys;
        std::vector<std::uint8_t> programmed_slot_hi;
        std::vector<std::uint64_t> programmed_values;
        std::vector<fpsi::Block128> query_keys;
        std::vector<fpsi::Block128> query_pair_keys;
        std::vector<std::size_t> query_pair_index;
        std::vector<std::uint8_t> query_slot_hi;

        const auto e0 = clock_type::now();
        expand_ossi_program(receiver_pairs, p.C, programmed_keys, programmed_pair_keys,
                            programmed_slot_hi, programmed_values);
        const auto e1 = clock_type::now();
        expand_ossi_queries(sender_keys, N, d, p.C, query_keys, query_pair_keys,
                            query_pair_index, query_slot_hi);
        const auto e2 = clock_type::now();

        const auto prf_key = sooprf::AltModPrf::random_key(p.prf_seed);
        opprf::Opprf64 op(prf_key, p.opprf_params);

        opprf::OpprfEncodeStats enc_stats;
        const auto enc0 = clock_type::now();
        auto enc = op.encode_packed(programmed_keys, programmed_pair_keys,
                                    programmed_slot_hi, programmed_values, &enc_stats);
        const auto enc1 = clock_type::now();
        (void)enc0;
        (void)enc1;

        const auto ot0 = clock_type::now();
        // Real libOTe silent-OT correlation tape when built with
        // -DSOOPRF_WITH_LIBOTE=ON; PRG fallback otherwise. The ot0/ot1 window
        // now measures the real silent-OT generation time.
        auto tape = sooprf::make_silent_ot(
            query_pair_keys.size(),
            p.sooprf_tape_seed ^ static_cast<std::uint64_t>(query_pair_keys.size()),
            &ot_cost);
        const auto ot1 = clock_type::now();

        sooprf::SharedAltModOprf soprf{sooprf::AltModPrf(prf_key)};
        const auto soprf0 = clock_type::now();
        auto shares = soprf.eval_altmod_wprf_tape(query_pair_keys, tape);
        const auto soprf1 = clock_type::now();
        if (!tape.consumed) {
            throw std::runtime_error("AltMod WPRF tape was not consumed");
        }

        std::vector<std::uint64_t> sender_share(query_keys.size());
        for (std::size_t i = 0; i < query_keys.size(); ++i) {
            const std::size_t pidx = query_pair_index[i];
            const auto b = shares.sender_share[pidx];
            sender_share[i] = query_slot_hi[i] ? b.hi : b.lo;
        }
        std::vector<std::uint64_t> receiver_share(query_keys.size());
        const auto dec0 = clock_type::now();
        op.receiver_share_into_packed(enc, query_keys, shares.receiver_share,
                                      query_pair_index, query_slot_hi, receiver_share);
        const auto dec1 = clock_type::now();

        const auto s3i0 = clock_type::now();
        auto res = fpsi::s3i_batch_intersect_shared_nonempty(receiver_share, sender_share, N, d, p);
        const auto s3i1 = clock_type::now();

        const double s3i_total = sec_since(s3i0, s3i1);
        const double s3i_pcg = res.stats.pcg_offline_seconds;
        // Direct S^3I online wall clock measured inside the S^3I routine after
        // all ideal OLE/PCG correlations have been materialized. Do not derive
        // this by subtracting preprocessing from an enclosing timer.
        const double s3i_online = res.stats.prof_s3i_online_wall_seconds;

        tavg.membership_encoding_s += sec_since(m0, m1);
        tavg.sender_membership_keys_s += sec_since(m1, m2);
        tavg.ossi_program_expansion_s += sec_since(e0, e1);
        tavg.ossi_query_expansion_s += sec_since(e1, e2);
        tavg.opprf_mask_prf_s += enc_stats.mask_prf_seconds;
        tavg.okvs_encode_s += enc_stats.okvs_encode_seconds;
        tavg.sooprf_ot_tape_generation_s += sec_since(ot0, ot1);
        tavg.sooprf_ot_tape_share_s += sec_since(soprf0, soprf1);
        tavg.opprf_receiver_decode_s += sec_since(dec0, dec1);
        tavg.s3i_total_s += s3i_total;
        tavg.s3i_pcg_offline_s += s3i_pcg;
        tavg.s3i_online_s += s3i_online;

        mem_stats.N = N;
        last_mem = mem_stats;
        last_s3i = res.stats;
        observed_hits = std::accumulate(res.matched.begin(), res.matched.end(), std::size_t{0});
        okvs_cols = enc.cols;
        programmed_pairs = programmed_keys.size();
        query_slots = query_keys.size();
        query_pairs = query_pair_keys.size();
    }

    const double denom = static_cast<double>(iters);
    tavg.membership_encoding_s /= denom;
    tavg.sender_membership_keys_s /= denom;
    tavg.ossi_program_expansion_s /= denom;
    tavg.ossi_query_expansion_s /= denom;
    tavg.opprf_mask_prf_s /= denom;
    tavg.okvs_encode_s /= denom;
    tavg.sooprf_ot_tape_generation_s /= denom;
    tavg.sooprf_ot_tape_share_s /= denom;
    tavg.opprf_receiver_decode_s /= denom;
    tavg.s3i_total_s /= denom;
    tavg.s3i_pcg_offline_s /= denom;
    tavg.s3i_online_s /= denom;

    const double offline_local_prep = tavg.membership_encoding_s + tavg.sender_membership_keys_s +
                                      tavg.ossi_program_expansion_s + tavg.ossi_query_expansion_s +
                                      tavg.opprf_mask_prf_s + tavg.okvs_encode_s +
                                      tavg.sooprf_ot_tape_generation_s +
                                      tavg.s3i_pcg_offline_s;
    const double online_local_hybrid = tavg.sooprf_ot_tape_share_s + tavg.opprf_receiver_decode_s +
                                       tavg.s3i_online_s;
    const double end_to_end_excluding_public_setup = offline_local_prep + online_local_hybrid;
    comm = compute_comm_accounting(N, C, programmed_pairs, query_slots, query_pairs, okvs_cols, last_s3i);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "FPSI benchmark (ideal OT/OLE tape)\n";
    std::cout << "params: delta=" << delta << " M=" << M << " N=" << N << " d=" << d
              << " C=" << C << " iters=" << iters << "\n";
    std::cout << "correctness: forced_hits=" << forced_hits
              << " observed_hits=" << observed_hits
              << " cmax=" << cmax << "\n";
    std::cout << "offline_seconds=" << offline_local_prep << "\n";
    std::cout << "online_seconds=" << online_local_hybrid << "\n";
    std::cout << "total_seconds=" << end_to_end_excluding_public_setup << "\n";
    std::cout << "comm_r_to_s_MB=" << mb(comm.online_r_to_s_bytes()) << "\n";
    std::cout << "comm_s_to_r_MB=" << mb(comm.online_s_to_r_bytes()) << "\n";
    std::cout << "comm_total_MB=" << mb(comm.online_bytes()) << "\n";

    if (verbose) {
        std::cout << "\n[model]\n";
        std::cout << "  membership + secure-join-style-AltMod-WPRF-tape/OKVS + "
                     "PCG-backed-SSrOLE + weak-F'_OSSI/no-OT-retrieval\n";
        std::cout << "  communication is formula/accounting for the online network transcript only; "
                     "socket/libOTe/coproto traffic is not measured here\n";

        std::cout << "\n[public setup / validation]\n";
        std::cout << "  public_setup_vandermonde_seconds=" << tavg.public_setup_s << "\n";
        std::cout << "  cmax_verify_seconds=" << tavg.cmax_verify_s << "\n";

        std::cout << "\n[offline/local-prep time: no protocol communication]\n";
        std::cout << "  membership_encoding_seconds=" << tavg.membership_encoding_s << "\n";
        std::cout << "  sender_membership_keygen_seconds=" << tavg.sender_membership_keys_s << "\n";
        std::cout << "  ossi_program_slot_expansion_seconds=" << tavg.ossi_program_expansion_s << "\n";
        std::cout << "  ossi_query_slot_expansion_seconds=" << tavg.ossi_query_expansion_s << "\n";
        std::cout << "  opprf_program_mask_prf_seconds=" << tavg.opprf_mask_prf_s << "\n";
        std::cout << "  okvs_linear_encode_seconds=" << tavg.okvs_encode_s << "\n";
        std::cout << "  sooprf_altmod_wprf_correlation_tape_generation_seconds=" << tavg.sooprf_ot_tape_generation_s << "\n";
        std::cout << "  pcg_correlation_generation_seconds=" << tavg.s3i_pcg_offline_s << "\n";
        std::cout << "  offline_local_prep_total_seconds=" << offline_local_prep << "\n";

        std::cout << "\n[online/local-hybrid compute time]\n";
        std::cout << "  sooprf_altmod_wprf_tape_online_seconds=" << tavg.sooprf_ot_tape_share_s << "\n";
        std::cout << "  opprf_receiver_okvs_decode_seconds=" << tavg.opprf_receiver_decode_s << "\n";
        std::cout << "  s3i_online_seconds=" << tavg.s3i_online_s << "\n";
        std::cout << "  online_local_hybrid_total_seconds=" << online_local_hybrid << "\n";
        std::cout << "  online_per_Y_item_us=" << (online_local_hybrid * 1e6 / static_cast<double>(N)) << "\n";
        std::cout << "  total_per_Y_item_us=" << (end_to_end_excluding_public_setup * 1e6 / static_cast<double>(N)) << "\n";

        std::cout << "\n[network communication/accounting]\n";
        std::cout << "  network_comm_MB=" << mb(comm.online_bytes())
                  << " network_comm_bytes=" << comm.online_bytes()
                  << " network_comm_MiB=" << mib(comm.online_bytes()) << "\n";
        std::cout << "    online_comm_r_to_s_MB=" << mb(comm.online_r_to_s_bytes())
                  << " online_comm_r_to_s_bytes=" << comm.online_r_to_s_bytes()
                  << " online_comm_r_to_s_MiB=" << mib(comm.online_r_to_s_bytes()) << "\n";
        std::cout << "      okvs_hint_r_to_s_bytes=" << comm.okvs_hint_bytes
                  << " sooprf_f4_msg_r_to_s_bytes=" << comm.sooprf_f4_msg_r_to_s_bytes
                  << " s3i_ole_correction_r_to_s_bytes=" << comm.s3i_ole_correction_r_to_s_bytes << "\n";
        std::cout << "    online_comm_s_to_r_MB=" << mb(comm.online_s_to_r_bytes())
                  << " online_comm_s_to_r_bytes=" << comm.online_s_to_r_bytes()
                  << " online_comm_s_to_r_MiB=" << mib(comm.online_s_to_r_bytes()) << "\n";
        std::cout << "      sooprf_key_mult_s_to_r_bytes=" << comm.sooprf_key_mult_s_to_r_bytes
                  << " sooprf_f4_choice_s_to_r_bytes=" << comm.sooprf_f4_choice_s_to_r_bytes
                  << " s3i_ole_correction_s_to_r_bytes=" << comm.s3i_ole_correction_s_to_r_bytes
                  << " s3i_hs_reveal_s_to_r_bytes=" << comm.s3i_hs_reveal_s_to_r_bytes << "\n";
        std::cout << "    sooprf_online_bytes=" << comm.sooprf_online_bytes()
                  << " sooprf_online_r_to_s_bytes=" << comm.sooprf_online_r_to_s_bytes()
                  << " sooprf_online_s_to_r_bytes=" << comm.sooprf_online_s_to_r_bytes() << "\n";
        std::cout << "    s3i_online_bytes=" << comm.s3i_online_bytes()
                  << " s3i_online_r_to_s_bytes=" << comm.s3i_online_r_to_s_bytes()
                  << " s3i_online_s_to_r_bytes=" << comm.s3i_online_s_to_r_bytes() << "\n";
        std::cout << "  network_comm_per_Y_item_bytes="
                  << (static_cast<double>(comm.online_bytes()) / static_cast<double>(N)) << "\n";
        std::cout << "  party convention: R=FPSI receiver/OPPRF sender/S3I receiver, "
                     "S=FPSI sender/OPPRF receiver/S3I sender\n";
        std::cout << "  formulas:\n";
        std::cout << "    okvs_hint_r_to_s_bytes = okvs_cols * 8\n";
        std::cout << "    sooprf_key_mult_s_to_r_bytes = key_mult_blocks * 2 * 16\n";
        std::cout << "    sooprf_f4_choice_s_to_r_bytes = f4_blocks * 2 * 16\n";
        std::cout << "    sooprf_f4_msg_r_to_s_bytes = f4_blocks * 2 * 16\n";
        std::cout << "    s3i_ole_correction_r_to_s_bytes = scalar_ole_correlations * 8\n";
        std::cout << "    s3i_ole_correction_s_to_r_bytes = scalar_ole_correlations * 8\n";
        std::cout << "    s3i_hs_reveal_s_to_r_bytes = N * 2 * (2*C) * 8\n";

        std::cout << "\n[count/accounting]\n";
        std::cout << "  membership_pairs=" << last_mem.programmed_pairs
                  << " total_tag_values=" << last_mem.total_tag_values
                  << " max_tag_size=" << last_mem.max_tag_size << "\n";
        std::cout << "  okvs_programmed_items=" << comm.okvs_programmed_items
                  << " okvs_query_items=" << comm.okvs_query_items
                  << " sooprf_query_pairs=" << comm.sooprf_query_pairs
                  << " okvs_cols=" << comm.okvs_cols
                  << " okvs_hint_table_bytes=" << comm.okvs_hint_bytes << "\n";
        std::cout << "  sooprf_packed_cols=" << comm.sooprf_packed_cols
                  << " key_mult_blocks=" << comm.sooprf_key_mult_blocks
                  << " key_mult_lanes=" << comm.sooprf_key_mult_lanes
                  << " f4_blocks=" << comm.sooprf_f4_blocks
                  << " f4_lanes=" << comm.sooprf_f4_lanes << "\n";
        std::cout << "  ot_tape_silent_gen_seconds=" << ot_cost.silent_gen_seconds
                  << " ot_tape_f4_correlations=" << ot_cost.f4_correlations
                  << " ot_tape_key_correlations=" << ot_cost.key_correlations
                  << " ot_tape_silent_ot_instances=" << ot_cost.silent_ot_instances << "\n";
        std::cout << "  s3i_scalar_ole_correlations=" << comm.s3i_scalar_ole_correlations
                  << " s3i_pcg_expanded_ole_correlations=" << comm.s3i_pcg_expanded_ole_correlations
                  << " pcg_expected_correlations=" << fpsi::s3i_required_ole_correlations(N, d, C)
                  << " pcg_refills=" << last_s3i.pcg_refills
                  << " ssrole_polymul_calls=" << last_s3i.ssrole_polymul_calls << "\n";
        std::cout << "  prof_subset_tree_seconds=" << last_s3i.prof_subset_tree_seconds
                  << " prof_rlc_seconds=" << last_s3i.prof_rlc_seconds
                  << " prof_gcd_seconds=" << last_s3i.prof_gcd_seconds
                  << " prof_s3i_build_seconds=" << last_s3i.prof_s3i_build_seconds << "\n";
    }

    if (cmax > C) {
        std::cerr << "ERROR: generated workload exceeds C\n";
        return 3;
    }
    if (observed_hits != forced_hits) {
        std::cerr << "ERROR: hit-count mismatch: forced_hits=" << forced_hits
                  << " observed_hits=" << observed_hits
                  << " (detects both false negatives and false positives)\n";
        return 4;
    }
    return 0;
}