#include "fpsi/fpsi.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fpsi {
namespace {

static std::uint64_t splitmix_stateless_local(std::uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct MembershipPoint {
    std::size_t axis = 0;
    std::uint64_t v = 0;

    friend bool operator==(MembershipPoint a, MembershipPoint b) noexcept {
        return a.axis == b.axis && a.v == b.v;
    }
};

struct MembershipPointHash {
    std::size_t operator()(MembershipPoint p) const noexcept {
        std::uint64_t h = splitmix_stateless_local(static_cast<std::uint64_t>(p.axis));
        h = splitmix_stateless_local(h ^ p.v ^ 0x4d454d4245524b56ULL);
        if constexpr (sizeof(std::size_t) >= 8) {
            return static_cast<std::size_t>(h);
        } else {
            return static_cast<std::size_t>(h ^ (h >> 32));
        }
    }
};

static void validate_flat_size(std::span<const std::uint64_t> flat,
                               std::size_t rows,
                               std::size_t d,
                               const char* name) {
    if (d == 0) throw std::invalid_argument("membership encoding requires d > 0");
    if (rows != 0 && flat.size() / rows != d) {
        throw std::invalid_argument(std::string(name) + " size is not rows*d");
    }
    if (flat.size() != rows * d) {
        throw std::invalid_argument(std::string(name) + " size is not rows*d");
    }
}

static std::pair<std::uint64_t, std::uint64_t> delta_window(std::uint64_t x,
                                                            std::uint64_t delta) noexcept {
    const std::uint64_t lo = (x > delta) ? (x - delta) : 0;
    const std::uint64_t hi = (std::numeric_limits<std::uint64_t>::max() - x < delta)
                           ? std::numeric_limits<std::uint64_t>::max()
                           : (x + delta);
    return {lo, hi};
}

} // namespace

Block128 membership_key(std::uint64_t v, std::size_t axis) noexcept {
    const std::uint64_t a = splitmix_stateless_local(v ^ 0x465053495f4d454dULL);
    const std::uint64_t b = splitmix_stateless_local(static_cast<std::uint64_t>(axis) ^
                                                    0x415849535f4d454dULL);
    const std::uint64_t lo = splitmix_stateless_local(a ^ (b << 1) ^ (b >> 3));
    const std::uint64_t hi = splitmix_stateless_local(b ^ (a << 7) ^ (a >> 11) ^
                                                     0x9e3779b97f4a7c15ULL);
    return Block128{lo, hi};
}

std::size_t membership_cmax_collision(std::span<const std::uint64_t> X_flat,
                                      std::size_t M,
                                      std::size_t d,
                                      std::uint64_t delta) {
    validate_flat_size(X_flat, M, d, "X_flat");
    if (M == 0) return 0;

    const std::uint64_t threshold = (delta > (std::numeric_limits<std::uint64_t>::max() / 2))
                                  ? std::numeric_limits<std::uint64_t>::max()
                                  : (2 * delta);
    std::vector<std::uint64_t> axis_values(M);
    std::size_t best = 0;
    for (std::size_t j = 0; j < d; ++j) {
        for (std::size_t i = 0; i < M; ++i) axis_values[i] = X_flat[i * d + j];
        std::sort(axis_values.begin(), axis_values.end());
        std::size_t left = 0;
        for (std::size_t right = 0; right < M; ++right) {
            while (left < right && axis_values[right] - axis_values[left] > threshold) ++left;
            best = std::max(best, right - left + 1);
        }
    }
    return best;
}

std::vector<KeySetPair> build_membership_pairs(std::span<const std::uint64_t> X_flat,
                                               std::size_t M,
                                               std::size_t d,
                                               std::uint64_t delta,
                                               std::size_t C_limit,
                                               MembershipEncodingStats* stats) {
    validate_flat_size(X_flat, M, d, "X_flat");

    if (stats) {
        *stats = MembershipEncodingStats{};
        stats->M = M;
        stats->d = d;
        stats->delta = delta;
    }

    std::unordered_map<MembershipPoint, std::vector<std::uint64_t>, MembershipPointHash> tags;
    if (M != 0) {
        // A lower-bound reserve: exact number may be smaller due to overlaps and boundary clipping.
        const std::uint64_t nominal_window = (delta > (std::numeric_limits<std::uint64_t>::max() - 1) / 2)
                                           ? std::numeric_limits<std::uint64_t>::max()
                                           : (2 * delta + 1);
        const auto max_size = std::numeric_limits<std::size_t>::max();
        if (M <= max_size / d) {
            const std::size_t md = M * d;
            if (nominal_window <= static_cast<std::uint64_t>(max_size / md)) {
                tags.reserve(static_cast<std::size_t>(nominal_window) * md);
            }
        }
    }

    for (std::size_t i = 0; i < M; ++i) {
        const std::uint64_t tag_value = static_cast<std::uint64_t>(i + 1);
        for (std::size_t j = 0; j < d; ++j) {
            const auto [lo, hi] = delta_window(X_flat[i * d + j], delta);
            for (std::uint64_t v = lo;; ++v) {
                tags[MembershipPoint{j, v}].push_back(tag_value);
                if (v == hi) break;
            }
        }
    }

    std::vector<std::pair<MembershipPoint, std::vector<std::uint64_t>>> ordered;
    ordered.reserve(tags.size());
    for (auto& kv : tags) ordered.push_back(std::move(kv));
    std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
        if (a.first.axis != b.first.axis) return a.first.axis < b.first.axis;
        return a.first.v < b.first.v;
    });

    std::vector<KeySetPair> out;
    out.reserve(ordered.size());
    for (auto& kv : ordered) {
        auto& values = kv.second;
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        if (C_limit != 0 && values.size() > C_limit) {
            throw std::invalid_argument("membership tag exceeds C_limit");
        }
        if (stats) {
            stats->max_tag_size = std::max(stats->max_tag_size, values.size());
            stats->total_tag_values += values.size();
        }
        out.push_back(KeySetPair{membership_key(kv.first.v, kv.first.axis), std::move(values)});
    }
    if (stats) stats->programmed_pairs = out.size();
    return out;
}

std::vector<Block128> build_sender_membership_keys(std::span<const std::uint64_t> Y_flat,
                                                   std::size_t N,
                                                   std::size_t d) {
    validate_flat_size(Y_flat, N, d, "Y_flat");
    std::vector<Block128> out(N * d);
    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t j = 0; j < d; ++j) {
            out[n * d + j] = membership_key(Y_flat[n * d + j], j);
        }
    }
    return out;
}

OssiBatchBitResult fpsi_match_ideal_batch_nonempty(std::span<const std::uint64_t> X_flat,
                                                   std::size_t M,
                                                   std::span<const std::uint64_t> Y_flat,
                                                   std::size_t N,
                                                   std::size_t d,
                                                   std::uint64_t delta,
                                                   const OssiParams& params,
                                                   MembershipEncodingStats* membership_stats) {
    validate_flat_size(X_flat, M, d, "X_flat");
    validate_flat_size(Y_flat, N, d, "Y_flat");
    auto pairs = build_membership_pairs(X_flat, M, d, delta, params.C, membership_stats);
    if (membership_stats) membership_stats->N = N;
    auto sender_keys = build_sender_membership_keys(Y_flat, N, d);
    return ossi_run_ideal_batch_nonempty(pairs, sender_keys, N, d, params);
}

} // namespace fpsi
