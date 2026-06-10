#include "fpsi/fpsi.h"

extern "C" {
#include "gf64.h"
#include "gf64_poly.h"
#include "gf64_lin.h"
#ifdef FPSI_HAVE_GF64X4
#include "gf64x4.h"
#include "gf64x4_gcd.h"
#include "gf64x4_linalg.h"
#endif
#include "dmpf64/ole_pcg_backend.h"
#include "dmpf64/gf64.h"
}

#include "sooprf/altmod_prf.h"
#include "sooprf/prng.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <cstring>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace fpsi {
namespace {

static inline double prof_elapsed(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}


using Poly = std::vector<std::uint64_t>; // low-to-high coefficients

static bool is_power_of_two(std::size_t x) noexcept {
    return x && ((x & (x - 1)) == 0);
}

static std::size_t s3i_subset_tree_ole_correlations_per_set(std::size_t C) {
    if (!is_power_of_two(C)) throw std::invalid_argument("S3I requires power-of-two C");
    std::size_t total = 0;
    for (std::size_t child = 1; child < C; child <<= 1) {
        const std::size_t pairs = C / (2 * child);
        const std::size_t eval_points = 2 * child - 1;
        // One SSrOLE multiplication needs two scalar cross-OLEs per evaluation point.
        total += pairs * 2 * eval_points;
    }
    return total;
}

static std::size_t s3i_required_ole_correlations_impl(std::size_t N, std::size_t d, std::size_t C) {
    if (!is_power_of_two(C)) throw std::invalid_argument("S3I requires power-of-two C");
    if (d < 2) throw std::invalid_argument("batched S3I requires d >= 2");
    const std::size_t subset = d * s3i_subset_tree_ole_correlations_per_set(C);

    // Two random linear combinations over a 2C-point evaluation domain, per axis.
    // The scalar fallback uses the generic SSrOLE routine and therefore consumes
    // two scalar OLEs per evaluation point.  The AVX512 batch-4 path has a
    // sender-private gamma specialization and consumes one scalar OLE per lane/point.
    const std::size_t rlc_scalar = 2 * d * (2 * (2 * C));
#if defined(FPSI_HAVE_GF64X4) && defined(FPSI_USE_GF64X4_GCD)
    const std::size_t rlc_x4 = 2 * d * (2 * C);
    const std::size_t full4 = N / 4;
    const std::size_t rem = N % 4;
    return full4 * 4 * (subset + rlc_x4) + rem * (subset + rlc_scalar);
#else
    return N * (subset + rlc_scalar);
#endif
}

static std::uint64_t splitmix_stateless(std::uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static void poly_trim(Poly& a) {
    a.resize(static_cast<std::size_t>(gf64_poly_trim_len(a.data(), static_cast<int>(a.size()))));
}

static int poly_deg(const Poly& a) {
    return gf64_poly_degree(a.data(), static_cast<int>(a.size()));
}

static void poly_make_monic(Poly& a) {
    if (a.empty()) return;
    const int d = gf64_poly_make_monic(a.data(), static_cast<int>(a.size()));
    a.resize(d < 0 ? 0u : static_cast<std::size_t>(d + 1));
}

static Poly poly_gcd(const Poly& a, const Poly& b) {
    const std::size_t cap = std::max<std::size_t>(1, std::max(a.size(), b.size()));
    Poly g(cap, 0);
    const int gd = gf64_poly_gcd(g.data(), static_cast<int>(g.size()),
                                 a.data(), static_cast<int>(a.size()),
                                 b.data(), static_cast<int>(b.size()));
    if (gd < 0) return {};
    g.resize(static_cast<std::size_t>(gd + 1));
    return g;
}

static std::vector<std::uint64_t> poly_roots_split(Poly f, std::uint64_t seed) {
    poly_trim(f);
    poly_make_monic(f);
    if (f.size() <= 1) return {};
    std::vector<std::uint64_t> roots(f.size() - 1, 0);
    const int n = gf64_poly_roots_split(roots.data(), static_cast<int>(roots.size()),
                                        f.data(), static_cast<int>(f.size()), seed);
    if (n < 0) {
        throw std::runtime_error("gf64_poly_roots_split failed; retry with another root_seed or check multiplicities");
    }
    roots.resize(static_cast<std::size_t>(n));
    return roots;
}

struct SharedPoly {
    Poly r;
    Poly s;
};

static std::size_t ceil_log2_size(std::size_t x) {
    std::size_t p = 1;
    std::size_t l = 0;
    while (p < x) { p <<= 1; ++l; }
    return l;
}

static void seed16_from_u64(std::uint8_t out[16], std::uint64_t a, std::uint64_t b) {
    for (int i = 0; i < 8; ++i) out[i] = static_cast<std::uint8_t>((a >> (8 * i)) & 0xffu);
    for (int i = 0; i < 8; ++i) out[8 + i] = static_cast<std::uint8_t>((b >> (8 * i)) & 0xffu);
}

struct PcgOleCorr {
    std::uint64_t m0 = 0;
    std::uint64_t a0 = 0;
    std::uint64_t m1 = 0;
    std::uint64_t a1 = 0;
};

class PcgOlePool {
public:
    explicit PcgOlePool(std::uint64_t seed, OssiStats* stats)
        : stats_(stats) {
        std::uint8_t s[16];
        seed16_from_u64(s, seed, splitmix_stateless(seed));
        dmpf64_rng_seed(&rng_, s);
    }

    PcgOleCorr next() {
        if (pos_ == pool_.size()) refill(kDefaultRefillLogN);
        return pool_[pos_++];
    }

    void ensure(std::size_t count) {
        if (pool_.size() - pos_ < count) {
            // Do not force a 2^20 OLE batch for small S^3I runs.  The benchmark
            // reports preprocessing/offline time, so over-materializing a huge
            // unused PCG tape makes small/medium workloads look artificially slow.
            refill(std::max<std::size_t>(kMinRefillLogN, ceil_log2_size(count)));
        }
    }

private:
    static constexpr std::size_t kMinRefillLogN = 8;
    static constexpr std::size_t kDefaultRefillLogN = 12;

    void refill(std::size_t logN) {
        const auto pcg_t0 = std::chrono::steady_clock::now();
        dmpf64_ole_seed_regular_any s0{};
        dmpf64_ole_seed_regular_any s1{};
        const std::size_t comp = 2;
        const std::size_t weight = 3;
        const std::size_t batch = 8;
        if (dmpf64_ole_gen_regular_backend(&s0, &s1, logN, comp, weight,
                                           DMPF64_BACKEND_BIG_STATE, batch, &rng_) != 0) {
            throw std::runtime_error("pcg OLE key generation failed");
        }
        dmpf64_ring_elem m0{};
        dmpf64_ring_elem a0{};
        dmpf64_ring_elem m1{};
        dmpf64_ring_elem a1{};
        if (dmpf64_ole_expand_regular_backend_eval(&s0, &m0, &a0) != 0 ||
            dmpf64_ole_expand_regular_backend_eval(&s1, &m1, &a1) != 0) {
            dmpf64_ole_seed_regular_any_free(&s0);
            dmpf64_ole_seed_regular_any_free(&s1);
            dmpf64_ring_free(&m0); dmpf64_ring_free(&a0); dmpf64_ring_free(&m1); dmpf64_ring_free(&a1);
            throw std::runtime_error("pcg OLE expansion failed");
        }
        const std::size_t N = static_cast<std::size_t>(1) << logN;
        if (m0.poly.len < N || a0.poly.len < N || m1.poly.len < N || a1.poly.len < N) {
            dmpf64_ole_seed_regular_any_free(&s0);
            dmpf64_ole_seed_regular_any_free(&s1);
            dmpf64_ring_free(&m0); dmpf64_ring_free(&a0); dmpf64_ring_free(&m1); dmpf64_ring_free(&a1);
            throw std::runtime_error("pcg OLE expansion returned short buffers");
        }
        pool_.clear();
        pool_.resize(N);
        for (std::size_t i = 0; i < N; ++i) {
            pool_[i] = PcgOleCorr{m0.poly.c[i], a0.poly.c[i], m1.poly.c[i], a1.poly.c[i]};
        }
        pos_ = 0;
        if (stats_) {
            stats_->pcg_refills++;
            const auto pcg_t1 = std::chrono::steady_clock::now();
            stats_->pcg_offline_seconds += std::chrono::duration<double>(pcg_t1 - pcg_t0).count();
        }
        dmpf64_ole_seed_regular_any_free(&s0);
        dmpf64_ole_seed_regular_any_free(&s1);
        dmpf64_ring_free(&m0); dmpf64_ring_free(&a0); dmpf64_ring_free(&m1); dmpf64_ring_free(&a1);
    }

    dmpf64_rng rng_{};
    std::vector<PcgOleCorr> pool_;
    std::size_t pos_ = 0;
    OssiStats* stats_ = nullptr;
};


struct VandermondeDomain {
    std::size_t L = 0;
    std::vector<std::uint64_t> alpha;     // L distinct field points
    std::vector<std::uint64_t> eval_mat;  // row-major: alpha_i^k
    std::vector<std::uint64_t> inv_mat;   // row-major: coefficient = inv_mat * values
};

static VandermondeDomain make_vandermonde_domain(std::size_t L) {
    if (L == 0) throw std::invalid_argument("zero-size Vandermonde domain");
    VandermondeDomain d;
    d.L = L;
    d.alpha.resize(L);
    d.eval_mat.assign(L * L, 0);
    d.inv_mat.assign(L * L, 0);

    // Small fixed public domain. Distinct uint64_t values are distinct GF(2^64)
    // elements.  The protocol only needs L <= 32 for C <= 16.
    for (std::size_t i = 0; i < L; ++i) d.alpha[i] = static_cast<std::uint64_t>(i + 1);
    if (gf64_vandermonde_matrices(d.alpha.data(), static_cast<int>(L),
                                  d.eval_mat.data(), d.inv_mat.data()) != 0) {
        throw std::runtime_error("singular Vandermonde matrix");
    }
    return d;
}

static std::map<std::size_t, VandermondeDomain>& vandermonde_domain_cache() {
    static std::map<std::size_t, VandermondeDomain> cache;
    return cache;
}

static const VandermondeDomain& vandermonde_domain(std::size_t L) {
    auto& cache = vandermonde_domain_cache();
    auto it = cache.find(L);
    if (it != cache.end()) return it->second;
    auto [pos, inserted] = cache.emplace(L, make_vandermonde_domain(L));
    (void)inserted;
    return pos->second;
}

struct VandermondeExtMap {
    std::size_t src_L = 0;
    std::size_t dst_L = 0;
    // tail_mat[(i-src_L)*src_L + k] gives the linear map from src evals to
    // evaluation at alpha_i, for i in [src_L,dst_L).  The first src_L rows are
    // implicit identity because our public domains are prefix-compatible.
    std::vector<std::uint64_t> tail_mat;
};

struct VandermondeShiftExtMap {
    std::size_t src_L = 0;
    std::size_t dst_L = 0;
    std::size_t shift = 0;
    // prefix_scale[i] = alpha_i^shift for i < src_L.
    std::vector<std::uint64_t> prefix_scale;
    // shifted_tail_mat[(i-src_L)*src_L + k] = alpha_i^shift * T_tail[i,k].
    std::vector<std::uint64_t> shifted_tail_mat;
};

static std::uint64_t ext_coeff_for_tail_row(const VandermondeDomain& src_dom,
                                            const VandermondeDomain& dst_dom,
                                            std::size_t dst_i,
                                            std::size_t src_k) {
    std::vector<std::uint64_t> col(src_dom.L, 0);
    for (std::size_t t = 0; t < src_dom.L; ++t) col[t] = src_dom.inv_mat[t * src_dom.L + src_k];
    return gf64_vec_dot(&dst_dom.eval_mat[dst_i * dst_dom.L], col.data(), static_cast<int>(src_dom.L));
}

static VandermondeExtMap make_vandermonde_ext_map(std::size_t src_L, std::size_t dst_L) {
    if (dst_L < src_L) throw std::invalid_argument("bad Vandermonde extension dimensions");
    VandermondeExtMap m;
    m.src_L = src_L;
    m.dst_L = dst_L;
    if (dst_L == src_L) return m;
    const auto& src_dom = vandermonde_domain(src_L);
    const auto& dst_dom = vandermonde_domain(dst_L);
    m.tail_mat.assign((dst_L - src_L) * src_L, 0);
    for (std::size_t i = src_L; i < dst_L; ++i) {
        for (std::size_t k = 0; k < src_L; ++k) {
            m.tail_mat[(i - src_L) * src_L + k] = ext_coeff_for_tail_row(src_dom, dst_dom, i, k);
        }
    }
    return m;
}

static std::map<std::pair<std::size_t, std::size_t>, VandermondeExtMap>& vandermonde_ext_cache() {
    static std::map<std::pair<std::size_t, std::size_t>, VandermondeExtMap> cache;
    return cache;
}

static const VandermondeExtMap& vandermonde_ext_map(std::size_t src_L, std::size_t dst_L) {
    auto& cache = vandermonde_ext_cache();
    const auto key = std::make_pair(src_L, dst_L);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    auto [pos, inserted] = cache.emplace(key, make_vandermonde_ext_map(src_L, dst_L));
    (void)inserted;
    return pos->second;
}

static VandermondeShiftExtMap make_vandermonde_shift_ext_map(std::size_t src_L,
                                                              std::size_t dst_L,
                                                              std::size_t shift) {
    if (dst_L < src_L) throw std::invalid_argument("bad shifted extension dimensions");
    VandermondeShiftExtMap m;
    m.src_L = src_L;
    m.dst_L = dst_L;
    m.shift = shift;
    const auto& dst_dom = vandermonde_domain(dst_L);
    m.prefix_scale.assign(src_L, 0);
    for (std::size_t i = 0; i < src_L; ++i) {
        m.prefix_scale[i] = dst_dom.eval_mat[i * dst_L + shift];
    }
    if (dst_L == src_L) return m;
    const auto& base = vandermonde_ext_map(src_L, dst_L);
    m.shifted_tail_mat.assign((dst_L - src_L) * src_L, 0);
    for (std::size_t i = src_L; i < dst_L; ++i) {
        const std::uint64_t alpha_shift = dst_dom.eval_mat[i * dst_L + shift];
        for (std::size_t k = 0; k < src_L; ++k) {
            m.shifted_tail_mat[(i - src_L) * src_L + k] =
                gf64_mul(alpha_shift, base.tail_mat[(i - src_L) * src_L + k]);
        }
    }
    return m;
}

struct ShiftKey {
    std::size_t src_L;
    std::size_t dst_L;
    std::size_t shift;
    bool operator<(const ShiftKey& o) const noexcept {
        if (src_L != o.src_L) return src_L < o.src_L;
        if (dst_L != o.dst_L) return dst_L < o.dst_L;
        return shift < o.shift;
    }
};

static std::map<ShiftKey, VandermondeShiftExtMap>& vandermonde_shift_cache() {
    static std::map<ShiftKey, VandermondeShiftExtMap> cache;
    return cache;
}

static const VandermondeShiftExtMap& vandermonde_shift_ext_map(std::size_t src_L,
                                                               std::size_t dst_L,
                                                               std::size_t shift) {
    auto& cache = vandermonde_shift_cache();
    const ShiftKey key{src_L, dst_L, shift};
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    auto [pos, inserted] = cache.emplace(key, make_vandermonde_shift_ext_map(src_L, dst_L, shift));
    (void)inserted;
    return pos->second;
}

static void vandermonde_eval_poly(std::vector<std::uint64_t>& out,
                                  const Poly& coeffs,
                                  std::size_t L) {
    const auto& dom = vandermonde_domain(L);
    out.assign(L, 0);
    std::vector<std::uint64_t> padded(L, 0);
    std::copy_n(coeffs.begin(), std::min(coeffs.size(), L), padded.begin());
    gf64_matvec_rowmajor(dom.eval_mat.data(), padded.data(), static_cast<int>(L), static_cast<int>(L), out.data());
}

static Poly vandermonde_interpolate_poly(const std::vector<std::uint64_t>& evals,
                                         std::size_t out_len) {
    const std::size_t L = evals.size();
    if (L != out_len) throw std::invalid_argument("Vandermonde interpolation length mismatch");
    const auto& dom = vandermonde_domain(L);
    Poly coeffs(L, 0);
    gf64_matvec_rowmajor(dom.inv_mat.data(), evals.data(), static_cast<int>(L), static_cast<int>(L), coeffs.data());
    poly_trim(coeffs);
    return coeffs;
}

static std::vector<std::uint64_t> vandermonde_extend_eval(const std::vector<std::uint64_t>& src,
                                                          std::size_t dst_L) {
    const std::size_t src_L = src.size();
    if (dst_L < src_L) throw std::invalid_argument("cannot shrink evaluation vector");
    std::vector<std::uint64_t> out(dst_L, 0);
    std::copy(src.begin(), src.end(), out.begin());
    if (dst_L == src_L) return out;
    const auto& m = vandermonde_ext_map(src_L, dst_L);
    for (std::size_t i = src_L; i < dst_L; ++i) {
        const std::uint64_t* row = &m.tail_mat[(i - src_L) * src_L];
        out[i] = gf64_vec_dot(row, src.data(), static_cast<int>(src_L));
    }
    return out;
}

static std::vector<std::uint64_t> vandermonde_eval_monomial_shifted_sum(
    const std::vector<std::uint64_t>& left_eval,
    const std::vector<std::uint64_t>& right_eval,
    std::size_t r,
    std::size_t dst_L) {
    const std::size_t src_L = left_eval.size();
    if (right_eval.size() != src_L) throw std::invalid_argument("shifted sum length mismatch");
    if (dst_L < src_L) throw std::invalid_argument("cannot shrink shifted extension");
    const auto& m = vandermonde_shift_ext_map(src_L, dst_L, r);
    std::vector<std::uint64_t> out(dst_L, 0);
    std::vector<std::uint64_t> sum(src_L, 0);
    for (std::size_t i = 0; i < src_L; ++i) {
        sum[i] = left_eval[i] ^ right_eval[i];
        out[i] = gf64_mul(m.prefix_scale[i], sum[i]);
    }
    for (std::size_t i = src_L; i < dst_L; ++i) {
        const std::uint64_t* row = &m.shifted_tail_mat[(i - src_L) * src_L];
        out[i] = gf64_vec_dot(row, sum.data(), static_cast<int>(src_L));
    }
    return out;
}

static void precompute_vandermonde_for_C_impl(std::size_t C) {
    if (!is_power_of_two(C)) throw std::invalid_argument("C must be a power of two for S3I precompute");
    const std::size_t final_L = 2 * C;
    // Domains needed by subset-tree levels and the final RLC/interpolation.
    (void)vandermonde_domain(final_L);
    std::size_t child_size = 1;
    while (child_size < C) {
        const std::size_t src_L = 2 * child_size - 1;
        const std::size_t parent_size = child_size << 1;
        const std::size_t dst_L = (parent_size == C) ? final_L : (2 * parent_size - 1);
        (void)vandermonde_domain(src_L);
        (void)vandermonde_domain(dst_L);
        (void)vandermonde_ext_map(src_L, dst_L);
        (void)vandermonde_shift_ext_map(src_L, dst_L, child_size);
        child_size = parent_size;
    }
}

static inline void pcg_cross_product_share(std::uint64_t x0,
                                           std::uint64_t y1,
                                           const PcgOleCorr& c,
                                           std::uint64_t& z0,
                                           std::uint64_t& z1) {
    // OLE/PCG correlation: c.m0 * c.m1 = c.a0 + c.a1.
    // Open e=x0+m0 and f=y1+m1, then output additive shares of x0*y1.
    const std::uint64_t e = x0 ^ c.m0;
    const std::uint64_t f = y1 ^ c.m1;
    const std::uint64_t ef = gf64_mul(e, f);
    z0 = gf64_mul(f, c.m0) ^ c.a0 ^ ef;
    z1 = gf64_mul(e, c.m1) ^ c.a1;
}


struct SharedEvalPoly {
    std::vector<std::uint64_t> r_eval;
    std::vector<std::uint64_t> s_eval;
    std::size_t degree_bound = 0; // polynomial degree < degree_bound
};

static SharedEvalPoly pcg_ssrole_mul_eval_only(const std::vector<std::uint64_t>& a_r_eval,
                                               const std::vector<std::uint64_t>& a_s_eval,
                                               const std::vector<std::uint64_t>& b_r_eval,
                                               const std::vector<std::uint64_t>& b_s_eval,
                                               PcgOlePool& ole,
                                               OssiStats* stats) {
    const std::size_t L = std::max({a_r_eval.size(), a_s_eval.size(),
                                    b_r_eval.size(), b_s_eval.size()});
    if (L == 0) return {};
    const auto prof_t0 = std::chrono::steady_clock::now();
    const double pcg_before = stats ? stats->pcg_offline_seconds : 0.0;
    auto at = [](const std::vector<std::uint64_t>& v, std::size_t i) -> std::uint64_t {
        return i < v.size() ? v[i] : 0;
    };

    ole.ensure(2 * L);
    SharedEvalPoly out;
    out.r_eval.assign(L, 0);
    out.s_eval.assign(L, 0);
    out.degree_bound = L;
    for (std::size_t t = 0; t < L; ++t) {
        const std::uint64_t ar = at(a_r_eval, t);
        const std::uint64_t as = at(a_s_eval, t);
        const std::uint64_t br = at(b_r_eval, t);
        const std::uint64_t bs = at(b_s_eval, t);
        const std::uint64_t local_r = gf64_mul(ar, br);
        const std::uint64_t local_s = gf64_mul(as, bs);
        std::uint64_t c10 = 0, c11 = 0, c20 = 0, c21 = 0;
        pcg_cross_product_share(ar, bs, ole.next(), c10, c11);
        pcg_cross_product_share(br, as, ole.next(), c20, c21);
        out.r_eval[t] = local_r ^ c10 ^ c20;
        out.s_eval[t] = local_s ^ c11 ^ c21;
    }
    if (stats) {
        stats->ssrole_polymul_calls++;
        stats->scalar_ole_equivalent += 2 * L;
        stats->pcg_ole_correlations += 2 * L;
        const double pcg_delta = stats->pcg_offline_seconds - pcg_before;
        stats->prof_ssrole_online_seconds += prof_elapsed(prof_t0) - pcg_delta;
    }
    return out;
}


struct S3iHPair {
    Poly h1;
    Poly h2;
};

#if defined(FPSI_HAVE_GF64X4)
struct PackedSharedEvalPoly {
    std::vector<__m512i> r_eval;
    std::vector<__m512i> s_eval;
    std::size_t degree_bound = 0;
};

struct S3iHPair4 {
    std::vector<__m512i> h1;
    std::vector<__m512i> h2;
};

static std::vector<__m512i> vandermonde_interpolate_poly_x4(const std::vector<__m512i>& evals,
                                                            std::size_t out_len) {
    const std::size_t L = evals.size();
    if (L != out_len) throw std::invalid_argument("x4 Vandermonde interpolation length mismatch");
    const auto& dom = vandermonde_domain(L);
    std::vector<__m512i> coeffs(L, gf64x4_zero());
    gf64x4_poly_eval_to_coeff(reinterpret_cast<const int64_t*>(dom.inv_mat.data()),
                              static_cast<int>(L), evals.data(), static_cast<int>(L),
                              coeffs.data());
    return coeffs;
}

static PackedSharedEvalPoly pcg_ssrole_mul_eval_only_x4(const std::vector<__m512i>& a_r_eval,
                                                        const std::vector<__m512i>& a_s_eval,
                                                        const std::vector<__m512i>& b_r_eval,
                                                        const std::vector<__m512i>& b_s_eval,
                                                        PcgOlePool& ole,
                                                        OssiStats* stats) {
    const std::size_t L = std::max({a_r_eval.size(), a_s_eval.size(), b_r_eval.size(), b_s_eval.size()});
    if (L == 0) return {};
    const auto prof_t0 = std::chrono::steady_clock::now();
    const double pcg_before = stats ? stats->pcg_offline_seconds : 0.0;
    auto at = [](const std::vector<__m512i>& v, std::size_t i) -> __m512i {
        return i < v.size() ? v[i] : gf64x4_zero();
    };

    ole.ensure(8 * L); // 4 lanes * 2 cross-products per evaluation point.
    PackedSharedEvalPoly out;
    out.r_eval.assign(L, gf64x4_zero());
    out.s_eval.assign(L, gf64x4_zero());
    out.degree_bound = L;
    for (std::size_t t = 0; t < L; ++t) {
        const __m512i ar = at(a_r_eval, t);
        const __m512i as = at(a_s_eval, t);
        const __m512i br = at(b_r_eval, t);
        const __m512i bs = at(b_s_eval, t);
        const __m512i local_r = gf64x4_mul(ar, br);
        const __m512i local_s = gf64x4_mul(as, bs);

        std::uint64_t ar_v[4], as_v[4], br_v[4], bs_v[4];
        gf64x4_unpack(ar, ar_v);
        gf64x4_unpack(as, as_v);
        gf64x4_unpack(br, br_v);
        gf64x4_unpack(bs, bs_v);
        std::uint64_t cr1[4], cs1[4], cr2[4], cs2[4];
        for (int lane = 0; lane < 4; ++lane) {
            pcg_cross_product_share(ar_v[lane], bs_v[lane], ole.next(), cr1[lane], cs1[lane]);
            pcg_cross_product_share(br_v[lane], as_v[lane], ole.next(), cr2[lane], cs2[lane]);
        }
        const __m512i cross_r = gf64x4_add(gf64x4_pack(cr1), gf64x4_pack(cr2));
        const __m512i cross_s = gf64x4_add(gf64x4_pack(cs1), gf64x4_pack(cs2));
        out.r_eval[t] = gf64x4_add(local_r, cross_r);
        out.s_eval[t] = gf64x4_add(local_s, cross_s);
    }
    if (stats) {
        stats->ssrole_polymul_calls += 4;
        stats->scalar_ole_equivalent += 8 * L;
        stats->pcg_ole_correlations += 8 * L;
        const double pcg_delta = stats->pcg_offline_seconds - pcg_before;
        stats->prof_ssrole_online_seconds += prof_elapsed(prof_t0) - pcg_delta;
    }
    return out;
}


static PackedSharedEvalPoly make_subset_parent_fused_x4(const PackedSharedEvalPoly& left,
                                                        const PackedSharedEvalPoly& right,
                                                        const PackedSharedEvalPoly& prod,
                                                        std::size_t child_size,
                                                        std::size_t out_L,
                                                        OssiStats* stats) {
    const std::size_t src_L = left.r_eval.size();
    const auto& ext = vandermonde_ext_map(prod.r_eval.size(), out_L);
    const auto& sh = vandermonde_shift_ext_map(src_L, out_L, child_size);
    if (right.r_eval.size() != src_L || prod.r_eval.size() != src_L || ext.src_L != src_L || sh.src_L != src_L) {
        throw std::runtime_error("x4 fused subset parent dimension mismatch");
    }

    PackedSharedEvalPoly parent;
    parent.r_eval.assign(out_L, gf64x4_zero());
    parent.s_eval.assign(out_L, gf64x4_zero());
    parent.degree_bound = child_size << 1;

    const auto prof_shift_t0 = std::chrono::steady_clock::now();
    for (std::size_t t = 0; t < src_L; ++t) {
        const __m512i sum_r = gf64x4_add(left.r_eval[t], right.r_eval[t]);
        const __m512i sum_s = gf64x4_add(left.s_eval[t], right.s_eval[t]);
        const __m512i lin_r = gf64x4_mul(gf64x4_set1_u64(sh.prefix_scale[t]), sum_r);
        const __m512i lin_s = gf64x4_mul(gf64x4_set1_u64(sh.prefix_scale[t]), sum_s);
        parent.r_eval[t] = gf64x4_add(prod.r_eval[t], lin_r);
        parent.s_eval[t] = gf64x4_add(prod.s_eval[t], lin_s);
    }
    if (stats) stats->prof_vand_shift_seconds += prof_elapsed(prof_shift_t0);

    const auto prof_tail_t0 = std::chrono::steady_clock::now();
    for (std::size_t t = src_L; t < out_L; ++t) {
        const std::uint64_t* erow = &ext.tail_mat[(t - src_L) * src_L];
        const std::uint64_t* srow = &sh.shifted_tail_mat[(t - src_L) * src_L];
        __m512i acc_r = gf64x4_zero();
        __m512i acc_s = gf64x4_zero();
        for (std::size_t k = 0; k < src_L; ++k) {
            const __m512i sum_r = gf64x4_add(left.r_eval[k], right.r_eval[k]);
            const __m512i sum_s = gf64x4_add(left.s_eval[k], right.s_eval[k]);
            if (erow[k]) {
                acc_r = gf64x4_add(acc_r, gf64x4_mul_const_no_red(erow[k], prod.r_eval[k]));
                acc_s = gf64x4_add(acc_s, gf64x4_mul_const_no_red(erow[k], prod.s_eval[k]));
            }
            if (srow[k]) {
                acc_r = gf64x4_add(acc_r, gf64x4_mul_const_no_red(srow[k], sum_r));
                acc_s = gf64x4_add(acc_s, gf64x4_mul_const_no_red(srow[k], sum_s));
            }
        }
        parent.r_eval[t] = gf64x4_reduce(acc_r);
        parent.s_eval[t] = gf64x4_reduce(acc_s);
    }
    if (stats) stats->prof_vand_extend_seconds += prof_elapsed(prof_tail_t0);
    return parent;
}

static PackedSharedEvalPoly encode_shared_set_subset_tree_fused_x4(const std::array<const std::uint64_t*, 4>& wr,
                                                                   const std::array<const std::uint64_t*, 4>& ws,
                                                                   std::size_t C,
                                                                   PcgOlePool& ole,
                                                                   OssiStats* stats) {
    const auto prof_subset_t0 = std::chrono::steady_clock::now();
    const double pcg_subset_before = stats ? stats->pcg_offline_seconds : 0.0;
    const std::size_t final_L = 2 * C;
    std::vector<PackedSharedEvalPoly> level(C);
    for (std::size_t i = 0; i < C; ++i) {
        level[i].r_eval = {gf64x4_pack4(wr[0][i], wr[1][i], wr[2][i], wr[3][i])};
        level[i].s_eval = {gf64x4_pack4(ws[0][i], ws[1][i], ws[2][i], ws[3][i])};
        level[i].degree_bound = 1;
    }

    std::size_t child_size = 1;
    while (level.size() > 1) {
        const std::size_t mul_L = 2 * child_size - 1;
        const std::size_t parent_size = child_size << 1;
        const std::size_t out_L = (parent_size == C) ? final_L : (2 * parent_size - 1);
        std::vector<PackedSharedEvalPoly> next;
        next.reserve(level.size() / 2);
        for (std::size_t i = 0; i < level.size(); i += 2) {
            const auto& left = level[i];
            const auto& right = level[i + 1];
            if (left.r_eval.size() != mul_L || right.r_eval.size() != mul_L) {
                throw std::runtime_error("x4 S3I subset tree domain mismatch");
            }
            PackedSharedEvalPoly prod = pcg_ssrole_mul_eval_only_x4(left.r_eval, left.s_eval,
                                                                    right.r_eval, right.s_eval,
                                                                    ole, stats);
            PackedSharedEvalPoly parent = make_subset_parent_fused_x4(left, right, prod,
                                                                      child_size, out_L, stats);
            parent.degree_bound = parent_size;
            next.push_back(std::move(parent));
        }
        level = std::move(next);
        child_size = parent_size;
    }
    if (level[0].r_eval.size() != final_L) {
        throw std::runtime_error("x4 S3I subset tree final domain mismatch");
    }
    if (stats) {
        const double pcg_delta = stats->pcg_offline_seconds - pcg_subset_before;
        stats->prof_subset_tree_seconds += prof_elapsed(prof_subset_t0) - pcg_delta;
    }
    return std::move(level[0]);
}

static std::vector<std::vector<__m512i>> sample_gamma_eval_x4(std::size_t d,
                                                               std::size_t C,
                                                               const std::uint64_t lane_seeds[4],
                                                               OssiStats* stats) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::size_t L = 2 * C;
    const auto& ext = vandermonde_ext_map(C, L); // V_bot * V_top^{-1}; exactly the Vcr used by gf64x4_linalg.
    sooprf::SplitMix64 rng0(lane_seeds[0]);
    sooprf::SplitMix64 rng1(lane_seeds[1]);
    sooprf::SplitMix64 rng2(lane_seeds[2]);
    sooprf::SplitMix64 rng3(lane_seeds[3]);
    std::vector<std::vector<__m512i>> gamma_eval(d, std::vector<__m512i>(L, gf64x4_zero()));
    std::vector<__m512i> top(C, gf64x4_zero());
    for (std::size_t j = 0; j < d; ++j) {
        // Sample the first C evaluations uniformly and extend them to the 2C-point
        // evaluation domain. This is equivalent to sampling a random polynomial of
        // degree < C, and is exactly gf64x4_linalg.c's random-sample algorithm.
        for (std::size_t k = 0; k < C; ++k) {
            top[k] = gf64x4_pack4(rng0.next_u64(), rng1.next_u64(),
                                  rng2.next_u64(), rng3.next_u64());
        }
        gf64x4_poly_random_sample(reinterpret_cast<const int64_t*>(ext.tail_mat.data()),
                                  static_cast<int>(L), top.data(), static_cast<int>(C),
                                  gamma_eval[j].data());
    }
    if (stats) stats->prof_vand_gamma_eval_seconds += prof_elapsed(t0);
    return gamma_eval;
}

static void pcg_mul_s_private_accum_x4(const std::vector<__m512i>& a_r_eval,
                                       const std::vector<__m512i>& a_s_eval,
                                       const std::vector<__m512i>& gamma_eval,
                                       PcgOlePool& ole,
                                       std::vector<__m512i>& h_r_eval,
                                       std::vector<__m512i>& h_s_eval,
                                       OssiStats* stats) {
    const std::size_t L = gamma_eval.size();
    if (a_r_eval.size() != L || a_s_eval.size() != L || h_r_eval.size() != L || h_s_eval.size() != L) {
        throw std::runtime_error("x4 S-owned multiplication accumulator dimension mismatch");
    }
    const auto prof_t0 = std::chrono::steady_clock::now();
    const double pcg_before = stats ? stats->pcg_offline_seconds : 0.0;
    ole.ensure(4 * L); // 4 lanes * one cross-product per evaluation point.
    for (std::size_t t = 0; t < L; ++t) {
        const __m512i ar = a_r_eval[t];
        const __m512i as = a_s_eval[t];
        const __m512i gam = gamma_eval[t];
        const __m512i local_s = gf64x4_mul(as, gam);

        std::uint64_t ar_v[4], gam_v[4];
        gf64x4_unpack(ar, ar_v);
        gf64x4_unpack(gam, gam_v);
        std::uint64_t cr[4], cs[4];
        for (int lane = 0; lane < 4; ++lane) {
            pcg_cross_product_share(ar_v[lane], gam_v[lane], ole.next(), cr[lane], cs[lane]);
        }
        h_r_eval[t] = gf64x4_add(h_r_eval[t], gf64x4_pack(cr));
        h_s_eval[t] = gf64x4_add(h_s_eval[t], gf64x4_add(local_s, gf64x4_pack(cs)));
    }
    if (stats) {
        stats->ssrole_polymul_calls += 4;
        stats->scalar_ole_equivalent += 4 * L;
        stats->pcg_ole_correlations += 4 * L;
        const double pcg_delta = stats->pcg_offline_seconds - pcg_before;
        stats->prof_ssrole_online_seconds += prof_elapsed(prof_t0) - pcg_delta;
    }
}

static std::vector<__m512i> build_random_linear_combination_fused_x4(
    const std::vector<PackedSharedEvalPoly>& g_eval,
    const std::vector<std::vector<__m512i>>& gamma_eval,
    std::size_t C,
    PcgOlePool& ole,
    OssiStats* stats) {
    const auto prof_rlc_t0 = std::chrono::steady_clock::now();
    const double pcg_rlc_before = stats ? stats->pcg_offline_seconds : 0.0;
    const std::size_t d = g_eval.size();
    const std::size_t L = 2 * C;
    std::vector<__m512i> h_r_eval(L, gf64x4_zero()), h_s_eval(L, gf64x4_zero());
    const auto& dom = vandermonde_domain(L);

    for (std::size_t j = 0; j < d; ++j) {
        if (gamma_eval[j].size() != L) throw std::runtime_error("x4 gamma eval domain length mismatch");
        // gamma is S-owned. Therefore gamma * g_j needs only one cross OLE per
        // evaluation point: R contributes g_j^R and S contributes gamma and g_j^S.
        pcg_mul_s_private_accum_x4(g_eval[j].r_eval, g_eval[j].s_eval,
                                   gamma_eval[j], ole, h_r_eval, h_s_eval, stats);
        for (std::size_t t = 0; t < L; ++t) {
            const __m512i monic = gf64x4_mul(gf64x4_set1_u64(dom.eval_mat[t * L + C]), gamma_eval[j][t]);
            h_s_eval[t] = gf64x4_add(h_s_eval[t], monic);
        }
    }

    std::vector<__m512i> h_eval(L, gf64x4_zero());
    for (std::size_t t = 0; t < L; ++t) h_eval[t] = gf64x4_add(h_r_eval[t], h_s_eval[t]);
    const auto prof_interp_t0 = std::chrono::steady_clock::now();
    std::vector<__m512i> coeffs = vandermonde_interpolate_poly_x4(h_eval, L);
    if (stats) {
        stats->prof_vand_interpolate_seconds += prof_elapsed(prof_interp_t0);
        const double pcg_delta = stats->pcg_offline_seconds - pcg_rlc_before;
        stats->prof_rlc_seconds += prof_elapsed(prof_rlc_t0) - pcg_delta;
    }
    return coeffs;
}

static S3iHPair4 build_s3i_h_pair4_with_pool(std::span<const std::uint64_t> receiver_share_all,
                                             std::span<const std::uint64_t> sender_share_all,
                                             std::size_t q0,
                                             std::size_t one,
                                             std::size_t d,
                                             std::size_t C,
                                             PcgOlePool& ole,
                                             const std::uint64_t gamma_seed_lanes[4],
                                             OssiStats* stats) {
    const auto prof_s3i_t0 = std::chrono::steady_clock::now();
    const double pcg_s3i_before = stats ? stats->pcg_offline_seconds : 0.0;
    std::vector<PackedSharedEvalPoly> encodings;
    encodings.reserve(d);
    for (std::size_t j = 0; j < d; ++j) {
        std::array<const std::uint64_t*, 4> wr{};
        std::array<const std::uint64_t*, 4> ws{};
        for (std::size_t lane = 0; lane < 4; ++lane) {
            const std::size_t off = (q0 + lane) * one + j * C;
            wr[lane] = receiver_share_all.data() + off;
            ws[lane] = sender_share_all.data() + off;
        }
        encodings.push_back(encode_shared_set_subset_tree_fused_x4(wr, ws, C, ole, stats));
    }

    std::uint64_t seeds1[4];
    std::uint64_t seeds2[4];
    for (int lane = 0; lane < 4; ++lane) {
        seeds1[lane] = gamma_seed_lanes[lane] ^ 0x1111111111111111ULL;
        seeds2[lane] = gamma_seed_lanes[lane] ^ 0x2222222222222222ULL;
    }
    auto gamma1 = sample_gamma_eval_x4(d, C, seeds1, stats);
    auto gamma2 = sample_gamma_eval_x4(d, C, seeds2, stats);
    S3iHPair4 out{build_random_linear_combination_fused_x4(encodings, gamma1, C, ole, stats),
                   build_random_linear_combination_fused_x4(encodings, gamma2, C, ole, stats)};
    if (stats) {
        const double pcg_delta = stats->pcg_offline_seconds - pcg_s3i_before;
        stats->prof_s3i_build_seconds += prof_elapsed(prof_s3i_t0) - pcg_delta;
    }
    return out;
}

static bool can_batch4_gcd_packed(const S3iHPair4& hp, std::size_t coeffs) {
    if (hp.h1.size() != coeffs || hp.h2.size() != coeffs || coeffs == 0) return false;
    std::uint64_t top1[4], top2[4];
    gf64x4_unpack(hp.h1.back(), top1);
    gf64x4_unpack(hp.h2.back(), top2);
    for (int lane = 0; lane < 4; ++lane) {
        if (top1[lane] == 0 || top2[lane] == 0) return false;
    }
    return true;
}

static void batch4_gcd_nonempty_packed(const S3iHPair4& hp,
                                       std::size_t coeffs,
                                       int out[4]) {
    const int Lc = static_cast<int>(coeffs - 1) + 2;
    std::vector<__m512i> work(static_cast<std::size_t>(3 * Lc));
    gf64x4_coprime_test(hp.h1.data(), static_cast<int>(coeffs),
                        hp.h2.data(), static_cast<int>(coeffs),
                        work.data(), out);
}

static std::array<S3iHPair, 4> unpack_hp4(const S3iHPair4& hp4) {
    std::array<S3iHPair, 4> hp;
    const std::size_t coeffs = hp4.h1.size();
    for (int lane = 0; lane < 4; ++lane) {
        hp[lane].h1.assign(coeffs, 0);
        hp[lane].h2.assign(coeffs, 0);
    }
    for (std::size_t i = 0; i < coeffs; ++i) {
        std::uint64_t h1v[4], h2v[4];
        gf64x4_unpack(hp4.h1[i], h1v);
        gf64x4_unpack(hp4.h2[i], h2v);
        for (int lane = 0; lane < 4; ++lane) {
            hp[lane].h1[i] = h1v[lane];
            hp[lane].h2[i] = h2v[lane];
        }
    }
    for (int lane = 0; lane < 4; ++lane) {
        poly_trim(hp[lane].h1);
        poly_trim(hp[lane].h2);
    }
    return hp;
}
#endif // FPSI_HAVE_GF64X4

static SharedEvalPoly encode_shared_set_subset_tree_fused(std::span<const std::uint64_t> wr,
                                                          std::span<const std::uint64_t> ws,
                                                          std::size_t C,
                                                          PcgOlePool& ole,
                                                          OssiStats* stats) {
    const auto prof_subset_t0 = std::chrono::steady_clock::now();
    const double pcg_subset_before = stats ? stats->pcg_offline_seconds : 0.0;
    if (wr.size() != C || ws.size() != C) throw std::invalid_argument("bad set-share length");
    const std::size_t final_L = 2 * C;

    std::vector<SharedEvalPoly> level(C);
    for (std::size_t i = 0; i < C; ++i) {
        // Leaf lower polynomial is the constant w.  For the first multiplication
        // domain L=1, its evaluation vector has one entry.
        level[i].r_eval = {wr[i]};
        level[i].s_eval = {ws[i]};
        level[i].degree_bound = 1;
    }

    std::size_t child_size = 1;
    while (level.size() > 1) {
        const std::size_t mul_L = 2 * child_size - 1;
        const std::size_t parent_size = child_size << 1;
        const std::size_t out_L = (parent_size == C) ? final_L : (2 * parent_size - 1);
        std::vector<SharedEvalPoly> next;
        next.reserve(level.size() / 2);

        for (std::size_t i = 0; i < level.size(); i += 2) {
            const SharedEvalPoly& left = level[i];
            const SharedEvalPoly& right = level[i + 1];
            if (left.r_eval.size() != mul_L || right.r_eval.size() != mul_L) {
                throw std::runtime_error("S3I fused subset tree domain mismatch");
            }

            // Product lower term g_left * g_right is obtained in the current
            // multiplication domain only.  Then it is extended to the next domain
            // by the fused map V_out * V_mul^{-1}; the prefix rows are identity.
            SharedEvalPoly prod = pcg_ssrole_mul_eval_only(left.r_eval, left.s_eval,
                                                           right.r_eval, right.s_eval,
                                                           ole, stats);
            const auto prof_ext_t0 = std::chrono::steady_clock::now();
            std::vector<std::uint64_t> prod_r_ext = vandermonde_extend_eval(prod.r_eval, out_L);
            std::vector<std::uint64_t> prod_s_ext = vandermonde_extend_eval(prod.s_eval, out_L);
            if (stats) stats->prof_vand_extend_seconds += prof_elapsed(prof_ext_t0);

            const auto prof_shift_t0 = std::chrono::steady_clock::now();
            std::vector<std::uint64_t> lin_r =
                vandermonde_eval_monomial_shifted_sum(left.r_eval, right.r_eval,
                                                      child_size, out_L);
            std::vector<std::uint64_t> lin_s =
                vandermonde_eval_monomial_shifted_sum(left.s_eval, right.s_eval,
                                                      child_size, out_L);
            if (stats) stats->prof_vand_shift_seconds += prof_elapsed(prof_shift_t0);

            SharedEvalPoly parent;
            parent.r_eval.resize(out_L);
            parent.s_eval.resize(out_L);
            parent.degree_bound = parent_size;
            for (std::size_t t = 0; t < out_L; ++t) {
                parent.r_eval[t] = prod_r_ext[t] ^ lin_r[t];
                parent.s_eval[t] = prod_s_ext[t] ^ lin_s[t];
            }
            next.push_back(std::move(parent));
        }
        level = std::move(next);
        child_size = parent_size;
    }
    if (level[0].r_eval.size() != final_L) {
        throw std::runtime_error("S3I fused subset tree final domain mismatch");
    }
    if (stats) {
        const double pcg_delta = stats->pcg_offline_seconds - pcg_subset_before;
        stats->prof_subset_tree_seconds += prof_elapsed(prof_subset_t0) - pcg_delta;
    }
    return std::move(level[0]); // evaluations of g_W where f_W(X)=X^C+g_W(X)
}

static std::vector<Poly> sample_gamma(std::size_t d, std::size_t C, sooprf::SplitMix64& rng) {
    std::vector<Poly> gamma(d);
    for (std::size_t j = 0; j < d; ++j) {
        gamma[j].resize(C, 0); // degree <= C-1
        for (auto& c : gamma[j]) c = rng.next_u64();
        poly_trim(gamma[j]);
    }
    return gamma;
}

static Poly build_random_linear_combination_fused(const std::vector<SharedEvalPoly>& g_eval,
                                                  const std::vector<Poly>& gamma,
                                                  std::size_t C,
                                                  PcgOlePool& ole,
                                                  OssiStats* stats) {
    const auto prof_rlc_t0 = std::chrono::steady_clock::now();
    const double pcg_rlc_before = stats ? stats->pcg_offline_seconds : 0.0;
    const std::size_t d = g_eval.size();
    const std::size_t L = 2 * C; // H has degree at most 2C-1.
    std::vector<std::uint64_t> h_r_eval(L, 0), h_s_eval(L, 0);
    const auto& dom = vandermonde_domain(L);

    for (std::size_t j = 0; j < d; ++j) {
        std::vector<std::uint64_t> gamma_eval;
        const auto prof_gamma_t0 = std::chrono::steady_clock::now();
        vandermonde_eval_poly(gamma_eval, gamma[j], L);
        if (stats) stats->prof_vand_gamma_eval_seconds += prof_elapsed(prof_gamma_t0);

        // gamma is S-owned: multiply shared g_j by shared gamma=(0,gamma_j).
        SharedEvalPoly prod = pcg_ssrole_mul_eval_only(g_eval[j].r_eval, g_eval[j].s_eval,
                                                       {}, gamma_eval, ole, stats);
        for (std::size_t t = 0; t < L; ++t) {
            h_r_eval[t] ^= prod.r_eval[t];
            // Add gamma_j(alpha_t) * alpha_t^C for the monic X^C term of f_j.
            h_s_eval[t] ^= prod.s_eval[t] ^ gf64_mul(gamma_eval[t], dom.eval_mat[t * L + C]);
        }
    }

    std::vector<std::uint64_t> h_eval(L, 0);
    for (std::size_t t = 0; t < L; ++t) h_eval[t] = h_r_eval[t] ^ h_s_eval[t];
    const auto prof_interp_t0 = std::chrono::steady_clock::now();
    Poly coeffs = vandermonde_interpolate_poly(h_eval, L);
    if (stats) {
        stats->prof_vand_interpolate_seconds += prof_elapsed(prof_interp_t0);
        const double pcg_delta = stats->pcg_offline_seconds - pcg_rlc_before;
        stats->prof_rlc_seconds += prof_elapsed(prof_rlc_t0) - pcg_delta;
    }
    return coeffs;
}


static S3iHPair build_s3i_h_pair_with_pool(std::span<const std::uint64_t> receiver_share,
                                           std::span<const std::uint64_t> sender_share,
                                           std::size_t d,
                                           std::size_t C,
                                           PcgOlePool& ole,
                                           std::uint64_t gamma_seed,
                                           OssiStats* stats) {
    const auto prof_s3i_t0 = std::chrono::steady_clock::now();
    const double pcg_s3i_before = stats ? stats->pcg_offline_seconds : 0.0;
    std::vector<SharedEvalPoly> encodings;
    encodings.reserve(d);
    for (std::size_t j = 0; j < d; ++j) {
        encodings.push_back(encode_shared_set_subset_tree_fused(receiver_share.subspan(j * C, C),
                                                                sender_share.subspan(j * C, C),
                                                                C, ole, stats));
    }

    sooprf::SplitMix64 gamma_rng1(gamma_seed ^ 0x1111111111111111ULL);
    sooprf::SplitMix64 gamma_rng2(gamma_seed ^ 0x2222222222222222ULL);
    auto gamma1 = sample_gamma(d, C, gamma_rng1);
    auto gamma2 = sample_gamma(d, C, gamma_rng2);
    S3iHPair out{build_random_linear_combination_fused(encodings, gamma1, C, ole, stats),
                  build_random_linear_combination_fused(encodings, gamma2, C, ole, stats)};
    if (stats) {
        const double pcg_delta = stats->pcg_offline_seconds - pcg_s3i_before;
        stats->prof_s3i_build_seconds += prof_elapsed(prof_s3i_t0) - pcg_delta;
    }
    return out;
}

static S3iHPair build_s3i_h_pair(std::span<const std::uint64_t> receiver_share,
                                 std::span<const std::uint64_t> sender_share,
                                 std::size_t d,
                                 std::size_t C,
                                 std::uint64_t ssrole_seed,
                                 std::uint64_t gamma_seed,
                                 OssiStats* stats) {
    PcgOlePool ole(ssrole_seed ^ (d << 32) ^ C, stats);
    return build_s3i_h_pair_with_pool(receiver_share, sender_share, d, C, ole, gamma_seed, stats);
}

#if defined(FPSI_HAVE_GF64X4) && defined(FPSI_USE_GF64X4_GCD)
#endif

static std::size_t ossi_pair_index_from_slot(std::size_t ell) noexcept {
    return (ell + 1) / 2;
}

static std::uint8_t ossi_slot_uses_hi(std::size_t ell) noexcept {
    // Fixed convention for PRF half-packing: slot 2p-1 uses .hi, slot 2p uses .lo.
    return static_cast<std::uint8_t>((ell & 1U) != 0U);
}

static std::uint64_t fingerprint_keysets(std::span<const KeySetPair> pairs) {
    std::uint64_t fp = 0x3141592653589793ULL ^ static_cast<std::uint64_t>(pairs.size());
    for (const auto& p : pairs) {
        fp = splitmix_stateless(fp ^ p.key.lo);
        fp = splitmix_stateless(fp ^ p.key.hi ^ static_cast<std::uint64_t>(p.values.size()));
        for (auto v : p.values) fp = splitmix_stateless(fp ^ v);
    }
    return fp;
}

} // namespace

std::size_t s3i_required_ole_correlations(std::size_t N, std::size_t d, std::size_t C) {
    return s3i_required_ole_correlations_impl(N, d, C);
}

void s3i_precompute_vandermonde(std::size_t C, OssiStats* stats) {
    const auto t0 = std::chrono::steady_clock::now();
    precompute_vandermonde_for_C_impl(C);
    if (stats) stats->vandermonde_precompute_seconds += prof_elapsed(t0);
}

Block128 ossi_slot_key(Block128 key, std::size_t ell) noexcept {
    const std::uint64_t e = static_cast<std::uint64_t>(ell);
    const std::uint64_t a = splitmix_stateless(key.lo ^ 0xa5a5a5a5a5a5a5a5ULL ^ (e * 0x9e3779b97f4a7c15ULL));
    const std::uint64_t b = splitmix_stateless(key.hi ^ 0x3c6ef372fe94f82bULL ^ (e * 0xbf58476d1ce4e5b9ULL));
    return Block128{key.lo ^ a ^ e, key.hi ^ b ^ (e << 32) ^ (e >> 32)};
}

Block128 ossi_pair_key(Block128 key, std::size_t pair_index) noexcept {
    const std::uint64_t p = static_cast<std::uint64_t>(pair_index);
    const std::uint64_t a = splitmix_stateless(key.lo ^ 0x706169725f6b6579ULL ^
                                               (p * 0xd6e8feb86659fd93ULL));
    const std::uint64_t b = splitmix_stateless(key.hi ^ 0x4f5052465f504149ULL ^
                                               (p * 0xa4093822299f31d0ULL));
    return Block128{key.lo ^ a ^ (p << 1), key.hi ^ b ^ (p << 33) ^ (p >> 31)};
}

std::vector<std::uint64_t> s3i_intersect_shared(std::span<const std::uint64_t> receiver_share,
                                                std::span<const std::uint64_t> sender_share,
                                                std::size_t d,
                                                std::size_t C,
                                                std::uint64_t ssrole_seed,
                                                std::uint64_t gamma_seed,
                                                std::uint64_t root_seed,
                                                OssiStats* stats) {
    if (d < 2) throw std::invalid_argument("S3I requires d >= 2");
    if (!is_power_of_two(C)) throw std::invalid_argument("S3I subset tree requires power-of-two C");
    if (receiver_share.size() != d * C || sender_share.size() != d * C) {
        throw std::invalid_argument("S3I share vector size mismatch");
    }
    if (stats) { stats->d = d; stats->C = C; }
    precompute_vandermonde_for_C_impl(C);

    S3iHPair hp = build_s3i_h_pair(receiver_share, sender_share, d, C,
                                   ssrole_seed, gamma_seed, stats);
    Poly g = poly_gcd(std::move(hp.h1), std::move(hp.h2));
    if (poly_deg(g) <= 0) return {};
    return poly_roots_split(std::move(g), root_seed ^ static_cast<std::uint64_t>(poly_deg(g)));
}


bool s3i_intersect_shared_nonempty(std::span<const std::uint64_t> receiver_share,
                                    std::span<const std::uint64_t> sender_share,
                                    std::size_t d,
                                    std::size_t C,
                                    std::uint64_t ssrole_seed,
                                    std::uint64_t gamma_seed,
                                    OssiStats* stats) {
    if (d < 2) throw std::invalid_argument("S3I requires d >= 2");
    if (!is_power_of_two(C)) throw std::invalid_argument("S3I subset tree requires power-of-two C");
    if (receiver_share.size() != d * C || sender_share.size() != d * C) {
        throw std::invalid_argument("S3I share vector size mismatch");
    }
    if (stats) { stats->d = d; stats->C = C; }
    precompute_vandermonde_for_C_impl(C);

    S3iHPair hp = build_s3i_h_pair(receiver_share, sender_share, d, C,
                                   ssrole_seed, gamma_seed, stats);
    Poly g = poly_gcd(std::move(hp.h1), std::move(hp.h2));
    return poly_deg(g) > 0;
}

OssiResult ossi_run_ideal(std::span<const KeySetPair> receiver_pairs,
                          std::span<const Block128> sender_keys,
                          const OssiParams& params) {
    if (sender_keys.size() < 2) throw std::invalid_argument("OSSI requires at least two sender keys");
    if (!is_power_of_two(params.C)) throw std::invalid_argument("OSSI requires power-of-two C");
    if ((params.C & 1U) != 0U) throw std::invalid_argument("OSSI half-packed OPPRF requires even C");
    precompute_vandermonde_for_C_impl(params.C);

    std::vector<Block128> programmed_keys;
    std::vector<Block128> programmed_pair_keys;
    std::vector<std::uint8_t> programmed_slot_hi;
    std::vector<std::uint64_t> programmed_values;
    std::size_t reserve = 0;
    for (const auto& kv : receiver_pairs) {
        if (kv.values.size() > params.C) throw std::invalid_argument("receiver set exceeds C");
        reserve += kv.values.size();
    }
    programmed_keys.reserve(reserve);
    programmed_pair_keys.reserve(reserve);
    programmed_slot_hi.reserve(reserve);
    programmed_values.reserve(reserve);
    for (const auto& kv : receiver_pairs) {
        for (std::size_t ell = 1; ell <= kv.values.size(); ++ell) {
            programmed_keys.push_back(ossi_slot_key(kv.key, ell));
            programmed_pair_keys.push_back(ossi_pair_key(kv.key, ossi_pair_index_from_slot(ell)));
            programmed_slot_hi.push_back(ossi_slot_uses_hi(ell));
            programmed_values.push_back(kv.values[ell - 1]);
        }
    }

    std::vector<Block128> query_keys;
    std::vector<Block128> query_pair_keys;
    std::vector<std::size_t> query_pair_index;
    std::vector<std::uint8_t> query_slot_hi;
    query_keys.reserve(sender_keys.size() * params.C);
    query_pair_keys.reserve(sender_keys.size() * (params.C / 2));
    query_pair_index.reserve(sender_keys.size() * params.C);
    query_slot_hi.reserve(sender_keys.size() * params.C);
    for (const auto& k : sender_keys) {
        for (std::size_t pidx = 1; pidx <= params.C / 2; ++pidx) {
            const std::size_t out_pair_idx = query_pair_keys.size();
            query_pair_keys.push_back(ossi_pair_key(k, pidx));
            for (std::size_t off = 0; off < 2; ++off) {
                const std::size_t ell = 2 * pidx - 1 + off;
                query_keys.push_back(ossi_slot_key(k, ell));
                query_pair_index.push_back(out_pair_idx);
                query_slot_hi.push_back(ossi_slot_uses_hi(ell));
            }
        }
    }

    auto prf_key = sooprf::AltModPrf::random_key(params.prf_seed ^ fingerprint_keysets(receiver_pairs));
    opprf::Opprf64 op(prf_key, params.opprf_params);
    auto tape = sooprf::AltModWPrfCorrelationTape::make_ideal(
        query_pair_keys.size(),
        params.sooprf_tape_seed ^ static_cast<std::uint64_t>(query_pair_keys.size()));
    auto opout = op.eval_altmod_wprf_tape_packed(programmed_keys, programmed_pair_keys,
                                                 programmed_slot_hi, programmed_values,
                                                 query_keys, query_pair_keys,
                                                 query_pair_index, query_slot_hi, tape);

    OssiResult res;
    res.stats.programmed_pairs = programmed_keys.size();
    res.stats.query_pairs = query_pair_keys.size();
    res.stats.okvs_cols = opout.encoding.cols;
    res.stats.C = params.C;
    res.stats.d = sender_keys.size();
    res.intersection = s3i_intersect_shared(opout.receiver_share, opout.sender_share,
                                            sender_keys.size(), params.C,
                                            params.ssrole_seed, params.gamma_seed,
                                            params.root_seed, &res.stats);
    std::sort(res.intersection.begin(), res.intersection.end());
    res.intersection.erase(std::unique(res.intersection.begin(), res.intersection.end()), res.intersection.end());
    return res;
}


OssiBatchBitResult s3i_batch_intersect_shared_nonempty(std::span<const std::uint64_t> receiver_share,
                                                       std::span<const std::uint64_t> sender_share,
                                                       std::size_t N,
                                                       std::size_t d,
                                                       const OssiParams& params) {
    if (N == 0) throw std::invalid_argument("batched S3I requires N > 0");
    if (d < 2) throw std::invalid_argument("batched S3I requires d >= 2");
    if (!is_power_of_two(params.C)) throw std::invalid_argument("S3I requires power-of-two C");
    const std::size_t one = d * params.C;
    if (receiver_share.size() != N * one || sender_share.size() != N * one) {
        throw std::invalid_argument("batched S3I share vectors must contain exactly N*d*C elements");
    }
    precompute_vandermonde_for_C_impl(params.C);

    OssiBatchBitResult res;
    res.N = N;
    res.matched.assign(N, 0);
    res.stats.query_pairs = N * one;
    res.stats.C = params.C;
    res.stats.d = d;

    // In the batched FPSI use case, all N S^3I instances consume one long PCG
    // OLE stream.  Do not instantiate/keygen the PCG pool per query.
    PcgOlePool batch_ole(params.ssrole_seed ^ (d << 32) ^ params.C ^ static_cast<std::uint64_t>(N),
                         &res.stats);
    // Materialize all ideal OLE/PCG correlations before the measured online
    // S^3I loop. This avoids deriving online time by subtracting lazy PCG
    // generation from an enclosing wall clock.
    batch_ole.ensure(s3i_required_ole_correlations_impl(N, d, params.C));

    const auto s3i_online_wall_t0 = std::chrono::steady_clock::now();
    std::size_t n = 0;
    while (n < N) {
#if defined(FPSI_HAVE_GF64X4) && defined(FPSI_USE_GF64X4_GCD)
        if (n + 4 <= N) {
            std::uint64_t gamma_lanes[4];
            for (std::size_t lane = 0; lane < 4; ++lane) {
                const std::size_t q = n + lane;
                const std::uint64_t tweak = splitmix_stateless(static_cast<std::uint64_t>(q));
                gamma_lanes[lane] = params.gamma_seed ^ splitmix_stateless(tweak);
            }
            S3iHPair4 hp4 = build_s3i_h_pair4_with_pool(receiver_share, sender_share, n, one,
                                                        d, params.C, batch_ole,
                                                        gamma_lanes, &res.stats);
            const std::size_t coeffs = 2 * params.C;
            if (can_batch4_gcd_packed(hp4, coeffs)) {
                int deg_gt0[4] = {0, 0, 0, 0};
                const auto prof_gcd_t0 = std::chrono::steady_clock::now();
                batch4_gcd_nonempty_packed(hp4, coeffs, deg_gt0);
                res.stats.prof_gcd_seconds += prof_elapsed(prof_gcd_t0);
                for (std::size_t lane = 0; lane < 4; ++lane) {
                    res.matched[n + lane] = static_cast<std::uint8_t>(deg_gt0[lane] ? 1 : 0);
                }
                n += 4;
                continue;
            }
            std::array<S3iHPair, 4> hp = unpack_hp4(hp4);
            for (std::size_t lane = 0; lane < 4; ++lane) {
                const auto prof_gcd_t0 = std::chrono::steady_clock::now();
                Poly g = poly_gcd(std::move(hp[lane].h1), std::move(hp[lane].h2));
                res.stats.prof_gcd_seconds += prof_elapsed(prof_gcd_t0);
                res.matched[n + lane] = static_cast<std::uint8_t>(poly_deg(g) > 0 ? 1 : 0);
            }
            n += 4;
            continue;
        }
#endif
        const std::uint64_t tweak = splitmix_stateless(static_cast<std::uint64_t>(n));
        S3iHPair hp = build_s3i_h_pair_with_pool(receiver_share.subspan(n * one, one),
                                                 sender_share.subspan(n * one, one),
                                                 d, params.C, batch_ole,
                                                 params.gamma_seed ^ splitmix_stateless(tweak),
                                                 &res.stats);
        const auto prof_gcd_t0 = std::chrono::steady_clock::now();
        Poly g = poly_gcd(std::move(hp.h1), std::move(hp.h2));
        res.stats.prof_gcd_seconds += prof_elapsed(prof_gcd_t0);
        res.matched[n] = static_cast<std::uint8_t>(poly_deg(g) > 0 ? 1 : 0);
        ++n;
    }
    res.stats.prof_s3i_online_wall_seconds += prof_elapsed(s3i_online_wall_t0);
    return res;
}


OssiBatchBitResult ossi_run_ideal_batch_nonempty(std::span<const KeySetPair> receiver_pairs,
                                                 std::span<const Block128> sender_keys_flat,
                                                 std::size_t N,
                                                 std::size_t d,
                                                 const OssiParams& params) {
    if (N == 0) throw std::invalid_argument("batched OSSI requires N > 0");
    if (d < 2) throw std::invalid_argument("batched OSSI requires d >= 2");
    if (!is_power_of_two(params.C)) throw std::invalid_argument("OSSI requires power-of-two C");
    if ((params.C & 1U) != 0U) throw std::invalid_argument("OSSI half-packed OPPRF requires even C");
    if (sender_keys_flat.size() != N * d) {
        throw std::invalid_argument("sender_keys_flat must contain exactly N*d keys");
    }
    precompute_vandermonde_for_C_impl(params.C);

    std::vector<Block128> programmed_keys;
    std::vector<Block128> programmed_pair_keys;
    std::vector<std::uint8_t> programmed_slot_hi;
    std::vector<std::uint64_t> programmed_values;
    std::size_t reserve = 0;
    for (const auto& kv : receiver_pairs) {
        if (kv.values.size() > params.C) throw std::invalid_argument("receiver set exceeds C");
        reserve += kv.values.size();
    }
    programmed_keys.reserve(reserve);
    programmed_pair_keys.reserve(reserve);
    programmed_slot_hi.reserve(reserve);
    programmed_values.reserve(reserve);
    for (const auto& kv : receiver_pairs) {
        for (std::size_t ell = 1; ell <= kv.values.size(); ++ell) {
            programmed_keys.push_back(ossi_slot_key(kv.key, ell));
            programmed_pair_keys.push_back(ossi_pair_key(kv.key, ossi_pair_index_from_slot(ell)));
            programmed_slot_hi.push_back(ossi_slot_uses_hi(ell));
            programmed_values.push_back(kv.values[ell - 1]);
        }
    }

    std::vector<Block128> query_keys;
    std::vector<Block128> query_pair_keys;
    std::vector<std::size_t> query_pair_index;
    std::vector<std::uint8_t> query_slot_hi;
    query_keys.reserve(N * d * params.C);
    query_pair_keys.reserve(N * d * (params.C / 2));
    query_pair_index.reserve(N * d * params.C);
    query_slot_hi.reserve(N * d * params.C);
    for (std::size_t n = 0; n < N; ++n) {
        const std::size_t base = n * d;
        for (std::size_t j = 0; j < d; ++j) {
            const auto& k = sender_keys_flat[base + j];
            for (std::size_t pidx = 1; pidx <= params.C / 2; ++pidx) {
                const std::size_t out_pair_idx = query_pair_keys.size();
                query_pair_keys.push_back(ossi_pair_key(k, pidx));
                for (std::size_t off = 0; off < 2; ++off) {
                    const std::size_t ell = 2 * pidx - 1 + off;
                    query_keys.push_back(ossi_slot_key(k, ell));
                    query_pair_index.push_back(out_pair_idx);
                    query_slot_hi.push_back(ossi_slot_uses_hi(ell));
                }
            }
        }
    }

    auto prf_key = sooprf::AltModPrf::random_key(params.prf_seed ^ fingerprint_keysets(receiver_pairs));
    opprf::Opprf64 op(prf_key, params.opprf_params);
    auto tape = sooprf::AltModWPrfCorrelationTape::make_ideal(
        query_pair_keys.size(),
        params.sooprf_tape_seed ^ static_cast<std::uint64_t>(query_pair_keys.size()));
    auto opout = op.eval_altmod_wprf_tape_packed(programmed_keys, programmed_pair_keys,
                                                 programmed_slot_hi, programmed_values,
                                                 query_keys, query_pair_keys,
                                                 query_pair_index, query_slot_hi, tape);

    OssiBatchBitResult res;
    res.N = N;
    res.matched.assign(N, 0);
    res.stats.programmed_pairs = programmed_keys.size();
    res.stats.query_pairs = query_pair_keys.size();
    res.stats.okvs_cols = opout.encoding.cols;
    res.stats.C = params.C;
    res.stats.d = d;

    const std::size_t one = d * params.C;
    std::span<const std::uint64_t> rspan(opout.receiver_share.data(), opout.receiver_share.size());
    std::span<const std::uint64_t> sspan(opout.sender_share.data(), opout.sender_share.size());

    // In the batched FPSI use case, all N S^3I instances consume one long PCG
    // OLE stream.  Do not instantiate/keygen the PCG pool per query.
    PcgOlePool batch_ole(params.ssrole_seed ^ (d << 32) ^ params.C ^ static_cast<std::uint64_t>(N),
                         &res.stats);
    // Materialize all ideal OLE/PCG correlations before the measured online
    // S^3I loop. This avoids deriving online time by subtracting lazy PCG
    // generation from an enclosing wall clock.
    batch_ole.ensure(s3i_required_ole_correlations_impl(N, d, params.C));

    const auto s3i_online_wall_t0 = std::chrono::steady_clock::now();
    std::size_t n = 0;
    while (n < N) {
#if defined(FPSI_HAVE_GF64X4) && defined(FPSI_USE_GF64X4_GCD)
        if (n + 4 <= N) {
            std::uint64_t gamma_lanes[4];
            for (std::size_t lane = 0; lane < 4; ++lane) {
                const std::size_t q = n + lane;
                const std::uint64_t tweak = splitmix_stateless(static_cast<std::uint64_t>(q));
                gamma_lanes[lane] = params.gamma_seed ^ splitmix_stateless(tweak);
            }
            S3iHPair4 hp4 = build_s3i_h_pair4_with_pool(rspan, sspan, n, one,
                                                        d, params.C, batch_ole,
                                                        gamma_lanes, &res.stats);
            const std::size_t coeffs = 2 * params.C;
            if (can_batch4_gcd_packed(hp4, coeffs)) {
                int deg_gt0[4] = {0, 0, 0, 0};
                const auto prof_gcd_t0 = std::chrono::steady_clock::now();
                batch4_gcd_nonempty_packed(hp4, coeffs, deg_gt0);
                res.stats.prof_gcd_seconds += prof_elapsed(prof_gcd_t0);
                for (std::size_t lane = 0; lane < 4; ++lane) {
                    res.matched[n + lane] = static_cast<std::uint8_t>(deg_gt0[lane] ? 1 : 0);
                }
                n += 4;
                continue;
            }
            // Extremely unlikely unless a leading random coefficient cancels in
            // at least one lane; unpack and fall back to scalar Euclidean gcd.
            std::array<S3iHPair, 4> hp = unpack_hp4(hp4);
            for (std::size_t lane = 0; lane < 4; ++lane) {
                const auto prof_gcd_t0 = std::chrono::steady_clock::now();
                Poly g = poly_gcd(std::move(hp[lane].h1), std::move(hp[lane].h2));
                res.stats.prof_gcd_seconds += prof_elapsed(prof_gcd_t0);
                res.matched[n + lane] = static_cast<std::uint8_t>(poly_deg(g) > 0 ? 1 : 0);
            }
            n += 4;
            continue;
        }
#endif
        const std::uint64_t tweak = splitmix_stateless(static_cast<std::uint64_t>(n));
        S3iHPair hp = build_s3i_h_pair_with_pool(rspan.subspan(n * one, one),
                                                   sspan.subspan(n * one, one),
                                                   d, params.C, batch_ole,
                                                   params.gamma_seed ^ splitmix_stateless(tweak),
                                                   &res.stats);
        const auto prof_gcd_t0 = std::chrono::steady_clock::now();
        Poly g = poly_gcd(std::move(hp.h1), std::move(hp.h2));
        res.stats.prof_gcd_seconds += prof_elapsed(prof_gcd_t0);
        res.matched[n] = static_cast<std::uint8_t>(poly_deg(g) > 0 ? 1 : 0);
        ++n;
    }
    res.stats.prof_s3i_online_wall_seconds += prof_elapsed(s3i_online_wall_t0);
    return res;
}

} // namespace fpsi
