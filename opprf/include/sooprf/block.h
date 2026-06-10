#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <ostream>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace sooprf {

struct alignas(16) Block128 {
    std::uint64_t lo;
    std::uint64_t hi;

    constexpr Block128(std::uint64_t lo_ = 0, std::uint64_t hi_ = 0) noexcept : lo(lo_), hi(hi_) {}

    friend constexpr Block128 operator^(Block128 a, Block128 b) noexcept {
        return Block128{a.lo ^ b.lo, a.hi ^ b.hi};
    }
    friend constexpr Block128 operator&(Block128 a, Block128 b) noexcept {
        return Block128{a.lo & b.lo, a.hi & b.hi};
    }
    friend constexpr Block128 operator|(Block128 a, Block128 b) noexcept {
        return Block128{a.lo | b.lo, a.hi | b.hi};
    }
    friend constexpr Block128 operator~(Block128 a) noexcept {
        return Block128{~a.lo, ~a.hi};
    }
    Block128& operator^=(Block128 b) noexcept { lo ^= b.lo; hi ^= b.hi; return *this; }
    Block128& operator&=(Block128 b) noexcept { lo &= b.lo; hi &= b.hi; return *this; }
    friend constexpr bool operator==(Block128 a, Block128 b) noexcept { return a.lo == b.lo && a.hi == b.hi; }
    friend constexpr bool operator!=(Block128 a, Block128 b) noexcept { return !(a == b); }
};

static_assert(sizeof(Block128) == 16, "Block128 must be 16 bytes");
static_assert(alignof(Block128) == 16, "Block128 must be 16-byte aligned");

inline bool bit(Block128 x, std::size_t i) noexcept {
    return i < 64 ? ((x.lo >> i) & 1u) : ((x.hi >> (i - 64)) & 1u);
}

inline void set_bit(Block128& x, std::size_t i) noexcept {
    if (i < 64) x.lo |= (std::uint64_t{1} << i);
    else x.hi |= (std::uint64_t{1} << (i - 64));
}

inline void clear_bit(Block128& x, std::size_t i) noexcept {
    if (i < 64) x.lo &= ~(std::uint64_t{1} << i);
    else x.hi &= ~(std::uint64_t{1} << (i - 64));
}

inline std::ostream& operator<<(std::ostream& os, Block128 b) {
    auto flags = os.flags();
    os << std::hex << b.hi << b.lo;
    os.flags(flags);
    return os;
}

inline std::array<std::uint64_t, 2> as_gf64_pair(Block128 b) noexcept {
    return {b.lo, b.hi};
}

constexpr Block128 ZeroBlock{0, 0};
constexpr Block128 AllOneBlock{~std::uint64_t{0}, ~std::uint64_t{0}};

} // namespace sooprf
