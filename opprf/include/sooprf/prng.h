#pragma once

#include <cstdint>
#include "sooprf/block.h"

namespace sooprf {

class SplitMix64 {
public:
    explicit SplitMix64(std::uint64_t seed = 0) noexcept : state_(seed) {}

    std::uint64_t next_u64() noexcept {
        std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    Block128 next_block() noexcept {
        return Block128{next_u64(), next_u64()};
    }

private:
    std::uint64_t state_;
};

inline std::uint64_t mix_seed(std::uint64_t a, std::uint64_t b) noexcept {
    SplitMix64 sm(a ^ 0x243f6a8885a308d3ULL);
    return sm.next_u64() ^ b ^ 0x9e3779b97f4a7c15ULL;
}

} // namespace sooprf
