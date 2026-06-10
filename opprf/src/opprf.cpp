#include "opprf/opprf.h"

extern "C" {
#include "okvs_encode.h"
}

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>

namespace opprf {
namespace {

static_assert(sizeof(Block128) == 16);

const std::uint8_t* key_bytes(std::span<const Block128> keys) noexcept {
    return reinterpret_cast<const std::uint8_t*>(keys.data());
}

inline std::uint64_t select_half(Block128 b, bool hi) noexcept {
    return hi ? b.hi : b.lo;
}

void validate_packed_slot_vectors(const char* where,
                                  std::span<const Block128> slot_keys,
                                  std::span<const Block128> pair_keys,
                                  std::span<const std::uint8_t> slot_hi,
                                  std::span<const std::uint64_t> values) {
    if (slot_keys.size() != values.size() ||
        pair_keys.size() != values.size() ||
        slot_hi.size() != values.size()) {
        throw std::invalid_argument(std::string(where) + ": size mismatch");
    }
}

} // namespace

EncodedOpprf64 Opprf64::encode(std::span<const Block128> keys,
                               std::span<const std::uint64_t> values,
                               OpprfEncodeStats* stats) const {
    if (keys.size() != values.size()) {
        throw std::invalid_argument("Opprf64::encode: key/value size mismatch");
    }
    if (params_.w == 0) {
        throw std::invalid_argument("Opprf64::encode: band width must be nonzero");
    }

    const std::size_t n = keys.size();
    EncodedOpprf64 enc;
    enc.params = params_;
    enc.cols = params_.cols ? params_.cols : okvs_w_cols(n, params_.w, params_.epsilon);
    enc.params.cols = enc.cols;
    enc.table.assign(enc.cols, 0);

    using clock_type = std::chrono::steady_clock;
    const auto mask0 = clock_type::now();
    std::vector<Block128> prf_out(n);
    prf_.eval_batch(keys, prf_out);
    std::vector<std::uint64_t> masked(n);
    for (std::size_t i = 0; i < n; ++i) {
        masked[i] = values[i] ^ prf_out[i].lo;
    }
    const auto mask1 = clock_type::now();

    std::uint64_t free_seed = params_.free_entry_seed;
    std::uint64_t* free_seed_ptr = params_.randomize_free_entries ? &free_seed : nullptr;
    const auto okvs0 = clock_type::now();
    const int rc = okvs_w_encode(key_bytes(keys), sizeof(Block128), masked.data(), n,
                                 params_.w, 1, enc.cols,
                                 params_.okvs_seed.data(), params_.okvs_seed.size(),
                                 enc.table.data(), free_seed_ptr);
    const auto okvs1 = clock_type::now();
    if (stats) {
        stats->mask_prf_seconds = std::chrono::duration<double>(mask1 - mask0).count();
        stats->okvs_encode_seconds = std::chrono::duration<double>(okvs1 - okvs0).count();
    }
    if (rc != 0) {
        throw std::runtime_error("Opprf64::encode: OKVS encode failed; retry with a different seed/cols");
    }
    return enc;
}

EncodedOpprf64 Opprf64::encode_packed(std::span<const Block128> slot_keys,
                                      std::span<const Block128> pair_prf_keys,
                                      std::span<const std::uint8_t> slot_hi,
                                      std::span<const std::uint64_t> values,
                                      OpprfEncodeStats* stats) const {
    validate_packed_slot_vectors("Opprf64::encode_packed", slot_keys, pair_prf_keys, slot_hi, values);
    if (params_.w == 0) {
        throw std::invalid_argument("Opprf64::encode_packed: band width must be nonzero");
    }

    const std::size_t n = slot_keys.size();
    EncodedOpprf64 enc;
    enc.params = params_;
    enc.cols = params_.cols ? params_.cols : okvs_w_cols(n, params_.w, params_.epsilon);
    enc.params.cols = enc.cols;
    enc.table.assign(enc.cols, 0);

    using clock_type = std::chrono::steady_clock;
    const auto mask0 = clock_type::now();
    std::vector<Block128> prf_out(n);
    prf_.eval_batch(pair_prf_keys, prf_out);
    std::vector<std::uint64_t> masked(n);
    for (std::size_t i = 0; i < n; ++i) {
        masked[i] = values[i] ^ select_half(prf_out[i], slot_hi[i] != 0);
    }
    const auto mask1 = clock_type::now();

    std::uint64_t free_seed = params_.free_entry_seed;
    std::uint64_t* free_seed_ptr = params_.randomize_free_entries ? &free_seed : nullptr;
    const auto okvs0 = clock_type::now();
    const int rc = okvs_w_encode(key_bytes(slot_keys), sizeof(Block128), masked.data(), n,
                                 params_.w, 1, enc.cols,
                                 params_.okvs_seed.data(), params_.okvs_seed.size(),
                                 enc.table.data(), free_seed_ptr);
    const auto okvs1 = clock_type::now();
    if (stats) {
        stats->mask_prf_seconds = std::chrono::duration<double>(mask1 - mask0).count();
        stats->okvs_encode_seconds = std::chrono::duration<double>(okvs1 - okvs0).count();
    }
    if (rc != 0) {
        throw std::runtime_error("Opprf64::encode_packed: OKVS encode failed; retry with a different seed/cols");
    }
    return enc;
}

void Opprf64::decode_correction_into(const EncodedOpprf64& enc,
                                     std::span<const Block128> query_keys,
                                     std::span<std::uint64_t> correction_out) const {
    if (correction_out.size() != query_keys.size()) {
        throw std::invalid_argument("Opprf64::decode_correction_into: output size mismatch");
    }
    okvs_w_ctx ctx;
    okvs_w_ctx_init(&ctx, enc.params.w, 1, enc.cols,
                    enc.params.okvs_seed.data(), enc.params.okvs_seed.size());
    okvs_w_decode_batch_ctx(&ctx, key_bytes(query_keys), sizeof(Block128),
                            enc.table.data(), query_keys.size(), correction_out.data());
}

std::vector<std::uint64_t> Opprf64::decode_correction(const EncodedOpprf64& enc,
                                                      std::span<const Block128> query_keys) const {
    std::vector<std::uint64_t> out(query_keys.size());
    decode_correction_into(enc, query_keys, out);
    return out;
}

void Opprf64::receiver_share_into(const EncodedOpprf64& enc,
                                  std::span<const Block128> query_keys,
                                  std::span<const Block128> receiver_oprf_share,
                                  std::span<std::uint64_t> receiver_share_out) const {
    if (receiver_oprf_share.size() != query_keys.size() ||
        receiver_share_out.size() != query_keys.size()) {
        throw std::invalid_argument("Opprf64::receiver_share_into: size mismatch");
    }
    decode_correction_into(enc, query_keys, receiver_share_out);
    for (std::size_t i = 0; i < query_keys.size(); ++i) {
        receiver_share_out[i] ^= receiver_oprf_share[i].lo;
    }
}

void Opprf64::receiver_share_into_packed(const EncodedOpprf64& enc,
                                         std::span<const Block128> query_slot_keys,
                                         std::span<const Block128> receiver_pair_oprf_share,
                                         std::span<const std::size_t> pair_index,
                                         std::span<const std::uint8_t> slot_hi,
                                         std::span<std::uint64_t> receiver_share_out) const {
    if (receiver_share_out.size() != query_slot_keys.size() ||
        pair_index.size() != query_slot_keys.size() ||
        slot_hi.size() != query_slot_keys.size()) {
        throw std::invalid_argument("Opprf64::receiver_share_into_packed: slot size mismatch");
    }
    decode_correction_into(enc, query_slot_keys, receiver_share_out);
    for (std::size_t i = 0; i < query_slot_keys.size(); ++i) {
        const std::size_t p = pair_index[i];
        if (p >= receiver_pair_oprf_share.size()) {
            throw std::invalid_argument("Opprf64::receiver_share_into_packed: pair index out of range");
        }
        receiver_share_out[i] ^= select_half(receiver_pair_oprf_share[p], slot_hi[i] != 0);
    }
}

IdealOpprf64Output Opprf64::eval_ideal(std::span<const Block128> programmed_keys,
                                       std::span<const std::uint64_t> values,
                                       std::span<const Block128> query_keys,
                                       std::uint64_t tape_seed) const {
    auto tape = sooprf::AltModWPrfCorrelationTape::make_ideal(query_keys.size(), tape_seed);
    return eval_altmod_wprf_tape(programmed_keys, values, query_keys, tape);
}


IdealOpprf64Output Opprf64::eval_altmod_wprf_tape(std::span<const Block128> programmed_keys,
                                                  std::span<const std::uint64_t> values,
                                                  std::span<const Block128> query_keys,
                                                  sooprf::AltModWPrfCorrelationTape& tape) const {
    IdealOpprf64Output out;
    out.encoding = encode(programmed_keys, values);

    sooprf::SharedAltModOprf soprf(prf_);
    auto shares = soprf.eval_altmod_wprf_tape(query_keys, tape);

    out.sender_share.resize(query_keys.size());
    for (std::size_t i = 0; i < query_keys.size(); ++i) {
        out.sender_share[i] = shares.sender_share[i].lo;
    }

    out.receiver_share.resize(query_keys.size());
    receiver_share_into(out.encoding, query_keys, shares.receiver_share, out.receiver_share);
    return out;
}

PackedOpprf64Output Opprf64::eval_altmod_wprf_tape_packed(
        std::span<const Block128> programmed_slot_keys,
        std::span<const Block128> programmed_pair_keys,
        std::span<const std::uint8_t> programmed_slot_hi,
        std::span<const std::uint64_t> values,
        std::span<const Block128> query_slot_keys,
        std::span<const Block128> query_pair_keys,
        std::span<const std::size_t> query_pair_index,
        std::span<const std::uint8_t> query_slot_hi,
        sooprf::AltModWPrfCorrelationTape& tape,
        OpprfEncodeStats* stats) const {
    if (query_pair_index.size() != query_slot_keys.size() ||
        query_slot_hi.size() != query_slot_keys.size()) {
        throw std::invalid_argument("Opprf64::eval_altmod_wprf_tape_packed: query size mismatch");
    }

    PackedOpprf64Output out;
    out.encoding = encode_packed(programmed_slot_keys, programmed_pair_keys,
                                 programmed_slot_hi, values, stats);

    sooprf::SharedAltModOprf soprf(prf_);
    auto shares = soprf.eval_altmod_wprf_tape(query_pair_keys, tape);

    out.sender_share.resize(query_slot_keys.size());
    for (std::size_t i = 0; i < query_slot_keys.size(); ++i) {
        const std::size_t p = query_pair_index[i];
        if (p >= shares.sender_share.size()) {
            throw std::invalid_argument("Opprf64::eval_altmod_wprf_tape_packed: pair index out of range");
        }
        out.sender_share[i] = select_half(shares.sender_share[p], query_slot_hi[i] != 0);
    }

    out.receiver_share.resize(query_slot_keys.size());
    receiver_share_into_packed(out.encoding, query_slot_keys, shares.receiver_share,
                               query_pair_index, query_slot_hi, out.receiver_share);
    return out;
}

// ---- keyless receiver-side decoder (no PRF key) -------------------------------

void Opprf64Decoder::decode_correction_into(const EncodedOpprf64& enc,
                                            std::span<const Block128> query_keys,
                                            std::span<std::uint64_t> correction_out) const {
    if (correction_out.size() != query_keys.size()) {
        throw std::invalid_argument("Opprf64Decoder::decode_correction_into: output size mismatch");
    }
    okvs_w_ctx ctx;
    okvs_w_ctx_init(&ctx, enc.params.w, 1, enc.cols,
                    enc.params.okvs_seed.data(), enc.params.okvs_seed.size());
    okvs_w_decode_batch_ctx(&ctx, key_bytes(query_keys), sizeof(Block128),
                            enc.table.data(), query_keys.size(), correction_out.data());
}

std::vector<std::uint64_t> Opprf64Decoder::decode_correction(const EncodedOpprf64& enc,
                                                             std::span<const Block128> query_keys) const {
    std::vector<std::uint64_t> out(query_keys.size());
    decode_correction_into(enc, query_keys, out);
    return out;
}

void Opprf64Decoder::receiver_share_into(const EncodedOpprf64& enc,
                                         std::span<const Block128> query_keys,
                                         std::span<const Block128> receiver_oprf_share,
                                         std::span<std::uint64_t> receiver_share_out) const {
    if (receiver_oprf_share.size() != query_keys.size() ||
        receiver_share_out.size() != query_keys.size()) {
        throw std::invalid_argument("Opprf64Decoder::receiver_share_into: size mismatch");
    }
    decode_correction_into(enc, query_keys, receiver_share_out);
    for (std::size_t i = 0; i < query_keys.size(); ++i) {
        receiver_share_out[i] ^= receiver_oprf_share[i].lo;
    }
}

void Opprf64Decoder::receiver_share_into_packed(const EncodedOpprf64& enc,
                                                std::span<const Block128> query_slot_keys,
                                                std::span<const Block128> receiver_pair_oprf_share,
                                                std::span<const std::size_t> pair_index,
                                                std::span<const std::uint8_t> slot_hi,
                                                std::span<std::uint64_t> receiver_share_out) const {
    if (receiver_share_out.size() != query_slot_keys.size() ||
        pair_index.size() != query_slot_keys.size() ||
        slot_hi.size() != query_slot_keys.size()) {
        throw std::invalid_argument("Opprf64Decoder::receiver_share_into_packed: slot size mismatch");
    }
    decode_correction_into(enc, query_slot_keys, receiver_share_out);
    for (std::size_t i = 0; i < query_slot_keys.size(); ++i) {
        const std::size_t p = pair_index[i];
        if (p >= receiver_pair_oprf_share.size()) {
            throw std::invalid_argument("Opprf64Decoder::receiver_share_into_packed: pair index out of range");
        }
        receiver_share_out[i] ^= select_half(receiver_pair_oprf_share[p], slot_hi[i] != 0);
    }
}

} // namespace opprf
