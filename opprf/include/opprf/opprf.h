#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "sooprf/altmod_prf.h"
#include "sooprf/shared_oprf.h"

namespace opprf {

using sooprf::AltModKey;
using sooprf::AltModPrf;
using sooprf::Block128;

struct Opprf64Params {
    // L=1 OKVS over 64-bit XOR values.  w=218 is the high-throughput setting
    // used by the supplied OKVS code for 64-bit-key/128-bit-key workloads.
    std::size_t w = 218;
    double epsilon = 0.05;
    std::size_t cols = 0;  // 0 => okvs_w_cols(n,w,epsilon)
    std::array<std::uint8_t, 16> okvs_seed{};
    bool randomize_free_entries = true;
    std::uint64_t free_entry_seed = 0x4f505052465f4f4bull;
};

struct EncodedOpprf64 {
    Opprf64Params params;
    std::size_t cols = 0;
    std::vector<std::uint64_t> table;
};

struct OpprfEncodeStats {
    double mask_prf_seconds = 0.0;
    double okvs_encode_seconds = 0.0;
};

struct IdealOpprf64Output {
    // sender_share[i] XOR receiver_share[i] reconstructs the programmed value
    // for programmed query keys.  Non-programmed query keys yield pseudorandom
    // correction values as expected from OPPRF.
    std::vector<std::uint64_t> sender_share;
    std::vector<std::uint64_t> receiver_share;
    EncodedOpprf64 encoding;
};

// Half-packed OPPRF support: OKVS remains slot-indexed (64-bit L=1),
// while the shared OPRF is evaluated on pair inputs.  Each 128-bit OPRF share
// block supplies two 64-bit slot masks.  The convention is fixed as:
//   slot 2p-1 -> .hi, slot 2p -> .lo.
// The *_hi selectors are byte-valued booleans: nonzero selects .hi, zero selects .lo.
struct PackedOpprf64Output {
    std::vector<std::uint64_t> sender_share;
    std::vector<std::uint64_t> receiver_share;
    EncodedOpprf64 encoding;
};

// Keyless receiver-side decoder. Holds NO PRF key: the OPPRF receiver only ever
// decodes the public hint and folds in its own SoOPRF share, so it must not be
// able to touch the key. The online receiver path (opprf_online.h) uses this.
class Opprf64Decoder {
public:
    Opprf64Decoder() = default;

    // correction[i] = Decode(enc, query_keys[i]).
    void decode_correction_into(const EncodedOpprf64& enc,
                                std::span<const Block128> query_keys,
                                std::span<std::uint64_t> correction_out) const;

    std::vector<std::uint64_t> decode_correction(const EncodedOpprf64& enc,
                                                 std::span<const Block128> query_keys) const;

    // receiver_share_out[i] = Decode(enc, q_i) ^ receiver_oprf_share[i].lo.
    void receiver_share_into(const EncodedOpprf64& enc,
                             std::span<const Block128> query_keys,
                             std::span<const Block128> receiver_oprf_share,
                             std::span<std::uint64_t> receiver_share_out) const;

    // Half-packed receiver-side share.  pair_index[i] selects the 128-bit
    // OPRF share block and slot_hi[i] chooses .hi (nonzero) or .lo (zero).
    void receiver_share_into_packed(const EncodedOpprf64& enc,
                                    std::span<const Block128> query_slot_keys,
                                    std::span<const Block128> receiver_pair_oprf_share,
                                    std::span<const std::size_t> pair_index,
                                    std::span<const std::uint8_t> slot_hi,
                                    std::span<std::uint64_t> receiver_share_out) const;
};

class Opprf64 {
public:
    explicit Opprf64(AltModPrf prf, Opprf64Params params = {})
        : prf_(std::move(prf)), params_(params) {}

    explicit Opprf64(const AltModKey& key, Opprf64Params params = {})
        : prf_(key), params_(params) {}

    const Opprf64Params& params() const noexcept { return params_; }
    const AltModPrf& prf() const noexcept { return prf_; }

    EncodedOpprf64 encode(std::span<const Block128> keys,
                          std::span<const std::uint64_t> values,
                          OpprfEncodeStats* stats = nullptr) const;

    // Half-packed encode: OKVS keys/values stay per slot, but PRF masks are
    // derived from pair_prf_keys and split into .hi/.lo halves by slot_hi.
    EncodedOpprf64 encode_packed(std::span<const Block128> slot_keys,
                                 std::span<const Block128> pair_prf_keys,
                                 std::span<const std::uint8_t> slot_hi,
                                 std::span<const std::uint64_t> values,
                                 OpprfEncodeStats* stats = nullptr) const;

    void decode_correction_into(const EncodedOpprf64& enc,
                                std::span<const Block128> query_keys,
                                std::span<std::uint64_t> correction_out) const;

    std::vector<std::uint64_t> decode_correction(const EncodedOpprf64& enc,
                                                 std::span<const Block128> query_keys) const;

    // Receiver-side share: correction(key) XOR receiver's SoOPRF share.
    void receiver_share_into(const EncodedOpprf64& enc,
                             std::span<const Block128> query_keys,
                             std::span<const Block128> receiver_oprf_share,
                             std::span<std::uint64_t> receiver_share_out) const;

    void receiver_share_into_packed(const EncodedOpprf64& enc,
                                    std::span<const Block128> query_slot_keys,
                                    std::span<const Block128> receiver_pair_oprf_share,
                                    std::span<const std::size_t> pair_index,
                                    std::span<const std::uint8_t> slot_hi,
                                    std::span<std::uint64_t> receiver_share_out) const;

    // Local ideal-tape harness. It uses the secure-join-style AltMod WPRF tape
    // path for the shared-output OPRF layer, then applies the OKVS correction.
    IdealOpprf64Output eval_ideal(std::span<const Block128> programmed_keys,
                                  std::span<const std::uint64_t> values,
                                  std::span<const Block128> query_keys,
                                  std::uint64_t tape_seed = 0x534f4f50505246ULL) const;

    // OPPRF harness backed by the local AltMod WPRF tape simulator.  This uses
    // the secure-join protocol flow for the shared-output OPRF layer while
    // keeping OKVS encode/decode unchanged.
    IdealOpprf64Output eval_altmod_wprf_tape(std::span<const Block128> programmed_keys,
                                             std::span<const std::uint64_t> values,
                                             std::span<const Block128> query_keys,
                                             sooprf::AltModWPrfCorrelationTape& tape) const;


    // Half-packed OPPRF harness.  query_pair_keys is the SoOPRF input vector,
    // while query_slot_keys remains the OKVS decode vector.  pair_index/slot_hi
    // unpack pair OPRF shares into slot shares.
    PackedOpprf64Output eval_altmod_wprf_tape_packed(
        std::span<const Block128> programmed_slot_keys,
        std::span<const Block128> programmed_pair_keys,
        std::span<const std::uint8_t> programmed_slot_hi,
        std::span<const std::uint64_t> values,
        std::span<const Block128> query_slot_keys,
        std::span<const Block128> query_pair_keys,
        std::span<const std::size_t> query_pair_index,
        std::span<const std::uint8_t> query_slot_hi,
        sooprf::AltModWPrfCorrelationTape& tape,
        OpprfEncodeStats* stats = nullptr) const;

private:
    AltModPrf prf_;
    Opprf64Params params_;
};

} // namespace opprf
