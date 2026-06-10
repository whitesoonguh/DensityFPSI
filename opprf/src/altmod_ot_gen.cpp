#include "sooprf/altmod_ot_gen.h"

#include <chrono>
#include <cstring>
#include <vector>

#include "sooprf/prng.h"

// =============================================================================
// libOTe-backed silent correlation generation.
//
// The actual libOTe API surface (headers, configure() arguments, silentSend /
// silentReceive signatures, the coproto socket type) churns between releases.
// The block below is written against a recent osu-crypto/libOTe master. Treat
// every line inside `#ifdef SOOPRF_WITH_LIBOTE` as "verify against YOUR pinned
// libOTe commit and adjust" — it has NOT been compiled in this environment.
//
// Build wiring: see the CMake snippet shipped alongside this file. You must
// define SOOPRF_WITH_LIBOTE and link the libOTe / cryptoTools / coproto targets
// for this path to be taken; otherwise make_silent_ot() falls back to the PRG
// tape so the project still builds.
// =============================================================================
#ifdef SOOPRF_WITH_LIBOTE
  #include "libOTe/TwoChooseOne/Silent/SilentOtExtSender.h"
  #include "libOTe/TwoChooseOne/Silent/SilentOtExtReceiver.h"
  #include "cryptoTools/Common/Defines.h"
  #include "cryptoTools/Crypto/PRNG.h"
  #include "coproto/Socket/LocalAsyncSock.h"
#endif

namespace sooprf {
namespace {

static Block128 valid_f3_lsb(Block128 a, Block128 b) noexcept { return a & ~b; }
static Block128 valid_f3_msb(Block128 a, Block128 b) noexcept { return b & ~a; }

#ifdef SOOPRF_WITH_LIBOTE
// Run libOTe silent (1-out-of-2) random OT in a single process and return
// `n_blocks` genuine, real-OT-derived 128-bit blocks (the sender's correlated
// messages). One silent-OT instance yields two messages; we run ceil(n/2)
// instances and flatten. MultType::Tungsten matches secure-join's choice and
// avoids the deprecated Silver code.
//
// NOTE: this is the part to verify against your libOTe version.
static std::vector<Block128> gen_silent_blocks(std::size_t n_blocks,
                                               std::uint64_t seed,
                                               std::size_t* instances_out) {
    using namespace oc;
    static_assert(sizeof(oc::block) == sizeof(Block128), "block size mismatch");

    const std::size_t n_ot = (n_blocks + 1) / 2;
    if (instances_out) *instances_out = n_ot;

    auto socks = coproto::LocalAsyncSocket::makePair();
    PRNG prngS(toBlock(seed, 0x5359ull));
    PRNG prngR(toBlock(seed, 0x5243ull));

    SilentOtExtSender   sender;
    SilentOtExtReceiver recver;
    sender.mMultType = MultType::Tungsten;   // verify enum spelling for your version
    recver.mMultType = MultType::Tungsten;

    std::vector<std::array<block, 2>> sMsg(n_ot);
    std::vector<block>                rMsg(n_ot);
    BitVector                          choice(n_ot);

    // Random-OT mode (random messages, random choice). silentSend/silentReceive
    // overloads + the OTType/ChoiceBitPacking flags vary across versions.
    auto tS = sender.silentSend(sMsg, prngS, socks[0]);
    auto tR = recver.silentReceive(choice, rMsg, prngR, socks[1]);
    coproto::sync_wait(coproto::when_all_ready(std::move(tS), std::move(tR)));

    std::vector<Block128> out(n_blocks);
    for (std::size_t i = 0; i < n_blocks; ++i) {
        const block b = sMsg[i / 2][i & 1];
        std::memcpy(&out[i], &b, sizeof(Block128));
    }
    return out;
}
#endif // SOOPRF_WITH_LIBOTE

} // namespace

AltModWPrfCorrelationTape make_silent_ot(std::size_t input_count,
                                         std::uint64_t seed,
                                         SilentTapeCost* cost) {
    const std::size_t cols       = (input_count + 127) / 128;
    const std::size_t key_blocks = AltModPrfParams::KeyBits * cols; // KeyBits = 512
    const std::size_t f4_blocks  = AltModPrfParams::MidBits * cols; // MidBits = 256

    // Faithful binary-correlation counts consumed by the AltMod WPRF
    // (secure-join: "each mod2 / 1-out-of-3 OT consumes 2 binary OLEs",
    //  AltModWPrf.cpp ~line 581; key-mult uses an analogous correction).
    // These are reported for accounting; see the note in the header about why
    // exact F4-VOLE packing (which is tighter) is not reproduced here.
    const std::size_t f4_corr  = 2 * f4_blocks * 128;   // = 2 * MidBits * cols * 128
    const std::size_t key_corr = 2 * key_blocks * 128;

    AltModWPrfCorrelationTape tape;
    tape.input_count = input_count;
    tape.cols        = cols;
    tape.key_mult_mask_msb.resize(key_blocks);
    tape.key_mult_mask_lsb.resize(key_blocks);
    tape.f4_sender_ots.resize(f4_blocks);

    // We need this many genuine 128-bit blocks to fill the pads directly:
    //   key part: 2 per key_block (a, b)  ->  2 * key_blocks
    //   f4  part: 4 per f4_block          ->  4 * f4_blocks
    const std::size_t pad_blocks = 2 * key_blocks + 4 * f4_blocks;

    double gen_seconds = 0.0;
    std::size_t instances = 0;

#ifdef SOOPRF_WITH_LIBOTE
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<Block128> mat = gen_silent_blocks(pad_blocks, seed, &instances);
    const auto t1 = std::chrono::steady_clock::now();
    gen_seconds = std::chrono::duration<double>(t1 - t0).count();
#else
    // No libOTe linked: fall back to a PRG so the project still builds/runs.
    // (Identical SHAPE to the real path; only the randomness source differs.)
    SplitMix64 rng(seed ^ (static_cast<std::uint64_t>(input_count) * 0x9e3779b97f4a7c15ULL));
    std::vector<Block128> mat(pad_blocks);
    for (auto& b : mat) b = rng.next_block();
#endif

    // ---- fill the tape from the (genuine, when libOTe is linked) material ----
    std::size_t p = 0;
    for (std::size_t i = 0; i < key_blocks; ++i) {
        const Block128 a = mat[p++];
        const Block128 b = mat[p++];
        tape.key_mult_mask_lsb[i] = valid_f3_lsb(a, b);
        tape.key_mult_mask_msb[i] = valid_f3_msb(a, b);
    }
    for (auto& row : tape.f4_sender_ots) {
        row[0] = mat[p++];
        row[1] = mat[p++];
        row[2] = mat[p++];
        row[3] = mat[p++];
    }

    if (cost) {
        cost->silent_gen_seconds   = gen_seconds;
        cost->f4_correlations      = f4_corr;
        cost->key_correlations     = key_corr;
        cost->silent_ot_instances  = instances;
    }
    return tape;
}

} // namespace sooprf