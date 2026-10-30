#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <limits>
#include <cstdlib>
#include <torch/extension.h>

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>

namespace {

using cute::Int;
using cute::Underscore;
using cute::_;


// The staged codebook costs 8 bytes per entry.  SM86 allows 99 KB of dynamic
// shared memory per block; stay below a conservative cap so an unusually wide
// index width falls back to the global-gather path instead of failing to launch.
// Split-K controls. The target CTA count is what we aim to reach before giving
// up on more parallelism.
inline int splitk_max_from_env() {
  const char* raw = std::getenv("GR00T_CODEBOOK_CUTLASS_SPLITK_MAX");
  return raw != nullptr ? atoi(raw) : 8;
}
static thread_local int kSplitKMax = splitk_max_from_env();
static thread_local int kRequestedCTAs = 384;

// 168 was two waves on the 84-SM A6000 this was tuned on. A 4090 has 128 SMs, so
// a fixed 168 leaves a third of that device idle; ask the device instead. Queried
// once, lazily, because a static initialiser would run before there is a context.
// The wide fetch is bit-for-bit the same index, so this is a pure instruction
// count change; gated only so the two can be measured against each other.
inline bool wide_fetch_from_env() {
  const char* raw = std::getenv("GR00T_CODEBOOK_CUTLASS_WIDE_FETCH");
  return raw == nullptr || raw[0] != '0';
}
static const bool kWideFetchHost = wide_fetch_from_env();

inline bool split_big_m_from_env() {
  const char* raw = std::getenv("GR00T_CODEBOOK_CUTLASS_SPLIT_BIG_M");
  return raw != nullptr && raw[0] == '1';
}
static const bool kSplitBigMHost = split_big_m_from_env();

inline int sm_count_now() {
  static int cached = -1;
  if (cached < 0) {
    int sms = 84;
    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess) {
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    }
    cached = sms;
  }
  return cached;
}

inline int target_ctas_now() { return kRequestedCTAs; }

// M=41 only fills 41 of a 64-row A tile, and the A tile is the larger half of the
// shared-memory budget (BM*64*2 bytes per stage, two stages). A 48-row tile still
// covers M=41, costs 12 KB instead of 16 KB, and so lets another CTA become
// resident per SM.
inline int smem_pad_kb() {
  const char* raw = std::getenv("GR00T_CODEBOOK_CUTLASS_SMEM_PAD_KB");
  return raw != nullptr ? atoi(raw) : 0;
}
static const int kSmemPadKB = smem_pad_kb();

inline bool bm48_from_env() {
  const char* raw = std::getenv("GR00T_CODEBOOK_CUTLASS_BM48");
  return raw != nullptr && raw[0] == '1';
}
static thread_local bool kUseBM48 = bm48_from_env();

inline bool splitk_enabled_from_env() {
  const char* raw = std::getenv("GR00T_CODEBOOK_CUTLASS_SPLITK");
  return raw != nullptr && raw[0] == '1';
}
static const bool kSplitKEnabled = splitk_enabled_from_env();

constexpr int kSmemCodebookMaxBytes = 32 * 1024;
constexpr int kSmemBudgetBytes = 96 * 1024;

__host__ __forceinline__ int codebook_entry_count(int static_bits,
                                                  int index_bits) {
  const int bits = static_bits == 0 ? index_bits : static_bits;
  if (bits <= 0 || bits > 20) {
    return 0;
  }
  return 1 << bits;
}

constexpr int kBlockK = 64;
constexpr int kStages = 2;

template <int StaticBits>
__device__ __forceinline__ uint32_t decode_index(
    const uint8_t* packed,
    int linear_index,
    int index_bits,
    int window_bytes) {
  const int bits = StaticBits == 0 ? index_bits : StaticBits;
  const int bit_position = linear_index * bits;
  const int byte_id = bit_position >> 3;
  const int bit_offset = bit_position & 7;
  uint32_t word = static_cast<uint32_t>(packed[byte_id]);
  if ((StaticBits == 0 && window_bytes >= 2) ||
      (StaticBits != 0 && StaticBits != 4 && StaticBits != 8)) {
    word |= static_cast<uint32_t>(packed[byte_id + 1]) << 8;
  }
  if ((StaticBits == 0 && window_bytes >= 3) || StaticBits >= 11) {
    word |= static_cast<uint32_t>(packed[byte_id + 2]) << 16;
  }
  if (StaticBits == 0 && window_bytes >= 4) {
    word |= static_cast<uint32_t>(packed[byte_id + 3]) << 24;
  }
  return (word >> bit_offset) & ((uint32_t{1} << bits) - 1);
}

// A group contains exactly 16 d4 vector indices, so a static-b-bit group
// occupies exactly 2*b bytes.  Load each packed byte once per 16-lane subgroup
// and broadcast the byte(s) needed by each index.  The packed representation
// and reconstructed values are unchanged.
template <int StaticBits>
__device__ __forceinline__ uint32_t decode_index_cooperative_wide(
    const uint8_t* packed,
    int linear_index,
    int packed_bytes) {
  // Same answer as decode_index_cooperative, fetched with a quarter of the
  // memory instructions. That version has each of the first 2*StaticBits lanes
  // issue a one-byte LDG -- ten of them for a 5-bit group -- to bring in ten
  // bytes. Ablation mode 9 (index taken from the loop counter, so the stream is
  // never read) put that fetch at roughly 40% of the kernel's real GPU work, and
  // it is instruction-bound rather than bandwidth-bound: ten bytes is a quarter
  // of one sector.
  //
  // A group's bytes always lie inside four aligned 32-bit words, so four lanes
  // can fetch the whole group. The group is not itself 4-byte aligned, hence the
  // `off` correction: byte j of the group is byte off+j from the aligned base.
  constexpr int kGroupVectors = 16;
  constexpr int kGroupBytes = 2 * StaticBits;
  constexpr int kWords = (3 + kGroupBytes + 3) / 4;
  const int subgroup_lane = threadIdx.x & (kGroupVectors - 1);
  const int group_id = linear_index >> 4;
  const int local_index = linear_index & (kGroupVectors - 1);
  const int group_byte = group_id * kGroupBytes;
  const int base = group_byte & ~3;
  const int off = group_byte & 3;
  const int byte_idx = base + 4 * subgroup_lane;
  // The final group can leave the last padding word past the end of the buffer.
  // Every byte the group actually needs is in range, so zeroing that word is
  // safe rather than merely tolerable.
  const uint32_t word =
      (subgroup_lane < kWords && byte_idx + 3 < packed_bytes)
          ? *reinterpret_cast<const uint32_t*>(packed + byte_idx)
          : 0u;
  const int local_bit = local_index * StaticBits;
  const int bit_offset = local_bit & 7;
  const int abs0 = off + (local_bit >> 3);
  const uint32_t w0 =
      __shfl_sync(0xffffffffu, word, abs0 >> 2, kGroupVectors);
  uint32_t value = (w0 >> ((abs0 & 3) * 8)) & 0xFFu;
  if constexpr (StaticBits != 4) {
    const int abs1 = abs0 + 1;
    const uint32_t w1 =
        __shfl_sync(0xffffffffu, word, abs1 >> 2, kGroupVectors);
    value |= ((w1 >> ((abs1 & 3) * 8)) & 0xFFu) << 8;
  }
  return (value >> bit_offset) & ((uint32_t{1} << StaticBits) - 1);
}

template <int StaticBits>
__device__ __forceinline__ uint32_t decode_index_cooperative(
    const uint8_t* packed,
    int linear_index) {
  static_assert(StaticBits == 4 || StaticBits == 5 || StaticBits == 6 || StaticBits == 7);
  constexpr int kGroupVectors = 16;
  constexpr int kGroupBytes = 2 * StaticBits;
  const int subgroup_lane = threadIdx.x & (kGroupVectors - 1);
  const int group_id = linear_index >> 4;
  const int local_index = linear_index & (kGroupVectors - 1);
  const int group_byte = group_id * kGroupBytes;
  const uint32_t loaded_byte = subgroup_lane < kGroupBytes
      ? static_cast<uint32_t>(packed[group_byte + subgroup_lane])
      : 0u;
  const int local_bit = local_index * StaticBits;
  const int byte_in_group = local_bit >> 3;
  const int bit_offset = local_bit & 7;
  const uint32_t byte0 = __shfl_sync(
      0xffffffffu, loaded_byte, byte_in_group, kGroupVectors);
  uint32_t word = byte0;
  if constexpr (StaticBits != 4) {
    // The last index can end exactly at the group boundary.  Clamp the unused
    // high-byte source so the shuffle always names a valid loader lane.
    const int byte1_source = min(byte_in_group + 1, kGroupBytes - 1);
    const uint32_t byte1 = __shfl_sync(
        0xffffffffu, loaded_byte, byte1_source, kGroupVectors);
    word |= byte1 << 8;
  }
  return (word >> bit_offset) & ((uint32_t{1} << StaticBits) - 1);
}

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
  cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> a;
  cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> b;
};

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct WarpSpecializedSharedStorage {
  cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> a;
  cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> b;
  int b_ready[kStages];
  int producer_done[kStages];
};

// Barrier 0 is reserved by __syncthreads().  The warp-specialized kernel has
// exactly eight 32-thread MMA warps, so barrier 1 can synchronize consumers
// without waiting for the two producer warps.
__device__ __forceinline__ void consumer_barrier_256() {
  asm volatile("bar.sync 1, 256;" ::: "memory");
}

__device__ __forceinline__ void producer_barrier_256() {
  asm volatile("bar.sync 2, 256;" ::: "memory");
}

template <
    int BlockM,
    int BlockN,
    int StaticBits,
    bool CooperativeDecode = false,
    bool VectorSharedStore = false,
    bool SmemCodebook = false>
__global__ void codebook_cute_sm80_kernel(
    const cute::bfloat16_t* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const __nv_bfloat16* __restrict__ codebook,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ shift,
    const __nv_bfloat16* __restrict__ bias,
    cute::bfloat16_t* __restrict__ output,
    int M,
    int N,
    int K,
    int num_groups,
    int index_bits,
    int window_bytes,
    bool has_bias,
    int codebook_entries,
    float* __restrict__ partials,
    int k_split,
    int packed_bytes,
    bool wide_fetch) {
  using namespace cute;

  constexpr int kWarpM = BlockM / 16;
  constexpr int kWarpN = BlockN / 16;
  constexpr int kThreads = kWarpM * kWarpN * 32;
  static_assert(kThreads == BlockM * BlockN / 8);

  auto bM = Int<BlockM>{};
  auto bN = Int<BlockN>{};
  auto bK = Int<kBlockK>{};
  auto bP = Int<kStages>{};

  // CUTLASS/CuTe's Ampere swizzle eliminates ldmatrix bank conflicts.
  auto swizzle_atom = composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{});
  auto sA_layout = tile_to_shape(swizzle_atom, make_shape(bM, bK, bP));
  auto sB_layout = tile_to_shape(swizzle_atom, make_shape(bN, bK, bP));

  using SmemStorage = SharedStorage<
      cute::bfloat16_t,
      cute::bfloat16_t,
      decltype(sA_layout),
      decltype(sB_layout)>;
  extern __shared__ char shared_memory[];
  SmemStorage& smem = *reinterpret_cast<SmemStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.a.begin()), sA_layout);
  Tensor sB = make_tensor(make_smem_ptr(smem.b.begin()), sB_layout);

  // A d4 codeword is exactly eight bytes, and a deployed codebook holds at most
  // 2^index_bits of them, so the whole table is a few kilobytes.  The baseline
  // gathered it from global memory once per reconstructed vector, which is a
  // fully uncoalesced LDG per lane.  Stage it once per CTA instead: the load
  // below is perfectly coalesced, and the per-vector lookup afterwards becomes
  // one LDS.64 into a table that stays resident for the whole K loop.
  uint64_t* smem_codebook = nullptr;
  if constexpr (SmemCodebook) {
    constexpr size_t kAlignedStorage =
        (sizeof(SmemStorage) + alignof(uint64_t) - 1) / alignof(uint64_t) *
        alignof(uint64_t);
    smem_codebook =
        reinterpret_cast<uint64_t*>(shared_memory + kAlignedStorage);
    const uint64_t* codebook_words =
        reinterpret_cast<const uint64_t*>(codebook);
    for (int entry = threadIdx.x; entry < codebook_entries;
         entry += kThreads) {
      smem_codebook[entry] = codebook_words[entry];
    }
    __syncthreads();
  }

  const int m0 = blockIdx.y * BlockM;
  const int n0 = blockIdx.x * BlockN;
  const int total_vectors = K / 4;
  const int total_k_tiles = K / kBlockK;
  // With k_split == 1 this is the original whole-K range, so the validated path
  // is bit-for-bit unchanged. Splitting K is the only way to add CTAs without
  // shrinking them: the deployed contraction key launches 48 CTAs of 256 threads
  // on an 84-SM device, i.e. 9.5% of the machine's thread slots.
  const int tiles_per_split = (total_k_tiles + k_split - 1) / k_split;
  const int k_begin = static_cast<int>(blockIdx.z) * tiles_per_split;
  const int k_end = min(total_k_tiles, k_begin + tiles_per_split);
  const int my_k_tiles = max(0, k_end - k_begin);
  const __nv_bfloat16 shift_value = shift[0];
  const __nv_bfloat162 shift_pair =
      __halves2bfloat162(shift_value, shift_value);
  const __nv_bfloat16 zero = __float2bfloat16(0.0f);
  const __nv_bfloat162 zero_pair = __halves2bfloat162(zero, zero);

  auto problem_shape = make_shape(M, N, K);
  auto dA = make_stride(K, Int<1>{});
  Tensor mA = make_tensor(
      make_gmem_ptr(x), select<0, 2>(problem_shape), dA);
  auto cta_tiler = make_shape(bM, bN, bK);
  auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});

  using ThreadLayout =
      Layout<Shape<Int<kThreads / 8>, _8>, Stride<_8, _1>>;
  auto copy_a = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::bfloat16_t>{},
      ThreadLayout{},
      Layout<Shape<_1, _8>>{});
  auto thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tAsA = thr_copy_a.partition_D(sA);

  // Predicates only need to guard M; every deployed K is a multiple of 64.
  Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));
  Tensor tAcA = thr_copy_a.partition_S(cA);
  Tensor tApA = make_tensor<bool>(
      make_shape(size<1>(tAsA), size<2>(tAsA)), Stride<_1, _0>{});
  const int valid_m = min(BlockM, M - m0);
  CUTE_UNROLL
  for (int m_it = 0; m_it < size<0>(tApA); ++m_it) {
    tApA(m_it, 0) = get<0>(tAcA(0, m_it, 0)) < valid_m;
  }

  auto decode_b_stage = [&](int pipe, int k_tile, bool valid) {
    const int vectors_per_tile = kBlockK / 4;
    const int vector_slots = BlockN * vectors_per_tile;
    for (int slot = threadIdx.x; slot < vector_slots; slot += kThreads) {
      const int local_n = slot / vectors_per_tile;
      const int local_vector = slot % vectors_per_tile;
      const int global_n = n0 + local_n;
      const int global_vector = k_tile * vectors_per_tile + local_vector;
      __nv_bfloat162 value_pairs[2];
      if (valid && global_n < N) {
        const int linear_index = global_n * total_vectors + global_vector;
        uint32_t index;
        if constexpr (CooperativeDecode &&
                      (StaticBits == 4 || StaticBits == 5 || StaticBits == 6 || StaticBits == 7)) {
          index = wide_fetch
              ? decode_index_cooperative_wide<StaticBits>(
                    packed, linear_index, packed_bytes)
              : decode_index_cooperative<StaticBits>(packed, linear_index);
        } else {
          index = decode_index<StaticBits>(
              packed, linear_index, index_bits, window_bytes);
        }
        const int group = k_tile;
        // All 16 d4 vectors in this K tile share one group scale for a given
        // output row. Threads are laid out as contiguous 16-lane row groups,
        // so load the scale once and broadcast its bits within each subgroup
        // instead of issuing 16 identical global loads.
        uint32_t scale_bits = 0;
        if ((threadIdx.x & 15) == 0) {
          const __nv_bfloat16 scale_source =
              scales[global_n * num_groups + group];
          scale_bits = static_cast<uint32_t>(
              *reinterpret_cast<const uint16_t*>(&scale_source));
        }
        scale_bits = __shfl_sync(0xffffffffu, scale_bits, 0, 16);
        const uint16_t scale_bits_u16 = static_cast<uint16_t>(scale_bits);
        const __nv_bfloat16 scale =
            *reinterpret_cast<const __nv_bfloat16*>(&scale_bits_u16);
        const __nv_bfloat162 scale_pair =
            __halves2bfloat162(scale, scale);
        __nv_bfloat162 codebook_pairs[2];
        if constexpr (SmemCodebook) {
          const uint64_t word = smem_codebook[index];
          const uint32_t lo = static_cast<uint32_t>(word);
          const uint32_t hi = static_cast<uint32_t>(word >> 32);
          codebook_pairs[0] = *reinterpret_cast<const __nv_bfloat162*>(&lo);
          codebook_pairs[1] = *reinterpret_cast<const __nv_bfloat162*>(&hi);
        } else {
          const auto* global_pairs =
              reinterpret_cast<const __nv_bfloat162*>(codebook + index * 4);
          codebook_pairs[0] = global_pairs[0];
          codebook_pairs[1] = global_pairs[1];
        }
        CUTE_UNROLL
        for (int pair = 0; pair < 2; ++pair) {
          value_pairs[pair] = __hadd2(
              __hmul2(codebook_pairs[pair], scale_pair), shift_pair);
        }
      } else {
        CUTE_UNROLL
        for (int pair = 0; pair < 2; ++pair) {
          value_pairs[pair] = zero_pair;
        }
      }
      if constexpr (VectorSharedStore) {
        // Four consecutive K components remain consecutive under the Ampere
        // swizzle and begin at an 8-byte-aligned shared-memory address.  Store
        // the reconstructed d4 vector in one instruction instead of four
        // independently addressed STS.U16 instructions.
        const uint32_t pair0 =
            *reinterpret_cast<const uint32_t*>(&value_pairs[0]);
        const uint32_t pair1 =
            *reinterpret_cast<const uint32_t*>(&value_pairs[1]);
        const uint64_t packed_values =
            static_cast<uint64_t>(pair0) |
            (static_cast<uint64_t>(pair1) << 32);
        const int smem_offset = sB_layout(
            make_coord(local_n, local_vector * 4, pipe));
        *reinterpret_cast<uint64_t*>(smem.b.begin() + smem_offset) =
            packed_values;
      } else {
        const auto* values =
            reinterpret_cast<const __nv_bfloat16*>(value_pairs);
        CUTE_UNROLL
        for (int component = 0; component < 4; ++component) {
          const int local_k = local_vector * 4 + component;
          const uint16_t bits =
              *reinterpret_cast<const uint16_t*>(&values[component]);
          sB(local_n, local_k, pipe) =
              *reinterpret_cast<const cute::bfloat16_t*>(&bits);
        }
      }
    }
  };

  // Two-stage prologue: A uses cp.async; B is reconstructed into the same
  // swizzled shared-memory pipeline stage.
  copy_if(copy_a, tApA, tAgA(_, _, _, k_begin), tAsA(_, _, _, 0));
  decode_b_stage(0, k_begin, true);
  cp_async_fence();

  auto tiled_mma = make_tiled_mma(
      SM80_16x8x16_F32BF16BF16F32_TN{},
      Layout<Shape<Int<kWarpM>, Int<kWarpN>>>{},
      Tile<Int<BlockM>, Int<BlockN>, _16>{});
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);

  auto output_shape = make_shape(M, N);
  auto dC = make_stride(N, Int<1>{});
  Tensor mC = make_tensor(make_gmem_ptr(output), output_shape, dC);
  Tensor gC = local_tile(
      mC,
      make_shape(bM, bN),
      make_coord(blockIdx.y, blockIdx.x));
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  auto s2r_copy_a = make_tiled_copy_A(
      Copy_Atom<SM75_U32x4_LDSM_N, cute::bfloat16_t>{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);

  auto s2r_copy_b = make_tiled_copy_B(
      Copy_Atom<SM75_U32x4_LDSM_N, cute::bfloat16_t>{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = s2r_thr_copy_b.partition_S(sB);
  Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);

  int read_pipe = 0;
  int write_pipe = 1;
  int next_k_tile = k_begin + 1;
  int remaining = my_k_tiles - 1;
  auto k_blocks = size<2>(tCrA);
  Tensor tXsA_pipe = tXsA(_, _, _, read_pipe);
  Tensor tXsB_pipe = tXsB(_, _, _, read_pipe);

  cp_async_wait<0>();
  __syncthreads();
  copy(
      s2r_copy_a,
      tXsA_pipe(_, _, Int<0>{}),
      tXrA(_, _, Int<0>{}));
  copy(
      s2r_copy_b,
      tXsB_pipe(_, _, Int<0>{}),
      tXrB(_, _, Int<0>{}));

  while (remaining > -1) {
    CUTE_UNROLL
    for (int k_block = 0; k_block < k_blocks; ++k_block) {
      if (k_block == k_blocks - 1) {
        // read_pipe was advanced when k_block==0, but the current tensor
        // slice must remain live through the rest of this K tile.
        tXsA_pipe = tXsA(_, _, _, read_pipe);
        tXsB_pipe = tXsB(_, _, _, read_pipe);
        cp_async_wait<0>();
        __syncthreads();
      }
      const int next_block = (k_block + 1) % k_blocks;
      copy(
          s2r_copy_a,
          tXsA_pipe(_, _, next_block),
          tXrA(_, _, next_block));
      copy(
          s2r_copy_b,
          tXsB_pipe(_, _, next_block),
          tXrB(_, _, next_block));

      if (k_block == 0) {
        if (remaining > 0) {
          copy_if(
              copy_a,
              tApA,
              tAgA(_, _, _, next_k_tile),
              tAsA(_, _, _, write_pipe));
          decode_b_stage(write_pipe, next_k_tile, true);
          ++next_k_tile;
        } else {
          clear(tAsA(_, _, _, write_pipe));
          decode_b_stage(write_pipe, k_begin, false);
        }
        cp_async_fence();
        --remaining;
        write_pipe = read_pipe;
        read_pipe = 1 - read_pipe;
      }

      gemm(
          tiled_mma,
          tCrA(_, _, k_block),
          tCrB(_, _, k_block),
          tCrC);
    }
  }

  cp_async_wait<0>();
  __syncthreads();

  Tensor cC = make_identity_tensor(make_shape(bM, bN));
  Tensor tCcC = thr_mma.partition_C(cC);
  CUTE_UNROLL
  for (int idx = 0; idx < size(tCrC); ++idx) {
    const auto coord = tCcC(idx);
    const int local_m = get<0>(coord);
    const int local_n = get<1>(coord);
    if (local_m < valid_m && n0 + local_n < N) {
      float value = tCrC(idx);
      if (k_split > 1) {
        // Bias is added once, by the reduction, not once per K slice.
        const size_t slice = static_cast<size_t>(blockIdx.z) * M * N;
        partials[slice + static_cast<size_t>(m0 + local_m) * N + n0 + local_n] =
            value;
      } else {
        if (has_bias) {
          value += __bfloat162float(bias[n0 + local_n]);
        }
        tCgC(idx) = cute::bfloat16_t(value);
      }
    }
  }
}

// Sums the k_split partial products in a fixed slice order, so the result is
// deterministic run to run. It differs from the unsplit kernel only by
// floating-point accumulation order, the same class of difference the direct
// packed GEMM already has against cuBLAS.
__global__ void codebook_splitk_reduce_kernel(
    const float* __restrict__ partials,
    const __nv_bfloat16* __restrict__ bias,
    cute::bfloat16_t* __restrict__ output,
    int M,
    int N,
    int k_split,
    bool has_bias) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = M * N;
  if (index >= total) {
    return;
  }
  float acc = 0.0f;
  const size_t stride = static_cast<size_t>(M) * N;
  for (int slice = 0; slice < k_split; ++slice) {
    acc += partials[slice * stride + index];
  }
  if (has_bias) {
    acc += __bfloat162float(bias[index % N]);
  }
  output[index] = cute::bfloat16_t(acc);
}

// M=17..64 candidate: eight producer warps reconstruct packed B tiles while the
// original eight consumer warps execute the previous tile's Tensor Core MMA.
// This is intentionally a separate kernel so the validated path remains
// untouched until all-shape and full-action A/B tests show a real win.
template <int BlockN, int StaticBits>
__global__ void codebook_cute_ws_sm80_kernel(
    const cute::bfloat16_t* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const __nv_bfloat16* __restrict__ codebook,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ shift,
    const __nv_bfloat16* __restrict__ bias,
    cute::bfloat16_t* __restrict__ output,
    int M,
    int N,
    int K,
    int num_groups,
    int index_bits,
    int window_bytes,
    bool has_bias) {
  using namespace cute;

  constexpr int BlockM = 64;
  constexpr int kWarpM = BlockM / 16;
  constexpr int kWarpN = BlockN / 16;
  constexpr int kConsumerThreads = kWarpM * kWarpN * 32;
  constexpr int kProducerThreads = 256;
  constexpr int kTotalThreads = kConsumerThreads + kProducerThreads;
  static_assert(BlockN == 32);
  static_assert(kConsumerThreads == 256);
  static_assert(kTotalThreads == 512);

  auto bM = Int<BlockM>{};
  auto bN = Int<BlockN>{};
  auto bK = Int<kBlockK>{};
  auto bP = Int<kStages>{};
  auto swizzle_atom = composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{});
  auto sA_layout = tile_to_shape(swizzle_atom, make_shape(bM, bK, bP));
  auto sB_layout = tile_to_shape(swizzle_atom, make_shape(bN, bK, bP));
  using SmemStorage = WarpSpecializedSharedStorage<
      cute::bfloat16_t,
      cute::bfloat16_t,
      decltype(sA_layout),
      decltype(sB_layout)>;
  extern __shared__ char shared_memory[];
  SmemStorage& smem = *reinterpret_cast<SmemStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.a.begin()), sA_layout);
  Tensor sB = make_tensor(make_smem_ptr(smem.b.begin()), sB_layout);

  const int m0 = blockIdx.y * BlockM;
  const int n0 = blockIdx.x * BlockN;
  const int total_vectors = K / 4;
  const int total_k_tiles = K / kBlockK;
  const __nv_bfloat16 shift_value = shift[0];
  const __nv_bfloat162 shift_pair =
      __halves2bfloat162(shift_value, shift_value);

  if (threadIdx.x < kStages) {
    smem.b_ready[threadIdx.x] = 0;
    smem.producer_done[threadIdx.x] = 0;
  }
  __syncthreads();

  if (threadIdx.x >= kConsumerThreads) {
    const int producer_tid = threadIdx.x - kConsumerThreads;
    const int lane = producer_tid & 31;
    constexpr int vectors_per_tile = kBlockK / 4;
    constexpr int vector_slots = BlockN * vectors_per_tile;

    for (int k_tile = 0; k_tile < total_k_tiles; ++k_tile) {
      const int pipe = k_tile & 1;
      if (lane == 0) {
        while (atomicAdd(&smem.b_ready[pipe], 0) != 0) {
          __nanosleep(64);
        }
      }
      __syncwarp();

      for (int slot = producer_tid; slot < vector_slots;
           slot += kProducerThreads) {
        const int local_n = slot / vectors_per_tile;
        const int local_vector = slot % vectors_per_tile;
        const int global_n = n0 + local_n;
        const int global_vector = k_tile * vectors_per_tile + local_vector;
        const int linear_index = global_n * total_vectors + global_vector;
        const uint32_t index = decode_index<StaticBits>(
            packed, linear_index, index_bits, window_bytes);

        uint32_t scale_bits = 0;
        if ((producer_tid & 15) == 0) {
          const __nv_bfloat16 scale_source =
              scales[global_n * num_groups + k_tile];
          scale_bits = static_cast<uint32_t>(
              *reinterpret_cast<const uint16_t*>(&scale_source));
        }
        scale_bits = __shfl_sync(0xffffffffu, scale_bits, 0, 16);
        const uint16_t scale_bits_u16 = static_cast<uint16_t>(scale_bits);
        const __nv_bfloat16 scale =
            *reinterpret_cast<const __nv_bfloat16*>(&scale_bits_u16);
        const __nv_bfloat162 scale_pair =
            __halves2bfloat162(scale, scale);
        const auto* codebook_pairs = reinterpret_cast<const __nv_bfloat162*>(
            codebook + index * 4);
        __nv_bfloat162 value_pairs[2];
        CUTE_UNROLL
        for (int pair = 0; pair < 2; ++pair) {
          value_pairs[pair] = __hadd2(
              __hmul2(codebook_pairs[pair], scale_pair), shift_pair);
        }
        const auto* values =
            reinterpret_cast<const __nv_bfloat16*>(value_pairs);
        CUTE_UNROLL
        for (int component = 0; component < 4; ++component) {
          const int local_k = local_vector * 4 + component;
          const uint16_t bits =
              *reinterpret_cast<const uint16_t*>(&values[component]);
          sB(local_n, local_k, pipe) =
              *reinterpret_cast<const cute::bfloat16_t*>(&bits);
        }
      }

      __syncwarp();
      if (lane == 0) {
        const int previous = atomicAdd(&smem.producer_done[pipe], 1);
        if (previous == 7) {
          atomicExch(&smem.producer_done[pipe], 0);
          __threadfence_block();
          atomicExch(&smem.b_ready[pipe], 1);
        }
      }
      // Keep all producer warps in the same K-tile phase. Without this named
      // barrier a fast warp can lap the last warp and overwrite the same
      // double-buffer stage before it has been published.
      producer_barrier_256();
    }
    return;
  }

  auto problem_shape = make_shape(M, N, K);
  auto dA = make_stride(K, Int<1>{});
  Tensor mA = make_tensor(
      make_gmem_ptr(x), select<0, 2>(problem_shape), dA);
  auto cta_tiler = make_shape(bM, bN, bK);
  auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});

  using ThreadLayout =
      Layout<Shape<Int<kConsumerThreads / 8>, _8>, Stride<_8, _1>>;
  auto copy_a = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::bfloat16_t>{},
      ThreadLayout{},
      Layout<Shape<_1, _8>>{});
  auto thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tAsA = thr_copy_a.partition_D(sA);
  Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));
  Tensor tAcA = thr_copy_a.partition_S(cA);
  Tensor tApA = make_tensor<bool>(
      make_shape(size<1>(tAsA), size<2>(tAsA)), Stride<_1, _0>{});
  const int valid_m = min(BlockM, M - m0);
  CUTE_UNROLL
  for (int m_it = 0; m_it < size<0>(tApA); ++m_it) {
    tApA(m_it, 0) = get<0>(tAcA(0, m_it, 0)) < valid_m;
  }

  copy_if(copy_a, tApA, tAgA(_, _, _, 0), tAsA(_, _, _, 0));
  cp_async_fence();

  auto tiled_mma = make_tiled_mma(
      SM80_16x8x16_F32BF16BF16F32_TN{},
      Layout<Shape<Int<kWarpM>, Int<kWarpN>>>{},
      Tile<Int<BlockM>, Int<BlockN>, _16>{});
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  auto output_shape = make_shape(M, N);
  auto dC = make_stride(N, Int<1>{});
  Tensor mC = make_tensor(make_gmem_ptr(output), output_shape, dC);
  Tensor gC = local_tile(
      mC,
      make_shape(bM, bN),
      make_coord(blockIdx.y, blockIdx.x));
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  auto s2r_copy_a = make_tiled_copy_A(
      Copy_Atom<SM75_U32x4_LDSM_N, cute::bfloat16_t>{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);
  auto s2r_copy_b = make_tiled_copy_B(
      Copy_Atom<SM75_U32x4_LDSM_N, cute::bfloat16_t>{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = s2r_thr_copy_b.partition_S(sB);
  Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);
  const int k_blocks = size<2>(tCrA);

  for (int k_tile = 0; k_tile < total_k_tiles; ++k_tile) {
    const int pipe = k_tile & 1;
    cp_async_wait<0>();
    if (threadIdx.x == 0) {
      while (atomicAdd(&smem.b_ready[pipe], 0) == 0) {
        __nanosleep(64);
      }
    }
    consumer_barrier_256();

    Tensor tXsA_pipe = tXsA(_, _, _, pipe);
    Tensor tXsB_pipe = tXsB(_, _, _, pipe);
    CUTE_UNROLL
    for (int k_block = 0; k_block < k_blocks; ++k_block) {
      copy(
          s2r_copy_a,
          tXsA_pipe(_, _, k_block),
          tXrA(_, _, k_block));
      copy(
          s2r_copy_b,
          tXsB_pipe(_, _, k_block),
          tXrB(_, _, k_block));
    }
    consumer_barrier_256();

    if (threadIdx.x == 0) {
      __threadfence_block();
      atomicExch(&smem.b_ready[pipe], 0);
    }
    if (k_tile + 1 < total_k_tiles) {
      const int write_pipe = 1 - pipe;
      copy_if(
          copy_a,
          tApA,
          tAgA(_, _, _, k_tile + 1),
          tAsA(_, _, _, write_pipe));
      cp_async_fence();
    }

    CUTE_UNROLL
    for (int k_block = 0; k_block < k_blocks; ++k_block) {
      gemm(
          tiled_mma,
          tCrA(_, _, k_block),
          tCrB(_, _, k_block),
          tCrC);
    }
  }

  cp_async_wait<0>();
  Tensor cC = make_identity_tensor(make_shape(bM, bN));
  Tensor tCcC = thr_mma.partition_C(cC);
  CUTE_UNROLL
  for (int idx = 0; idx < size(tCrC); ++idx) {
    const auto coord = tCcC(idx);
    const int local_m = get<0>(coord);
    const int local_n = get<1>(coord);
    if (local_m < valid_m && n0 + local_n < N) {
      float value = tCrC(idx);
      if (has_bias) {
        value += __bfloat162float(bias[n0 + local_n]);
      }
      tCgC(idx) = cute::bfloat16_t(value);
    }
  }
}

template <int BlockM, int BlockN, int StaticBits>
void launch_codebook_cute(
    const cute::bfloat16_t* x,
    const uint8_t* packed,
    const __nv_bfloat16* codebook,
    const __nv_bfloat16* scales,
    const __nv_bfloat16* shift,
    const __nv_bfloat16* bias,
    cute::bfloat16_t* output,
    int M,
    int N,
    int K,
    int num_groups,
    int index_bits,
    int window_bytes,
    bool has_bias,
    int packed_bytes,
    cudaStream_t stream) {
  using namespace cute;
  constexpr int kWarpM = BlockM / 16;
  constexpr int kWarpN = BlockN / 16;
  constexpr int kThreads = kWarpM * kWarpN * 32;
  auto swizzle_atom = composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{});
  auto sA_layout = tile_to_shape(
      swizzle_atom,
      make_shape(Int<BlockM>{}, Int<kBlockK>{}, Int<kStages>{}));
  auto sB_layout = tile_to_shape(
      swizzle_atom,
      make_shape(Int<BlockN>{}, Int<kBlockK>{}, Int<kStages>{}));
  using Storage = SharedStorage<
      cute::bfloat16_t,
      cute::bfloat16_t,
      decltype(sA_layout),
      decltype(sB_layout)>;
  const int base_smem = sizeof(Storage);
  const int entries = codebook_entry_count(StaticBits, index_bits);
  const int table_bytes = entries * 8;
  const bool stage_codebook =
      entries > 0 && table_bytes <= kSmemCodebookMaxBytes &&
      base_smem + table_bytes <= kSmemBudgetBytes;
  const int smem_bytes = base_smem + (stage_codebook ? table_bytes : 0)
                         + kSmemPadKB * 1024;
  // The d4 vector starts at an 8-byte-aligned address in the swizzled B tile;
  // emit one STS.64 instead of four independently addressed STS.U16 stores.
  auto kernel =
      stage_codebook
          ? codebook_cute_sm80_kernel<BlockM, BlockN, StaticBits, false, true,
                                      true>
          : codebook_cute_sm80_kernel<BlockM, BlockN, StaticBits, false, true,
                                      false>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  const dim3 block(kThreads);
  const int gx = (N + BlockN - 1) / BlockN;
  const int gy = (M + BlockM - 1) / BlockM;

  // Pick a K split only when the N/M grid alone cannot occupy the device. Every
  // slice must still hold whole 64-wide K tiles, so the split is capped by the
  // tile count as well as by kSplitKMax.
  const int total_k_tiles = K / kBlockK;
  int k_split = 1;
  // Only the small-M tiles are starved. BlockM=160 (the M=149 path) already
  // issues 640 threads per CTA, and measurement shows splitting K there costs
  // about 2% for no occupancy gain, so it is excluded by construction.
  const bool splittable =
      BlockM <= 64 || (kSplitBigMHost && BlockM > 64);
  if (kSplitKEnabled && splittable) {
    const int target_ctas = target_ctas_now();
    const int target = (target_ctas + gx * gy - 1) / (gx * gy);
    k_split = max(1, min(min(target, kSplitKMax), total_k_tiles));
  }
  float* partials = nullptr;
  at::Tensor workspace;
  if (k_split > 1) {
    workspace = at::empty({k_split, M, N},
                          at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
    partials = workspace.data_ptr<float>();
  }
  const dim3 grid(gx, gy, k_split);
  kernel<<<grid, block, smem_bytes, stream>>>(
      x,
      packed,
      codebook,
      scales,
      shift,
      bias,
      output,
      M,
      N,
      K,
      num_groups,
      index_bits,
      window_bytes,
      has_bias,
      stage_codebook ? entries : 0,
      partials,
      k_split,
      packed_bytes,
      kWideFetchHost);
  if (k_split > 1) {
    const int total = M * N;
    const int threads = 256;
    codebook_splitk_reduce_kernel<<<(total + threads - 1) / threads, threads, 0,
                                    stream>>>(
        partials, bias, output, M, N, k_split, has_bias);
  }
}

template <int BlockM, int StaticBits>
void launch_codebook_cute_scalarstore(
    const cute::bfloat16_t* x,
    const uint8_t* packed,
    const __nv_bfloat16* codebook,
    const __nv_bfloat16* scales,
    const __nv_bfloat16* shift,
    const __nv_bfloat16* bias,
    cute::bfloat16_t* output,
    int M,
    int N,
    int K,
    int num_groups,
    int index_bits,
    int window_bytes,
    bool has_bias,
    int packed_bytes,
    cudaStream_t stream) {
  using namespace cute;
  constexpr int BlockN = 32;
  constexpr int kWarpM = BlockM / 16;
  constexpr int kWarpN = BlockN / 16;
  constexpr int kThreads = kWarpM * kWarpN * 32;
  auto swizzle_atom = composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{});
  auto sA_layout = tile_to_shape(
      swizzle_atom,
      make_shape(Int<BlockM>{}, Int<kBlockK>{}, Int<kStages>{}));
  auto sB_layout = tile_to_shape(
      swizzle_atom,
      make_shape(Int<BlockN>{}, Int<kBlockK>{}, Int<kStages>{}));
  using Storage = SharedStorage<
      cute::bfloat16_t,
      cute::bfloat16_t,
      decltype(sA_layout),
      decltype(sB_layout)>;
  const int base_smem = sizeof(Storage);
  const int entries = codebook_entry_count(StaticBits, index_bits);
  const int table_bytes = entries * 8;
  const bool stage_codebook =
      entries > 0 && table_bytes <= kSmemCodebookMaxBytes &&
      base_smem + table_bytes <= kSmemBudgetBytes;
  const int smem_bytes = base_smem + (stage_codebook ? table_bytes : 0)
                         + kSmemPadKB * 1024;
  auto kernel =
      stage_codebook
          ? codebook_cute_sm80_kernel<BlockM, BlockN, StaticBits, false, false,
                                      true>
          : codebook_cute_sm80_kernel<BlockM, BlockN, StaticBits, false, false,
                                      false>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  const dim3 block(kThreads);
  const dim3 grid((N + BlockN - 1) / BlockN, (M + BlockM - 1) / BlockM);
  kernel<<<grid, block, smem_bytes, stream>>>(
      x,
      packed,
      codebook,
      scales,
      shift,
      bias,
      output,
      M,
      N,
      K,
      num_groups,
      index_bits,
      window_bytes,
      has_bias,
      stage_codebook ? entries : 0,
      nullptr,
      1,
      packed_bytes,
      kWideFetchHost);
}

}  // namespace

torch::Tensor codebook_cutlass_gemm_cuda(
    torch::Tensor x,
    torch::Tensor packed_indices,
    torch::Tensor codebook,
    torch::Tensor scales,
    torch::Tensor shift,
    torch::Tensor bias,
    int64_t index_bits,
    int64_t index_window_bytes) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16,
              "x must be CUDA BF16");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous(),
              "x must be contiguous [M,K]");
  TORCH_CHECK(packed_indices.is_cuda() &&
                  packed_indices.scalar_type() == torch::kUInt8,
              "packed_indices must be CUDA uint8");
  TORCH_CHECK(codebook.is_cuda() &&
                  codebook.scalar_type() == torch::kBFloat16 &&
                  codebook.dim() == 2 && codebook.size(1) == 4,
              "codebook must be CUDA BF16 [C,4]");
  TORCH_CHECK(scales.is_cuda() && scales.scalar_type() == torch::kBFloat16,
              "scales must be CUDA BF16");
  TORCH_CHECK(shift.is_cuda() && shift.scalar_type() == torch::kBFloat16 &&
                  shift.numel() == 1,
              "shift must be one CUDA BF16 value");
  TORCH_CHECK(index_bits >= 1 && index_bits <= 24,
              "index_bits must be in [1,24]");
  TORCH_CHECK(index_window_bytes >= 1 && index_window_bytes <= 4,
              "index_window_bytes must be in [1,4]");

  const int M = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(scales.size(0));
  const int num_groups = static_cast<int>(scales.size(1));
  TORCH_CHECK(K % kBlockK == 0, "K must be divisible by 64");
  TORCH_CHECK(N % 32 == 0, "N must be divisible by 32");
  TORCH_CHECK(num_groups == K / 64,
              "scales shape does not match K/group_size");
  TORCH_CHECK(
      static_cast<int64_t>(N) * (K / 4) * index_bits <=
          std::numeric_limits<int>::max(),
      "packed bit position exceeds the int32 decoder range");
  const bool has_bias = bias.defined() && bias.numel() != 0;
  if (has_bias) {
    TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == torch::kBFloat16 &&
                    bias.numel() == N,
                "bias must be CUDA BF16 [N]");
  }

  c10::cuda::CUDAGuard device_guard(x.device());
  auto output = torch::empty({M, N}, x.options());
  const auto* x_ptr = reinterpret_cast<const cute::bfloat16_t*>(
      x.data_ptr<at::BFloat16>());
  const auto* codebook_ptr = reinterpret_cast<const __nv_bfloat16*>(
      codebook.data_ptr<at::BFloat16>());
  const auto* scales_ptr = reinterpret_cast<const __nv_bfloat16*>(
      scales.data_ptr<at::BFloat16>());
  const auto* shift_ptr = reinterpret_cast<const __nv_bfloat16*>(
      shift.data_ptr<at::BFloat16>());
  const auto* bias_ptr = has_bias
      ? reinterpret_cast<const __nv_bfloat16*>(
            bias.data_ptr<at::BFloat16>())
      : nullptr;
  auto* output_ptr = reinterpret_cast<cute::bfloat16_t*>(
      output.data_ptr<at::BFloat16>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_CUTE(BLOCK_M, STATIC_BITS)                                      \
  launch_codebook_cute<BLOCK_M, 32, STATIC_BITS>(                             \
      x_ptr, packed_indices.data_ptr<uint8_t>(), codebook_ptr, scales_ptr,     \
      shift_ptr, bias_ptr, output_ptr, M, N, K, num_groups,                   \
      static_cast<int>(index_bits), static_cast<int>(index_window_bytes),      \
      has_bias, static_cast<int>(packed_indices.numel()), stream)
#define DISPATCH_BITS(BLOCK_M)                                                  \
  do {                                                                          \
    if (index_bits == 4) {                                                       \
      LAUNCH_CUTE(BLOCK_M, 4);                                                   \
    } else if (index_bits == 5) {                                                \
      LAUNCH_CUTE(BLOCK_M, 5);                                                   \
    } else if (index_bits == 6) {                                                \
      LAUNCH_CUTE(BLOCK_M, 6);                                                   \
    } else if (index_bits == 7) {                                                \
      LAUNCH_CUTE(BLOCK_M, 7);                                                   \
    } else {                                                                     \
      LAUNCH_CUTE(BLOCK_M, 0);                                                   \
    }                                                                            \
  } while (false)
  if (M <= 16) {
    DISPATCH_BITS(16);
  } else if (kUseBM48 && M > 16 && M <= 48) {
    DISPATCH_BITS(48);
  } else if (M >= 129 && M <= 160) {
    // M=149 previously launched three BM64 CTAs, decoding every packed B tile
    // three times. A single BM160 CTA reuses each reconstructed B tile across
    // all ten M warps; the N grid still provides ample parallelism.
    DISPATCH_BITS(160);
  } else {
    DISPATCH_BITS(64);
  }
#undef DISPATCH_BITS
#undef LAUNCH_CUTE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor codebook_cutlass_scalarstore_gemm_cuda(
    torch::Tensor x,
    torch::Tensor packed_indices,
    torch::Tensor codebook,
    torch::Tensor scales,
    torch::Tensor shift,
    torch::Tensor bias,
    int64_t index_bits,
    int64_t index_window_bytes) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16,
              "x must be CUDA BF16");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous(),
              "x must be contiguous [M,K]");
  TORCH_CHECK(packed_indices.is_cuda() &&
                  packed_indices.scalar_type() == torch::kUInt8,
              "packed_indices must be CUDA uint8");
  TORCH_CHECK(codebook.is_cuda() &&
                  codebook.scalar_type() == torch::kBFloat16 &&
                  codebook.dim() == 2 && codebook.size(1) == 4,
              "codebook must be CUDA BF16 [C,4]");
  TORCH_CHECK(scales.is_cuda() && scales.scalar_type() == torch::kBFloat16,
              "scales must be CUDA BF16");
  TORCH_CHECK(shift.is_cuda() && shift.scalar_type() == torch::kBFloat16 &&
                  shift.numel() == 1,
              "shift must be one CUDA BF16 value");
  TORCH_CHECK(index_bits >= 1 && index_bits <= 24,
              "index_bits must be in [1,24]");
  TORCH_CHECK(index_window_bytes >= 1 && index_window_bytes <= 4,
              "index_window_bytes must be in [1,4]");

  const int M = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(scales.size(0));
  const int num_groups = static_cast<int>(scales.size(1));
  TORCH_CHECK(K % kBlockK == 0, "K must be divisible by 64");
  TORCH_CHECK(N % 32 == 0, "N must be divisible by 32");
  TORCH_CHECK(num_groups == K / 64,
              "scales shape does not match K/group_size");
  TORCH_CHECK(
      static_cast<int64_t>(N) * (K / 4) * index_bits <=
          std::numeric_limits<int>::max(),
      "packed bit position exceeds the int32 decoder range");
  const bool has_bias = bias.defined() && bias.numel() != 0;
  if (has_bias) {
    TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == torch::kBFloat16 &&
                    bias.numel() == N,
                "bias must be CUDA BF16 [N]");
  }

  c10::cuda::CUDAGuard device_guard(x.device());
  auto output = torch::empty({M, N}, x.options());
  const auto* x_ptr = reinterpret_cast<const cute::bfloat16_t*>(
      x.data_ptr<at::BFloat16>());
  const auto* codebook_ptr = reinterpret_cast<const __nv_bfloat16*>(
      codebook.data_ptr<at::BFloat16>());
  const auto* scales_ptr = reinterpret_cast<const __nv_bfloat16*>(
      scales.data_ptr<at::BFloat16>());
  const auto* shift_ptr = reinterpret_cast<const __nv_bfloat16*>(
      shift.data_ptr<at::BFloat16>());
  const auto* bias_ptr = has_bias
      ? reinterpret_cast<const __nv_bfloat16*>(
            bias.data_ptr<at::BFloat16>())
      : nullptr;
  auto* output_ptr = reinterpret_cast<cute::bfloat16_t*>(
      output.data_ptr<at::BFloat16>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_CUTE_SCALARSTORE(BLOCK_M, STATIC_BITS)                          \
  launch_codebook_cute_scalarstore<BLOCK_M, STATIC_BITS>(                     \
      x_ptr, packed_indices.data_ptr<uint8_t>(), codebook_ptr, scales_ptr,     \
      shift_ptr, bias_ptr, output_ptr, M, N, K, num_groups,                   \
      static_cast<int>(index_bits), static_cast<int>(index_window_bytes),      \
      has_bias, static_cast<int>(packed_indices.numel()), stream)
#define DISPATCH_SCALARSTORE_BITS(BLOCK_M)                                     \
  do {                                                                          \
    if (index_bits == 4) {                                                       \
      LAUNCH_CUTE_SCALARSTORE(BLOCK_M, 4);                                       \
    } else if (index_bits == 5) {                                                \
      LAUNCH_CUTE_SCALARSTORE(BLOCK_M, 5);                                       \
    } else if (index_bits == 6) {                                                \
      LAUNCH_CUTE_SCALARSTORE(BLOCK_M, 6);                                       \
    } else {                                                                     \
      LAUNCH_CUTE_SCALARSTORE(BLOCK_M, 0);                                       \
    }                                                                            \
  } while (false)
  if (M <= 16) {
    DISPATCH_SCALARSTORE_BITS(16);
  } else if (M >= 129 && M <= 160) {
    DISPATCH_SCALARSTORE_BITS(160);
  } else {
    DISPATCH_SCALARSTORE_BITS(64);
  }
#undef DISPATCH_SCALARSTORE_BITS
#undef LAUNCH_CUTE_SCALARSTORE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

void set_ffn_launch_config(int64_t ctas, int64_t splits, bool bm48) {
  TORCH_CHECK(ctas > 0 && splits > 0 && splits <= 8, "Invalid operational launch config");
  kRequestedCTAs = static_cast<int>(ctas); kSplitKMax = static_cast<int>(splits); kUseBM48 = bm48;
}
