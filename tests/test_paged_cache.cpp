#include <cassert>
#include <iostream>
#include "paged_kv_cache.hpp"

using namespace inference;

int main() {
    std::cout << "[TEST] Running Paged KV Cache Manager Tests...\n";

    PagedKVCacheManager cache_mgr(64); // 64 physical blocks (64 * 16 = 1024 tokens capacity)

    assert(cache_mgr.get_free_blocks() == 64);

    // 1. Allocate blocks for sequence of 35 tokens (requires 3 blocks of 16 tokens)
    auto blocks_seq1 = cache_mgr.allocate_sequence_blocks(35);
    assert(blocks_seq1.size() == 3);
    assert(cache_mgr.get_free_blocks() == 61);

    // 2. Prefix Caching: Sequence 2 shares first 2 blocks with Sequence 1
    int shared = cache_mgr.share_prefix_blocks({blocks_seq1[0], blocks_seq1[1]});
    assert(shared == 2);

    // 3. Free Sequence 1: first 2 blocks should remain active because Sequence 2 shares them
    cache_mgr.free_sequence_blocks(blocks_seq1);
    assert(cache_mgr.get_free_blocks() == 62); // block 3 freed, blocks 1 and 2 still held by seq 2

    // 4. Free Sequence 2 prefix references
    cache_mgr.free_sequence_blocks({blocks_seq1[0], blocks_seq1[1]});
    assert(cache_mgr.get_free_blocks() == 64); // All blocks completely reclaimed

    std::cout << "  Paged KV Cache allocation, reference counting, and prefix sharing verified!\n";
    std::cout << "[PASS] Paged KV Cache Manager tests passed cleanly!\n";
    return 0;
}
