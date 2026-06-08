#ifndef CUDA_ENGINE_PAGED_KV_CACHE_HPP
#define CUDA_ENGINE_PAGED_KV_CACHE_HPP

#include <vector>
#include <unordered_map>
#include <memory>
#include <deque>
#include <cstdint>
#include <iostream>
#include <cassert>

namespace inference {

struct KVBlock {
    int block_id;
    int ref_count{0};
    bool is_allocated{false};
};

class PagedKVCacheManager {
public:
    static constexpr int BLOCK_SIZE = 16; // 16 tokens per KV block (vLLM standard)

    PagedKVCacheManager(int num_blocks = 1024)
        : total_blocks_(num_blocks), free_blocks_count_(num_blocks) {
        blocks_.resize(num_blocks);
        for (int i = 0; i < num_blocks; ++i) {
            blocks_[i].block_id = i;
            free_list_.push_back(i);
        }
    }

    // Allocate physical blocks for a sequence with sequence length
    std::vector<int> allocate_sequence_blocks(int seq_len) {
        int blocks_needed = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (blocks_needed > free_blocks_count_) {
            return {}; // Out of KV memory
        }

        std::vector<int> allocated_ids;
        for (int i = 0; i < blocks_needed; ++i) {
            int bid = free_list_.front();
            free_list_.pop_front();
            blocks_[bid].is_allocated = true;
            blocks_[bid].ref_count = 1;
            allocated_ids.push_back(bid);
            free_blocks_count_--;
        }
        return allocated_ids;
    }

    void free_sequence_blocks(const std::vector<int>& block_ids) {
        for (int bid : block_ids) {
            if (bid >= 0 && bid < total_blocks_ && blocks_[bid].is_allocated) {
                blocks_[bid].ref_count--;
                if (blocks_[bid].ref_count == 0) {
                    blocks_[bid].is_allocated = false;
                    free_list_.push_back(bid);
                    free_blocks_count_++;
                }
            }
        }
    }

    // Prefix Caching: shares KV blocks across requests with common prompt prefix
    int share_prefix_blocks(const std::vector<int>& prefix_block_ids) {
        int shared_count = 0;
        for (int bid : prefix_block_ids) {
            if (bid >= 0 && bid < total_blocks_ && blocks_[bid].is_allocated) {
                blocks_[bid].ref_count++;
                shared_count++;
            }
        }
        return shared_count;
    }

    int get_free_blocks() const { return free_blocks_count_; }
    int get_total_blocks() const { return total_blocks_; }

    double get_memory_utilization() const {
        return static_cast<double>(total_blocks_ - free_blocks_count_) / total_blocks_;
    }

private:
    int total_blocks_;
    int free_blocks_count_;
    std::vector<KVBlock> blocks_;
    std::deque<int> free_list_;
};

} // namespace inference

#endif // CUDA_ENGINE_PAGED_KV_CACHE_HPP
