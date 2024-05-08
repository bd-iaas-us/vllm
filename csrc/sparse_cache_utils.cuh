// sparse_cache_utils.h

#ifndef SPARSE_CACHE_UTILS_H_
#define SPARSE_CACHE_UTILS_H_

#include <cstdint>

enum class EvictMode {
    NONE,
    H2O,
};

typedef void (*EvictFunction)(const uint8_t* input, size_t input_size, EvictMode evict_mode);

inline void H2OEvictFunction(const uint8_t* input, size_t input_size, EvictMode evict_mode) {
    // Evict logic based on attention score for H2O.
}

#endif // SPARSE_CACHE_UTILS_H_