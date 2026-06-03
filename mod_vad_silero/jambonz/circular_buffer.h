#ifndef CIRCULAR_BUFFER_H
#define CIRCULAR_BUFFER_H

#include <cstring>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

/**
 * This is a simple circular buffer implementation that can be used to store
 * 16-bit audio samples. It is designed to be used in a single-producer,
 * single-consumer scenario, where one thread is adding data to the buffer
 * and another is reading from it. The buffer is not thread-safe, so it is
 * up to the user to ensure that the producer and consumer do not access the
 * buffer concurrently.
 *
 * This implementation was motivated by the need to find a more efficient
 * cirtular buffer implementation than the one provided by Boost. The Boost
 * implementation requires erase operations to be performed on the buffer
 * which can be expensive for large buffers. This implementation adjusts
 * read and write pointers to avoid the need for erase operations.
 *
 */
class CircularBuffer {
public:
  CircularBuffer(uint32_t capacity) : m_capacity(capacity), m_size(0), m_start(0), m_end(0) {
    m_data = static_cast<int16_t*>(std::aligned_alloc(64, capacity * sizeof(int16_t)));
    if (!m_data) {
      throw std::runtime_error("Aligned memory allocation failed");
    }
  }

  ~CircularBuffer() {
    std::free(m_data);
  }

  void add(const int16_t* data, uint32_t count) {
    if (count > available()) {
      throw std::runtime_error("Buffer overflow: not enough space");
    }

    for (uint32_t i = 0; i < count; ++i) {
      m_data[m_end] = data[i];
      m_end = (m_end + 1) % m_capacity;
    }
    m_size += count;
  }

  // Retrieve a block of data, handling wrap-around
  uint32_t getBlock(int16_t* dest, uint32_t count, bool resizing = false) {
    if (count > m_size) {
      count = m_size;  // Only retrieve what's available
    }

    uint32_t firstPart = std::min(count, m_capacity - m_start);

    // Copy the first part of the data
    memcpy(dest, &m_data[m_start], firstPart * sizeof(int16_t));

    uint32_t secondPart = count - firstPart;
    if (secondPart > 0) {
      // Copy the wrapped-around portion
      memcpy(dest + firstPart, &m_data[0], secondPart * sizeof(int16_t));
    }

    // Update the start position and size
    uint32_t newStart = (m_start + count) % m_capacity;
    m_start = newStart;
    if (!resizing) m_size -= count;

    // Prefetch the next chunk
    uint32_t nextOffset = (newStart + count) % m_capacity;
    __builtin_prefetch(&m_data[nextOffset], 0, 1);  // High locality hint

    return count;
  }

  // Resize the buffer to a new capacity
  uint32_t set_capacity(uint32_t newCapacity) {
    if (newCapacity < m_size) {
        throw std::runtime_error("New capacity cannot be smaller than current size");
    }

    // Calculate the required size and round up to the next multiple of 64
    size_t totalBytes = newCapacity * sizeof(int16_t);
    size_t alignedSize = (totalBytes + 63) & ~63; // Round up to 64-byte boundary
    
    // Use aligned_alloc with the proper alignment requirements
    int16_t* newData = static_cast<int16_t*>(std::aligned_alloc(64, alignedSize));
    if (!newData) {
        throw std::runtime_error("Aligned memory allocation failed");
    }
    
    uint32_t retrieved = getBlock(newData, m_size, true);
    std::free(m_data);  // Free the old buffer

    m_data = newData;
    m_capacity = newCapacity;
    m_start = 0;
    m_end = retrieved;

    return alignedSize / sizeof(int16_t); // Return the new capacity
  }
  uint32_t removeValue(int16_t value) {
      uint32_t countRemoved = 0;
      uint32_t writeIdx = m_start; // Start writing at the current start position
      uint32_t readIdx = m_start;
      uint32_t elementsToProcess = m_size;

      while (elementsToProcess > 0) {
          if (m_data[readIdx] != value) {
              if (readIdx != writeIdx) {
                  m_data[writeIdx] = m_data[readIdx]; // Only copy if necessary
              }
              writeIdx = (writeIdx + 1) % m_capacity; // Move write index forward
          } else {
              ++countRemoved; // Count the removed element
          }

          readIdx = (readIdx + 1) % m_capacity; // Move read index forward
          --elementsToProcess;
      }

      // Update the buffer size and end pointer
      m_size -= countRemoved;
      m_end = writeIdx;

      return countRemoved; // Return the number of removed elements
  }

  // Get the current capacity of the buffer
  uint32_t capacity() const {
    return m_capacity;
  }

  // Get the current number of elements in the buffer
  uint32_t size() const {
    return m_size;
  }

  // Get the remaining available space in the buffer
  uint32_t available() const {
    return m_capacity - m_size;
  }

  void clear() {
    m_size = 0;
    m_start = 0;
    m_end = 0;
  }

  void getState(uint32_t& start, uint32_t& end, uint32_t& size, uint32_t& capacity) {
    start = m_start;
    end = m_end;
    size = m_size;
    capacity = m_capacity;
  }

private:
    int16_t* m_data;  // Internal storage
    uint32_t m_capacity;  // Total capacity of the buffer
    uint32_t m_size;  // Current number of elements in the buffer
    uint32_t m_start;  // Read pointer
    uint32_t m_end;  // Write pointer
};

#endif
