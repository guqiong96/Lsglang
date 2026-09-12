// Exact immutable FP8 row retrieval; bounded direct-mapped RAM cache.
// No CUDA calls in the callback: suitable for cudaLaunchHostFunc graph nodes.
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <linux/aio_abi.h>
#include <mutex>
#include <new>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

struct Store {
  int fd;
  aio_context_t aio = 0;
  uint8_t *pages = nullptr;
  std::mutex batch_lock;
  std::atomic<uint64_t> batches{0}, peak_pages{0};
  uint64_t rows, weight_offset, scale_offset, slots, row_lo, row_hi;
  uint8_t *cache;
  uint64_t *keys;
  std::atomic<uint64_t> hits{0}, misses{0}, reads{0};
};
struct Work {
  Store *store;
  const int64_t *ids;
  uint8_t *weights, *scales;
  uint64_t count;
};

static void fail(const char *reason) {
  std::fprintf(stderr, "Engram retrieval failed: %s (errno=%d)\n", reason,
               errno);
  std::abort(); // Never allow a generation to continue with missing/stale rows.
}

extern "C" void row_store_close(Store *s);

extern "C" Store *row_store_open(const char *path, uint64_t rows, uint64_t woff,
                                 uint64_t soff, uint64_t budget) {
  auto *s = new (std::nothrow) Store{};
  if (!s) {
    errno = ENOMEM;
    return nullptr;
  }
  s->fd = open(path, O_RDONLY | O_CLOEXEC | O_DIRECT);
  if (s->fd < 0) {
    delete s;
    return nullptr;
  }
  struct stat statbuf;
  if (fstat(s->fd, &statbuf) || !S_ISREG(statbuf.st_mode) ||
      rows > uint64_t(INT64_MAX) || woff > uint64_t(statbuf.st_size) ||
      soff > uint64_t(statbuf.st_size) ||
      rows > (uint64_t(statbuf.st_size) - woff) / 256 ||
      rows > (uint64_t(statbuf.st_size) - soff) / 8) {
    row_store_close(s);
    errno = EINVAL;
    return nullptr;
  }
  s->rows = rows;
  s->weight_offset = woff;
  s->scale_offset = soff;
  s->row_lo = 0;
  s->row_hi = rows;
  s->slots = budget / 272;
  if (s->slots) {
    s->cache = static_cast<uint8_t *>(mmap(nullptr, s->slots * 264,
                                           PROT_READ | PROT_WRITE,
                                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    s->keys = static_cast<uint64_t *>(mmap(nullptr, s->slots * sizeof(uint64_t),
                                           PROT_READ | PROT_WRITE,
                                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (s->cache == MAP_FAILED || s->keys == MAP_FAILED) {
      int error = errno;
      row_store_close(s);
      errno = error;
      return nullptr;
    }
  }
  int error =
      posix_memalign(reinterpret_cast<void **>(&s->pages), 4096, 128 * 4096);
  if (error) {
    row_store_close(s);
    errno = error;
    return nullptr;
  }
  if (syscall(SYS_io_setup, 128, &s->aio) < 0) {
    error = errno;
    row_store_close(s);
    errno = error;
    return nullptr;
  }
  return s;
}

// At most 32 misses and 128 unique pages per batch, including boundary
// crossings.
struct ReadBatch {
  std::array<uint64_t, 128> offsets{};
  std::array<size_t, 128> required{};
  size_t count = 0;
  void add(uint64_t offset, size_t length) {
    while (length) {
      uint64_t base = offset & ~uint64_t(4095);
      size_t delta = offset - base, take = std::min(length, 4096 - delta);
      size_t j = 0;
      while (j < count && offsets[j] != base)
        ++j;
      if (j == count) {
        if (count == offsets.size())
          fail("batch page limit");
        offsets[count++] = base;
      }
      required[j] = std::max(required[j], delta + take);
      offset += take;
      length -= take;
    }
  }
  void read(Store *s) {
    if (!count)
      return;
    std::array<iocb, 128> requests{};
    std::array<iocb *, 128> pointers{};
    for (size_t j = 0; j < count; ++j) {
      auto &request = requests[j];
      request.aio_data = j;
      request.aio_lio_opcode = IOCB_CMD_PREAD;
      request.aio_fildes = s->fd;
      request.aio_buf = reinterpret_cast<uint64_t>(s->pages + j * 4096);
      request.aio_nbytes = 4096;
      request.aio_offset = offsets[j];
      pointers[j] = &request;
    }
    size_t submitted = 0;
    while (submitted < count) {
      long queued;
      do {
        queued = syscall(SYS_io_submit, s->aio, count - submitted,
                         pointers.data() + submitted);
      } while (queued < 0 && errno == EINTR);
      if (queued <= 0)
        fail("io_submit");
      std::array<io_event, 128> events{};
      long completed = 0;
      while (completed < queued) {
        long got;
        do {
          got = syscall(SYS_io_getevents, s->aio, 1, queued - completed,
                        events.data(), nullptr);
        } while (got < 0 && errno == EINTR);
        if (got <= 0)
          fail("io_getevents");
        for (long j = 0; j < got; ++j) {
          const auto &event = events[j];
          if (event.data < submitted || event.data >= submitted + queued ||
              event.res < 0 || event.res2 != 0 ||
              uint64_t(event.res) < required[event.data])
            fail("short or failed asynchronous read");
        }
        completed += got;
      }
      submitted += queued;
    }
    s->reads.fetch_add(count, std::memory_order_relaxed);
    ++s->batches;
    if (count > s->peak_pages.load())
      s->peak_pages.store(count);
  }
  void copy(Store *s, uint64_t offset, uint8_t *out, size_t length) const {
    while (length) {
      uint64_t base = offset & ~uint64_t(4095);
      size_t delta = offset - base, take = std::min(length, 4096 - delta);
      size_t j = 0;
      while (j < count && offsets[j] != base)
        ++j;
      if (j == count)
        fail("missing batch page");
      std::memcpy(out, s->pages + j * 4096 + delta, take);
      offset += take;
      out += take;
      length -= take;
    }
  }
};

extern "C" void row_store_lookup(void *opaque) {
  auto *work = static_cast<Work *>(opaque);
  Store *s = work->store;
  // One callback owns the bounded I/O buffers and cache until every read
  // completes.
  std::lock_guard<std::mutex> guard(s->batch_lock);
  for (uint64_t begin = 0; begin < work->count; begin += 32) {
    ReadBatch batch;
    std::array<uint64_t, 32> misses{};
    size_t miss_count = 0;
    uint64_t end = begin + std::min(uint64_t(32), work->count - begin);
    for (uint64_t i = begin; i < end; ++i) {
      int64_t id = work->ids[i];
      if (id < 0 || uint64_t(id) >= s->rows)
        fail("row ID out of bounds");
      if (uint64_t(id) < s->row_lo || uint64_t(id) >= s->row_hi) {
        std::memset(work->weights + i * 256, 0, 256);
        std::memset(work->scales + i * 8, 0, 8);
        continue;
      }
      uint64_t slot = s->slots ? uint64_t(id) % s->slots : 0;
      if (s->slots && s->keys[slot] == uint64_t(id) + 1) {
        std::memcpy(work->weights + i * 256, s->cache + slot * 264, 256);
        std::memcpy(work->scales + i * 8, s->cache + slot * 264 + 256, 8);
        ++s->hits;
      } else {
        misses[miss_count++] = i;
        batch.add(s->weight_offset + uint64_t(id) * 256, 256);
        batch.add(s->scale_offset + uint64_t(id) * 8, 8);
        ++s->misses;
      }
    }
    batch.read(s);
    for (size_t j = 0; j < miss_count; ++j) {
      uint64_t i = misses[j], id = work->ids[i];
      uint8_t row[264];
      batch.copy(s, s->weight_offset + id * 256, row, 256);
      batch.copy(s, s->scale_offset + id * 8, row + 256, 8);
      std::memcpy(work->weights + i * 256, row, 256);
      std::memcpy(work->scales + i * 8, row + 256, 8);
      if (s->slots) {
        uint64_t slot = id % s->slots;
        std::memcpy(s->cache + slot * 264, row, 264);
        s->keys[slot] = id + 1;
      }
    }
  }
}

extern "C" void row_store_stats(Store *s, uint64_t *out) {
  out[0] = s->hits.load();
  out[1] = s->misses.load();
  out[2] = s->reads.load();
  out[3] = s->slots * 272;
}
extern "C" void row_store_range(Store *s, uint64_t lo, uint64_t hi) {
  if (lo > hi || hi > s->rows)
    fail("invalid row ownership range");
  s->row_lo = lo;
  s->row_hi = hi;
}
extern "C" void row_store_close(Store *s) {
  if (s->cache && s->cache != MAP_FAILED)
    munmap(s->cache, s->slots * 264);
  if (s->keys && s->keys != MAP_FAILED)
    munmap(s->keys, s->slots * sizeof(uint64_t));
  if (s->aio && syscall(SYS_io_destroy, s->aio) < 0)
    fail("io_destroy");
  std::free(s->pages);
  close(s->fd);
  delete s;
}
