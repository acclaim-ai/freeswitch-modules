#ifndef TIMESTAMP_H
#define TIMESTAMP_H

#include <string>
#include <cstdarg>
#include <cstdio>
#include <memory>

/**
 * stores the start and end (in samples) of a speech segment.
 */
class timestamp_t {
public:
    int start;
    int end;

    timestamp_t(int start = -1, int end = -1)
        : start(start), end(end) { }

    timestamp_t& operator=(const timestamp_t& a) {
        start = a.start;
        end = a.end;
        return *this;
    }

    bool operator==(const timestamp_t& a) const {
        return (start == a.start && end == a.end);
    }

    // Returns a formatted string of the timestamp.
    std::string c_str() const {
        return format("{start:%08d, end:%08d}", start, end);
    }
private:
    // Helper function for formatting.
    std::string format(const char* fmt, ...) const {
        char buf[256];
        va_list args;
        va_start(args, fmt);
        const auto r = std::vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
        if (r < 0)
            return {};
        const size_t len = r;
        if (len < sizeof(buf))
            return std::string(buf, len);
#if __cplusplus >= 201703L
        std::string s(len, '\0');
        va_start(args, fmt);
        std::vsnprintf(s.data(), len + 1, fmt, args);
        va_end(args);
        return s;
#else
        auto vbuf = std::unique_ptr<char[]>(new char[len + 1]);
        va_start(args, fmt);
        std::vsnprintf(vbuf.get(), len + 1, fmt, args);
        va_end(args);
        return std::string(vbuf.get(), len);
#endif
    }
};

#endif
