#pragma once

#include <cstdlib>

// Keep the in-pass spin and short idle naps unchanged. After 1024 empty naps,
// an opt-in longer nap cuts standby wakeups; each new dispatch resets idle to 0.
// The counter saturates so a long-lived idle server cannot overflow signed int.
inline bool moe_idle_should_spin(int& idle)
{
    constexpr int spin_limit = 65536;
    constexpr int long_idle_limit = spin_limit + 1024;
    if (idle < spin_limit) return ++idle < spin_limit;
    if (idle < long_idle_limit) ++idle;
    return false;
}

inline int moe_idle_sleep_us(int idle)
{
    if (idle < 65536 + 1024) return 50;
    static const int long_nap = [] {
        const char* value = std::getenv("EXL3_MOE_IDLE_SLEEP_US");
        if (!value || !*value) return 50;
        char* end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        return end && *end == '\0' && parsed >= 50 && parsed <= 10000
            ? static_cast<int>(parsed) : 50;
    }();
    return long_nap;
}
