#ifndef DEBUG_H
#define DEBUG_H

#include <stdio.h>
#include <time.h>
#include <CL/cl.h>

#define MAX_KERNELS 20

static clock_t timer;

typedef struct {
    char name[64];
    double total_ms;
    int call_count;
} Profiler;

extern Profiler g_timers[MAX_KERNELS];
extern int g_timer_count;

void init_profiler();
void print_profiler_stats();
void profile_event(cl_event event, const char* name);

void start_timer();
void stop_timer(const char* message);

#endif
