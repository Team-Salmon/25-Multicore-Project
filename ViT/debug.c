#include "debug.h"

static clock_t timer;

void start_timer() {
	timer = clock();
}

void stop_timer(const char* message) {
	clock_t end_time = clock();
	double elapsed_time = (double)(end_time - timer) / CLOCKS_PER_SEC * 1000.0;

	printf("%s... (%.0fms)\n", message, elapsed_time);
}

Profiler g_timers[MAX_KERNELS];
int g_timer_count = 0;

void init_profiler() {
    g_timer_count = 0;
    memset(g_timers, 0, sizeof(g_timers));
}

void profile_event(cl_event event, const char* name) {
    cl_int err;

    err = clWaitForEvents(1, &event);

    cl_ulong start, end;
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);

    if (err != CL_SUCCESS) {
        printf("[Profiler Error] Failed to get info for %s\n", name);
        return;
    }

    double duration_ms = (double)(end - start) * 1e-6;

    int found = -1;
    for (int i = 0; i < g_timer_count; i++) {
        if (strcmp(g_timers[i].name, name) == 0) {
            found = i;
            break;
        }
    }

    if (found == -1) {
        if (g_timer_count < MAX_KERNELS) {
            found = g_timer_count++;
            strncpy(g_timers[found].name, name, 63);
        }
        else {
            return;
        }
    }

    g_timers[found].total_ms += duration_ms;
    g_timers[found].call_count++;
}

void print_profiler_stats() {
    printf("\n================ Kernel Profiling Results ================\n");
    printf("%-25s | %-10s | %-12s | %-10s\n", "Kernel Name", "Calls", "Total (ms)", "Avg (ms)");
    printf("----------------------------------------------------------\n");

    double global_total = 0;

    for (int i = 0; i < g_timer_count; i++) {
        double avg = g_timers[i].total_ms / g_timers[i].call_count;
        printf("%-25s | %-10d | %-12.3f | %-10.3f\n",
            g_timers[i].name,
            g_timers[i].call_count,
            g_timers[i].total_ms,
            avg);
        global_total += g_timers[i].total_ms;
    }
    printf("----------------------------------------------------------\n");
    printf("TOTAL GPU KERNEL TIME: %.3f ms\n", global_total);
    printf("==========================================================\n\n");
}