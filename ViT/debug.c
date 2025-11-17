#include "debug.h"

static clock_t timer;

void start_timer() {
	timer = clock();
}

void stop_timer(const char* message) {
	clock_t end_time = clock();
	double elapsed_time = (double)(end_time - timer) / CLOCKS_PER_SEC * 1000.0;

	printf("%s... (%.2fms)\n", message, elapsed_time);
}