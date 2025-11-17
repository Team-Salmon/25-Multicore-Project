#ifndef DEBUG_H
#define DEBUG_H

#include <stdio.h>
#include <time.h>
#include <CL/cl.h>

static clock_t timer;

void start_timer();
void stop_timer(const char* message);

#endif
