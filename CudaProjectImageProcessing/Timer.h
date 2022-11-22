#include <Windows.h>
#include <time.h>

struct timezone {
    int tz_minuteswest;
    int tz_dsttime;
};

int gettimeofday(struct timeval* tv, struct timezone* tz);
float timevalToFloat(struct timeval* time);
void getGapTime(struct timeval* start_time, struct timeval* end_time, struct timeval* gap_time);