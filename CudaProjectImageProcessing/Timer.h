#include <Windows.h>
#include <time.h>

struct timezone {
    int tz_minuteswest;
    int tz_dsttime;
};

class Timer
{
private:
    timeval start;

public:
    Timer();
    float get_elapsed_seconds();
    void reset();

private:
    int get_time_of_day(struct timeval* tv, struct timezone* tz);
    float timeval_to_float(struct timeval* time);
    void get_gap_time(struct timeval* start_time, struct timeval* end_time, struct timeval* gap_time);
};