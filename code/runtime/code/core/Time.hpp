#ifndef CORE_TIME_HPP
#define CORE_TIME_HPP

namespace core
{

// This function returns the number of milliseconds elapsed since the most recent previous call
// to this function.  If the function has never been called, it will return 0.
double TimeGetTimeDeltaSeconds();

}

#endif