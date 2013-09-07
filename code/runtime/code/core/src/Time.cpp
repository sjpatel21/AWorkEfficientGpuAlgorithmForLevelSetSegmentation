#include "core/Time.hpp"

#ifdef PLATFORM_WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <stdio.h>
#include <stdarg.h>

#if defined(PLATFORM_OSX)
#include <sys/time.h>
/*
 This is the definition of LARGE_INTEGER
 for unix platforms only, Windows already has
 this defined.
 */
typedef union _LARGE_INTEGER
{
    struct
    {
        unsigned long LowPart;
        long HighPart;
    };
    struct
    {
        unsigned long LowPart;
        long HighPart;
    } u;
    long long QuadPart;
} LARGE_INTEGER;

#endif

namespace core
{
    
#if defined(PLATFORM_WIN32)
    static bool sIsInitialized = false;
    static LARGE_INTEGER sPreviousCount, sCurrentCount;
    
    double TimeGetTimeDeltaSeconds()
    {
        LARGE_INTEGER frequency;
        
        QueryPerformanceFrequency( &frequency );
        QueryPerformanceCounter( &sCurrentCount );
        
        __int64 countDelta = sCurrentCount.QuadPart - sPreviousCount.QuadPart;
        sPreviousCount     = sCurrentCount;
        
        if ( !sIsInitialized )
        {
            sIsInitialized = true;
            return 0;
        }
        else
        {
            double timeDeltaSeconds = static_cast< double >( countDelta ) / static_cast< double >( frequency.QuadPart );
            
            return timeDeltaSeconds;
        }
    }
#elif defined(PLATFORM_OSX)
    double TimeGetTimeDeltaSeconds()
    {
        static struct timeval _tstart, _tend;
        static struct timezone tz;
        double t1, t2;
        
        gettimeofday(&_tstart, &tz);
        gettimeofday(&_tend,&tz);
        
        t1 =  (double)_tstart.tv_sec + (double)_tstart.tv_usec/(1000*1000);
        t2 =  (double)_tend.tv_sec + (double)_tend.tv_usec/(1000*1000);
        
        return t2-t1;
    }    
#endif
}