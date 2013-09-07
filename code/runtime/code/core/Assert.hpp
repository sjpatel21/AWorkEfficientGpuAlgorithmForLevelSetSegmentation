#ifndef CORE_ASSERT_HPP
#define CORE_ASSERT_HPP

#if defined(PLATFORM_WIN32)
#include <intrin.h>
#define DEBUG_BREAK __debugbreak()
#elif defined(PLATFORM_OSX)
#define DEBUG_BREAK __asm {int 3};
#endif

#include <stdlib.h>

#include "core/Config.hpp"

enum AssertResponse
{
    AssertResponse_Abort,
    AssertResponse_Retry,
    AssertResponse_Ignore
};


#ifdef ENABLE_ASSERTS

    extern AssertResponse AssertHelper( const char* expressionString, int lineNumber, const char* fileName );

    #define Assert( expression )                                                                   \
    do                                                                                             \
    {                                                                                              \
        if ( !( expression ) )                                                                     \
        {                                                                                          \
            AssertResponse response = AssertHelper( #expression, __LINE__, __FILE__ );             \
                                                                                                   \
            switch( response )                                                                     \
            {                                                                                      \
            case AssertResponse_Ignore:                                                            \
                break;                                                                             \
                                                                                                   \
            case AssertResponse_Retry:                                                             \
                DEBUG_BREAK;                                                                    \
                break;                                                                             \
                                                                                                   \
            case AssertResponse_Abort:                                                             \
                exit( -1 );                                                                        \
                break;                                                                             \
            }                                                                                      \
        }                                                                                          \
    } while ( 0 )                                                                                  \

#else

    #define Assert( expression )

#endif



#ifdef ENABLE_RELEASE_ASSERTS

    extern AssertResponse AssertHelper( const char* expressionString, int lineNumber, const char* fileName );

    #define ReleaseAssert( expression )                                                            \
    do                                                                                             \
    {                                                                                              \
        if ( !( expression ) )                                                                     \
        {                                                                                          \
            AssertResponse response = AssertHelper( #expression, __LINE__, __FILE__ );             \
                                                                                                   \
            switch( response )                                                                     \
            {                                                                                      \
            case AssertResponse_Ignore:                                                            \
                break;                                                                             \
                                                                                                   \
            case AssertResponse_Retry:                                                             \
                DEBUG_BREAK;                                                                    \
                break;                                                                             \
                                                                                                   \
            case AssertResponse_Abort:                                                             \
                exit( -1 );                                                                        \
                break;                                                                             \
            }                                                                                      \
        }                                                                                          \
    } while ( 0 )                                                                                  \

#else

#define ReleaseAssert( expression )

#endif

#endif