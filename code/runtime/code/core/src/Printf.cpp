#include "core/Printf.hpp"

#ifdef PLATFORM_WIN32
#define NOMINMAX
#include <windows.h>
#elif PLATFORM_OSX
#include <iostream>
#endif

#include "core/String.hpp"

namespace core
{

void Printf( const String& string )
{
#ifdef PLATFORM_WIN32
    OutputDebugString( string.ToAscii() );
#elif PLATFORM_OSX
    std::cout << string.ToAscii();
#endif
}

}