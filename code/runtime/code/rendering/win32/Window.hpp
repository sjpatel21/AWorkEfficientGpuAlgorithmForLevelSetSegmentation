#ifndef RENDERING_WINDOW_HPP
#define RENDERING_WINDOW_HPP

#if defined PLATFORM_WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace rendering
{

namespace win32
{

enum WindowStatus
{
    WindowStatus_Error  = -1,
    WindowStatus_Quit   = 0,
    WindowStatus_NoQuit = 1
};

namespace Window
{

#if defined PLATFORM_WIN32
void Initialize( HWND* windowInstance = NULL );
#endif

void Terminate();

int  Update( double timeDeltaSeconds );

}

}

}

#endif