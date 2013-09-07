#include "rendering/win32/Window.hpp"

#include "core/Assert.hpp"

namespace rendering
{

namespace win32
{

namespace Window
{

static const int  MAX_WINDOW_NAME_LENGTH = 256;

static WNDCLASSEX sWindowClass;
static HWND       sWindowHandle;
static char       sWindowClassName[ MAX_WINDOW_NAME_LENGTH ] = "Sandbox Window Class";
static char       sWindowTitle[ MAX_WINDOW_NAME_LENGTH ]     = "Sandbox Window Title";
static bool       sInitialized                               = false;
static bool       sVisible                                   = false;

LRESULT WINAPI MsgProc( HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam );

void Initialize( HWND* windowHandle )
{
    Assert( !sInitialized && !sVisible );

    // Register the window class.
    WNDCLASSEX tmpWindowClass = {
        sizeof(WNDCLASSEX),    // size of WNDCLASSEX struct
        CS_CLASSDC,            // CS_CLASSDC allocates one device context to be shared by all windows in the class.
        MsgProc,               // message callback
        0L,                    // number of extra bytes to allocate following the window-class structure
        0L,                    // number of extra bytes to allocate following the window instance
        GetModuleHandle(NULL), // GetModuleHandle returns a handle to the file used to create the calling process (.exe file).
        NULL,                  // icon
        NULL,                  // cursor
        NULL,                  // background brush
        NULL,                  // menu name
        sWindowClassName,      // window class name
        NULL };                // small icon

    // more convenient to declare this way
    sWindowClass = tmpWindowClass;

    RegisterClassEx( &sWindowClass );

    // Create the application's window.
    sWindowHandle = CreateWindow(
        sWindowClass.lpszClassName,           // window class name
        sWindowTitle,                         // window title
        WS_OVERLAPPEDWINDOW,                  // window style = overlapped
        100,                                  // window x position
        100,                                  // window y position
        800,                                  // window width
        600,                                  // window height
        GetDesktopWindow(),                   // parent window
        NULL,                                 // menu
        sWindowClass.hInstance,               // instance handle
        NULL );                               // user data?

    if ( NULL != windowHandle )
    {
        *windowHandle = sWindowHandle;
    }

    // Show the window
    ShowWindow( sWindowHandle, SW_SHOWDEFAULT );
    UpdateWindow( sWindowHandle );

    sInitialized = true;
    sVisible     = true;
}

void Terminate()
{
    Assert( sInitialized );

    sInitialized = false;
    sVisible     = false;

    PostQuitMessage( 0 );

    UnregisterClass( sWindowClassName, sWindowClass.hInstance );
}

int Update( double )
{
    Assert( sInitialized && sVisible );

    MSG windowsMessage;

    if ( PeekMessage( &windowsMessage, NULL, 0U, 0U, PM_REMOVE ) )
    {
        if ( WM_CLOSE == windowsMessage.message )
        {
            return WindowStatus_Quit;
        }
        else
        {
            // do internal windows stuff
            TranslateMessage( &windowsMessage );
            DispatchMessage( &windowsMessage );
        }
    }

    // the internal windows stuff above might have closed the window.  if so, we should quit
    if ( !sVisible )
    {
        return WindowStatus_Quit;
    }
    else
    {
        return WindowStatus_NoQuit;
    }
}

LRESULT WINAPI MsgProc( HWND windowHandle, UINT windowMessage, WPARAM messageInfoWParam, LPARAM messageInfoLParam )
{
    switch( windowMessage )
    {
        case WM_CLOSE:
            sVisible = false;
            return 0;
    }

    return DefWindowProc( windowHandle, windowMessage, messageInfoWParam, messageInfoLParam );
}

}

}

}