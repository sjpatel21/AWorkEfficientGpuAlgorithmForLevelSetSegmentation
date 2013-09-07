#include "core/Assert.hpp"

#if defined(PLATFORM_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

#include "core/String.hpp"
#include "core/Printf.hpp"

#if defined(PLATFORM_WIN32)
static bool sMessageBoxExists = false;

AssertResponse AssertHelper( const char* expressionString, int lineNumber, const char* fileName )
{
    core::String assertMessage;
    
    assertMessage += "Filename: ";
    assertMessage += fileName;
    assertMessage += "\n\n\nLine Number: %1 \n\n\nExpression: ";
    assertMessage += expressionString;
    assertMessage += "\n\n\n";
    
    assertMessage = assertMessage.arg( lineNumber );

    core::Printf( assertMessage );

    int result = 0;

    if ( !sMessageBoxExists )
    {
        sMessageBoxExists = true;
        result            = MessageBoxA( 0, assertMessage.ToAscii(), "Assert", MB_ABORTRETRYIGNORE | MB_SETFOREGROUND | MB_SYSTEMMODAL | MB_ICONEXCLAMATION );
        sMessageBoxExists = false;

        switch ( result )
        {
        case IDABORT:
            return AssertResponse_Abort;
            break;

        case IDRETRY:
            return AssertResponse_Retry;
            break;

        case IDIGNORE:
            return AssertResponse_Ignore;
            break;

        default:
            __debugbreak();
            return AssertResponse_Retry;            
        }
    }

    _asm { int 3 }
    return AssertResponse_Retry;
}
#elif defined(PLATFORM_OSX)
AssertResponse AssertHelper( const char* expressionString, int lineNumber, const char* fileName )
{
    return AssertResponse_Retry;
}
#endif
