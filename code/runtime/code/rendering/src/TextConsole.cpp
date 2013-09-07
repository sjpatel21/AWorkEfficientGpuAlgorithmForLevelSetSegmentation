#include "rendering/TextConsole.hpp"

#include "core/Assert.hpp"
#include "core/String.hpp"

#include "math/Vector3.hpp"

#include "container/InsertOrderedMap.hpp"

#include "rendering/rtgi/RTGI.hpp"
#include "rendering/rtgi/TextFont.hpp"

#include "rendering/Context.hpp"

namespace rendering
{

static container::InsertOrderedMap< core::String, core::String > sStaticTextBuffer;

void TextConsole::PrintToStaticConsole( const core::String& channel, const core::String& text )
{
    sStaticTextBuffer.Insert( channel, text );
}

void TextConsole::PrintToStaticConsole( const core::String& channel, const math::Vector3& vector )
{
    PrintToStaticConsole( channel, core::String( "x = %1   y = %2    z = %3" ).arg( vector[0] ).arg( vector[1] ).arg( vector[2] ) );
}

void TextConsole::RenderCallback()
{
    int viewportWidth, viewportHeight; 
    Context::GetCurrentViewport( viewportWidth, viewportHeight );

    int currentX = 15;
    int currentY = viewportHeight - 20;

    foreach ( const core::String text, sStaticTextBuffer )
    {
        Context::GetDebugTextFont()->Render( currentX, currentY, text );

        currentY -= 20;
    }
}

}