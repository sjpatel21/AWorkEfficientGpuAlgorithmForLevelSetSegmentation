#ifndef RENDERING_TEXT_CONSOLE_HPP
#define RENDERING_TEXT_CONSOLE_HPP

#include "core/NameSpaceID.hpp"

namespace core
{
    class String;
}

namespace math
{
    class Vector3;
}

namespace rendering
{

class TextConsole
{
public:
    static void RenderCallback();
    static void PrintToStaticConsole( const core::String& channel, const core::String& text );
    static void PrintToStaticConsole( const core::String& channel, const math::Vector3& vector );
};

}

#endif