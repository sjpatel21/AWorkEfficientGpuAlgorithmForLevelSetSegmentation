#ifndef RENDERING_RTGI_OPENGL_TEXT_FONT_HPP
#define RENDERING_RTGI_OPENGL_TEXT_FONT_HPP

#include "rendering/rtgi/TextFont.hpp"

namespace rendering
{

namespace rtgi
{

class OpenGLTextFont : public TextFont
{
public:
    OpenGLTextFont();
    virtual ~OpenGLTextFont();

    virtual void Render( int x, int y, const core::String& text ) const;
};

}

}

#endif