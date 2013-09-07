#ifndef RENDERING_RTGI_TEXT_FONT_HPP
#define RENDERING_RTGI_TEXT_FONT_HPP

#include "core/RefCounted.hpp"

namespace core
{
    class String;
}

namespace rendering
{

namespace rtgi
{

class TextFont : public core::RefCounted
{
public:
    virtual void Render( int x, int y, const core::String& text ) const = 0;

protected:
    TextFont() {};
    virtual ~TextFont() {};
};

}

}

#endif