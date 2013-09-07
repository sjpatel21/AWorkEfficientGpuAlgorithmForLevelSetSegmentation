#ifndef RENDERING_RTGI_COLOR_HPP
#define RENDERING_RTGI_COLOR_HPP

#include "math/Vector3.hpp"
#include "math/Vector4.hpp"

namespace rendering
{

namespace rtgi
{

typedef math::Vector3 ColorRGB;
typedef math::Vector4 ColorRGBA;

static const int R = 0;
static const int G = 1;
static const int B = 2;
static const int A = 3;
}

}
#endif