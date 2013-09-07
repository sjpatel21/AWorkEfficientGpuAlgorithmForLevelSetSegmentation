#ifndef RENDERING_DEBUG_DRAW_HPP
#define RENDERING_DEBUG_DRAW_HPP

#include "math/Vector3.hpp"
#include "math/Vector4.hpp"
#include "math/Utility.hpp"

#include "rendering/rtgi/Effect.hpp"
#include "rendering/rtgi/Color.hpp"

namespace rendering
{

namespace DebugDraw
{

// 3D
void DrawPoint   ( const math::Vector3& position,                             const rtgi::ColorRGB& color, int pointSize );
void DrawSphere  ( const math::Vector3& position, float sphereRadius,         const rtgi::ColorRGB& color );
void DrawTeapot  ( const math::Vector3& position, float boundingSphereRadius, const rtgi::ColorRGB& color );

void DrawLine    ( const math::Vector3& p1, const math::Vector3& p2, const rtgi::ColorRGB& color );
void DrawTriangle( const math::Vector3& p1, const math::Vector3& p2, const math::Vector3& p3, const rtgi::ColorRGB& color );
void DrawQuad    ( const math::Vector3& p1, const math::Vector3& p2, const math::Vector3& p3, const math::Vector3& p4, const rtgi::ColorRGB& color );
void DrawCube    ( const math::Vector3& p1, const math::Vector3& p2, const rtgi::ColorRGB& color );

// 2D
void DrawPoint2D( int virtualScreenX, int virtualScreenY, const rtgi::ColorRGB& color, int pointSize );
void DrawLine2D ( int virtualScreenX1, int virtualScreenY1, int virtualScreenX2, int virtualScreenY2, const rtgi::ColorRGB& color );
void DrawFullScreenQuad2D( const rtgi::ColorRGB& color );
void DrawQuad2D( int topLeftX, int topLeftY, int width, int height, const rtgi::ColorRGB& color );

}

}

#endif