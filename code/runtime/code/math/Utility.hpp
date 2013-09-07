#ifndef MATH_UTILITY_HPP
#define MATH_UTILITY_HPP

#include "core/Assert.hpp"

#include "math/Constants.hpp"

namespace math
{

class Vector3;
class Vector4;
class Matrix44;

float DegreesToRadians( float degree );
float RadiansToDegrees( float radian );
float Abs             ( float value );
int   Abs             ( int   value );
float Squared         ( float value );
int   Squared         ( int   value );
bool  Equals          ( float          left, float          right, float epsilon = 0.01f );
bool  Equals          ( const Vector3& left, const Vector3& right, float epsilon = 0.01f );
float Clamp           ( float min, float max, float value );
float Min             ( float a, float b );
int   Min             ( int   a, int   b );
float Max             ( float a, float b );
int   Max             ( int   a, int   b );
int   EndianSwap      ( int s );

}

#endif
