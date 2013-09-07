#include "math/Utility.hpp"

#include "math/Vector4.hpp"
#include "math/Matrix44.hpp"

namespace math
{

float DegreesToRadians( float degree )
{
    return degree * PI / 180.0f;
};

float RadiansToDegrees( float radian )
{
    return radian * 180.0f / PI;
};

float Abs( float value )
{
    if ( value < 0 )
    {
        return value * -1.0f;
    }
    else
    {
        return value;
    }
};

int Abs( int value )
{
    if ( value < 0 )
    {
        return value * -1;
    }
    else
    {
        return value;
    }
};

float Squared( float value )
{
    return value * value;
};

int Squared( int value )
{
    return value * value;
};

bool Equals( float left, float right, float epsilon )
{
    float delta = left - right;

    return Abs( delta ) < epsilon;
};

bool Equals( const Vector3& left, const Vector3& right, float epsilon )
{
    return Equals( left[ X ], right[ X ], epsilon ) && Equals( left[ Y ], right[ Y ], epsilon ) && Equals( left[ Z ], right[ Z ], epsilon );
};


float Clamp( float min, float max, float value )
{
    Assert( min <= max );

    if ( value < min )
    {
        value = min;
    }
    else if ( value > max )
    {
        value = max;
    }

    return value;
}

float Min( float a, float b )
{
    if ( a < b ) { return a; }
    else { return b; }
}

int Min( int a, int b )
{
    if ( a < b ) { return a; }
    else { return b; }
}

float Max( float a, float b )
{
    if ( a > b ) { return a; }
    else { return b; }
}

int Max( int a, int b )
{
    if ( a > b ) { return a; }
    else { return b; }
}

int EndianSwap( int i )
{
    int s = ( ( i >> 24 ) & 0xFFFFFFFF ) | 
            ( ( i << 8  ) & 0x00FF0000 ) |
            ( ( i >> 8  ) & 0x0000FF00 ) |
            ( ( i << 24 ) & 0xFFFFFFFF );

    return s;
}

}                