#include "math/Vector3.hpp"

#include "core/Assert.hpp"

namespace math
{

Vector3::Vector3()
{
}


Vector3::Vector3( float x, float y, float z )
{
    mElements[ X ] = x;
    mElements[ Y ] = y;
    mElements[ Z ] = z;
}


Vector3::Vector3( const Vector3& v )
{
    mElements[ X ] = v[ X ];
    mElements[ Y ] = v[ Y ];
    mElements[ Z ] = v[ Z ];
}

#ifdef CONVERT_TO_AND_FROM_FCOLLADA
Vector3::Vector3( const FMVector3& v )
{
    mElements[ X ] = v[ 0 ];
    mElements[ Y ] = v[ 1 ];
    mElements[ Z ] = v[ 2 ];
}
#endif

float& Vector3::operator [] ( int i )
{
    CheckRange( i, 0, 3, "(Vec4::[i]) index out of range" );
    return( mElements[ i ] );
}


const float& Vector3::operator [] ( int i ) const
{
    CheckRange( i, 0, 3, "(Vec4::[i]) index out of range" );
    return( mElements[ i ] );
}


const float* Vector3::Ref() const
{
    return reinterpret_cast< const float* >( mElements );
}

Vector3& Vector3::operator = ( const Vector3 &v )
{
    mElements[ X ] = v[ X ];
    mElements[ Y ] = v[ Y ];
    mElements[ Z ] = v[ Z ];

    return *this;
}


Vector3& Vector3::operator += (const Vector3 &v)
{
    mElements[ X ] += v[ X ];
    mElements[ Y ] += v[ Y ];
    mElements[ Z ] += v[ Z ];

    return *this;
}


Vector3& Vector3::operator -= ( const Vector3 &v )
{
    mElements[ X ] -= v[ X ];
    mElements[ Y ] -= v[ Y ];
    mElements[ Z ] -= v[ Z ];

    return *this;
}


Vector3& Vector3::operator *= ( const Vector3 &v )
{
    mElements[ X ] *= v[ X ];
    mElements[ Y ] *= v[ Y ];
    mElements[ Z ] *= v[ Z ];

    return *this;
}


Vector3& Vector3::operator *= (float s)
{
    mElements[ X ] *= s;
    mElements[ Y ] *= s;
    mElements[ Z ] *= s;

    return *this;
}


Vector3& Vector3::operator /= (const Vector3 &v)
{
    mElements[ X ] /= v[ X ];
    mElements[ Y ] /= v[ Y ];
    mElements[ Z ] /= v[ Z ];

    return *this;
}


Vector3& Vector3::operator /= (float s)
{
    mElements[ X ] /= s;
    mElements[ Y ] /= s;
    mElements[ Z ] /= s;

    return *this;
}


bool Vector3::operator == ( const Vector3 &a ) const
{
    return ( mElements[ X ] == a[ X ] &&
             mElements[ Y ] == a[ Y ] &&
             mElements[ Z ] == a[ Z ] );
}


bool Vector3::operator != ( const Vector3 &a ) const
{
    return ( mElements[ X ] != a[ X ] ||
             mElements[ Y ] != a[ Y ] ||
             mElements[ Z ] != a[ Z ] );
}


Vector3 Vector3::operator + ( const Vector3 &a ) const
{
    Vector3 result;

    result[ X ] = mElements[ X ] + a[ X ];
    result[ Y ] = mElements[ Y ] + a[ Y ];
    result[ Z ] = mElements[ Z ] + a[ Z ];

    return result;
}


Vector3 Vector3::operator - ( const Vector3 &a ) const
{
    Vector3 result;

    result[ X ] = mElements[ X ] - a[ X ];
    result[ Y ] = mElements[ Y ] - a[ Y ];
    result[ Z ] = mElements[ Z ] - a[ Z ];

    return result;
}


Vector3 Vector3::operator - () const
{
    Vector3 result;

    result[ X ] = - mElements[ X ];
    result[ Y ] = - mElements[ Y ];
    result[ Z ] = - mElements[ Z ];

    return result;
}


Vector3 Vector3::operator * ( const Vector3 &a ) const
{
    Vector3 result;

    result[ X ] = mElements[ X ] * a[ X ];
    result[ Y ] = mElements[ Y ] * a[ Y ];
    result[ Z ] = mElements[ Z ] * a[ Z ];

    return result;
}


Vector3 Vector3::operator * ( float s ) const
{
    Vector3 result;

    result[ X ] = mElements[ X ] * s;
    result[ Y ] = mElements[ Y ] * s;
    result[ Z ] = mElements[ Z ] * s;

    return result;
}


Vector3 Vector3::operator / ( const Vector3 &a ) const
{
    Vector3 result;

    result[ X ] = mElements[ X ] / a[ X ];
    result[ Y ] = mElements[ Y ] / a[ Y ];
    result[ Z ] = mElements[ Z ] / a[ Z ];

    return result;
}


Vector3 Vector3::operator / ( float s ) const
{
    Vector3 result;

    result[ X ] = mElements[ X ] / s;
    result[ Y ] = mElements[ Y ] / s;
    result[ Z ] = mElements[ Z ] / s;

    return result;
}


Vector3 operator * ( float s, const Vector3 &v )
{
    return ( v * s );
}


void Vector3::Normalize()
{
    Assert( SquaredLength() > 0.00000001f ); // normalizing length-zero vector

    *this /= Length();
}

float Vector3::Length() const
{
    return sqrt( DotProduct( *this, *this ) );
}

float Vector3::SquaredLength() const
{
    return( DotProduct( *this, *this ) );
}

float DotProduct( const Vector3 &a, const Vector3 &b )
{
    return( a[ X ] * b[ X ] + a[ Y ] * b[ Y ] + a[ Z ] * b[ Z ] );
}


Vector3 CrossProduct( const Vector3 &a, const Vector3 &b )
{
    Vector3 result;

    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];

    return result;
}

void Vector3::CheckRange( int index, int min, int max, const char* ) const
{
    Assert( index < max && index >= min );
}

}