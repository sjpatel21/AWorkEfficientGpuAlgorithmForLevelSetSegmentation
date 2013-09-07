#include "math/Vector4.hpp"

#include "core/Assert.hpp"

#include "math/Vector3.hpp"

namespace math
{

Vector4::Vector4()
{
}


Vector4::Vector4( float x, float y, float z, float h )
{
    mElements[ X ] = x;
    mElements[ Y ] = y;
    mElements[ Z ] = z;
    mElements[ H ] = h;
}


Vector4::Vector4( const Vector4 &v )
{
    mElements[ X ] = v[ X ];
    mElements[ Y ] = v[ Y ];
    mElements[ Z ] = v[ Z ];
    mElements[ H ] = v[ H ];
}


Vector4::Vector4( const Vector3 &v, float h )
{
    mElements[ X ] = v[ X ];
    mElements[ Y ] = v[ Y ];
    mElements[ Z ] = v[ Z ];
    mElements[ H ] = h;
}


float& Vector4::operator [] ( int i )
{
    CheckRange( i, 0, 4, "(Vec4::[i]) index out of range" );
    return( mElements[ i ] );
}


const float& Vector4::operator [] ( int i ) const
{
    CheckRange( i, 0, 4, "(Vec4::[i]) index out of range" );
    return( mElements[ i ] );
}


const float* Vector4::Ref() const
{
    return reinterpret_cast< const float* >( mElements );
}


Vector4& Vector4::operator = ( const Vector4 &v )
{
    mElements[ X ] = v[ X ];
    mElements[ Y ] = v[ Y ];
    mElements[ Z ] = v[ Z ];
    mElements[ H ] = v[ H ];

    return *this;
}


Vector4& Vector4::operator += (const Vector4 &v)
{
    mElements[ X ] += v[ X ];
    mElements[ Y ] += v[ Y ];
    mElements[ Z ] += v[ Z ];
    mElements[ H ] += v[ H ];

    return *this;
}


Vector4& Vector4::operator -= ( const Vector4 &v )
{
    mElements[ X ] -= v[ X ];
    mElements[ Y ] -= v[ Y ];
    mElements[ Z ] -= v[ Z ];
    mElements[ H ] -= v[ H ];

    return *this;
}


Vector4& Vector4::operator *= ( const Vector4 &v )
{
    mElements[ X ] *= v[ X ];
    mElements[ Y ] *= v[ Y ];
    mElements[ Z ] *= v[ Z ];
    mElements[ H ] *= v[ H ];

    return *this;
}


Vector4& Vector4::operator *= (float s)
{
    mElements[ X ] *= s;
    mElements[ Y ] *= s;
    mElements[ Z ] *= s;
    mElements[ H ] *= s;

    return *this;
}


Vector4& Vector4::operator /= (const Vector4 &v)
{
    mElements[ X ] /= v[ X ];
    mElements[ Y ] /= v[ Y ];
    mElements[ Z ] /= v[ Z ];
    mElements[ H ] /= v[ H ];

    return *this;
}


Vector4& Vector4::operator /= (float s)
{
    mElements[ X ] /= s;
    mElements[ Y ] /= s;
    mElements[ Z ] /= s;
    mElements[ H ] /= s;

    return *this;
}


bool Vector4::operator == ( const Vector4 &a ) const
{
    return ( mElements[ X ] == a[ X ] &&
        mElements[ Y ] == a[ Y ] &&
        mElements[ Z ] == a[ Z ] &&
        mElements[ H ] == a[ H ] );
}


bool Vector4::operator != ( const Vector4 &a ) const
{
    return ( mElements[ X ] != a[ X ] ||
        mElements[ Y ] != a[ Y ] ||
        mElements[ Z ] != a[ Z ] ||
        mElements[ H ] != a[ H ] );
}


Vector4 Vector4::operator + ( const Vector4 &a ) const
{
    Vector4 result;

    result[ X ] = mElements[ X ] + a[ X ];
    result[ Y ] = mElements[ Y ] + a[ Y ];
    result[ Z ] = mElements[ Z ] + a[ Z ];
    result[ H ] = mElements[ H ] + a[ H ];

    return result;
}


Vector4 Vector4::operator - ( const Vector4 &a ) const
{
    Vector4 result;

    result[ X ] = mElements[ X ] - a[ X ];
    result[ Y ] = mElements[ Y ] - a[ Y ];
    result[ Z ] = mElements[ Z ] - a[ Z ];
    result[ H ] = mElements[ H ] - a[ H ];

    return result;
}


Vector4 Vector4::operator - () const
{
    Vector4 result;

    result[ X ] = - mElements[ X ];
    result[ Y ] = - mElements[ Y ];
    result[ Z ] = - mElements[ Z ];
    result[ H ] = - mElements[ H ];

    return result;
}


Vector4 Vector4::operator * ( const Vector4 &a ) const
{
    Vector4 result;

    result[ X ] = mElements[ X ] * a[ X ];
    result[ Y ] = mElements[ Y ] * a[ Y ];
    result[ Z ] = mElements[ Z ] * a[ Z ];
    result[ H ] = mElements[ H ] * a[ H ];

    return result;
}


Vector4 Vector4::operator * ( float s ) const
{
    Vector4 result;

    result[ X ] = mElements[ X ] * s;
    result[ Y ] = mElements[ Y ] * s;
    result[ Z ] = mElements[ Z ] * s;
    result[ H ] = mElements[ H ] * s;

    return result;
}


Vector4 Vector4::operator / ( const Vector4 &a ) const
{
    Vector4 result;

    result[ X ] = mElements[ X ] / a[ X ];
    result[ Y ] = mElements[ Y ] / a[ Y ];
    result[ Z ] = mElements[ Z ] / a[ Z ];
    result[ H ] = mElements[ H ] / a[ H ];

    return result;
}


Vector4 Vector4::operator / ( float s ) const
{
    Vector4 result;

    result[ X ] = mElements[ X ] / s;
    result[ Y ] = mElements[ Y ] / s;
    result[ Z ] = mElements[ Z ] / s;
    result[ H ] = mElements[ H ] / s;

    return result;
}


Vector4 operator * ( float s, const Vector4 &v )
{
    return ( v * s );
}


void Vector4::Normalize()
{
    Assert( SquaredLength() > 0.0 ); // normalizing length-zero vector

    *this /= Length();
}

Vector3 Vector4::Projection() const
{
    Vector3 result;

    Assert( mElements[3] != 0 ); // (Vec4/proj) last elt. is zero

    result[0] = mElements[0] / mElements[3];
    result[1] = mElements[1] / mElements[3];
    result[2] = mElements[2] / mElements[3];

    return result;
}

float Vector4::Length() const
{
    return sqrt( DotProduct( *this, *this ) );
}


float Vector4::SquaredLength() const
{
    return( DotProduct( *this, *this ) );
}


Vector4 CrossProduct( const Vector4& a, const Vector4& b, const Vector4& c )
{
    Vector4 result;
    // XXX can this be improved? Look at assembly.
#define ROW(i)       a[i], b[i], c[i]
#define DET(i,j,k)   DotProduct( Vector3( ROW( i ) ), CrossProduct( Vector3( ROW( j ) ), Vector3( ROW( k ) ) ) )

    result[0] =  DET(1,2,3);
    result[1] = -DET(0,2,3);
    result[2] =  DET(0,1,3);
    result[3] = -DET(0,1,2);

    return result;

#undef ROW
#undef DET
}

float DotProduct( const Vector4 &a, const Vector4 &b )
{
    return( a[ X ] * b[ X ] + a[ Y ] * b[ Y ] + a[ Z ] * b[ Z ] + a[ H ] * b[ H ] );
}

void Vector4::CheckRange( int index, int min, int max, const char* ) const
{
    Assert( index < max && index >= min );
}

}