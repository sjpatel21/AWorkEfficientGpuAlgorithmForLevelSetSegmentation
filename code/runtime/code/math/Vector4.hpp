#ifndef MATH_VECTOR4_HPP
#define MATH_VECTOR4_HPP

#include "core/Assert.hpp"

#include "math/Utility.hpp"
#include "math/Vector3.hpp"

namespace math
{

class Vector4
{

public:
    // Constructors
    Vector4();
    Vector4( float x, float y, float z, float h );   // [x, y, z, h]
    Vector4( const Vector4& v );                     // Copy constructor
    Vector4( const Vector3& v, float h );

    // Accesser operators
    float&       operator [] ( int i );
    const float& operator [] ( int i ) const;

    const float* Ref() const; // get pointer to the array


    // Assignment operators
    Vector4&       operator  = ( const Vector4 &a );
    Vector4&       operator += ( const Vector4 &a );
    Vector4&       operator -= ( const Vector4 &a );
    Vector4&       operator *= ( const Vector4 &a );
    Vector4&       operator *= ( float s );          
    Vector4&       operator /= ( const Vector4 &a );
    Vector4&       operator /= ( float s );


    // Comparison operators
    bool           operator == ( const Vector4 &a ) const;   // v == a ?
    bool           operator != ( const Vector4 &a ) const;   // v != a ?


    // Arithmetic operators
    Vector4        operator +  ( const Vector4 &a ) const;   // v + a
    Vector4        operator -  ( const Vector4 &a ) const;   // v - a
    Vector4        operator -  ()                   const;   // -v
    Vector4        operator *  ( const Vector4 &a ) const;   // v * a (vx * ax, ...)
    Vector4        operator *  ( float s )          const;   // v * s
    Vector4        operator /  ( const Vector4 &a ) const;   // v / a (vx / ax, ...)
    Vector4        operator /  ( float s )          const;   // v / s

    // In-place mutators
    void           Normalize();                              // normalize vector

    // Accessor helper functions
    float          Length       () const;  // || v ||
    float          SquaredLength() const;  // v . v
    Vector3        Projection   () const;  // hom. projection

private:
    void           CheckRange( int index, int min, int max, const char* ) const;
    float          mElements[ 4 ];
};


Vector4 operator *   ( float s, const Vector4 &v );                              // Left mult. by s

float   DotProduct   ( const Vector4 &a, const Vector4 &b );                     // v . a
Vector4 CrossProduct ( const Vector4 &a, const Vector4 &b, const Vector4 &c );   // a x b x c

}

#endif