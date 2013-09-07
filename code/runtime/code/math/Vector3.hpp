#ifndef MATH_VECTOR3_HPP
#define MATH_VECTOR3_HPP

#include "math/Config.hpp"

#ifdef CONVERT_TO_AND_FROM_FCOLLADA
#include <FCollada.h>
#include <FMath/FMVector3.h>
#endif

#include "core/Assert.hpp"

#include "math/Constants.hpp"
#include "math/Utility.hpp"

namespace math
{

class Vector3
{

public:
    // Constructors
    Vector3();
    Vector3( float x, float y, float z );   // [x, y, z]
    Vector3( const Vector3& v );                             // Copy constructor

#ifdef CONVERT_TO_AND_FROM_FCOLLADA
    Vector3( const FMVector3& v );
#endif

    // Accesser operators
    float&       operator [] ( int i );
    const float& operator [] ( int i ) const;

    const float* Ref() const; // get pointer to the array

    // Assignment operators
    Vector3&       operator  = ( const Vector3 &a );
    Vector3&       operator += ( const Vector3 &a );
    Vector3&       operator -= ( const Vector3 &a );
    Vector3&       operator *= ( const Vector3 &a );
    Vector3&       operator *= ( float s );          
    Vector3&       operator /= ( const Vector3 &a );
    Vector3&       operator /= ( float s );


    // Comparison operators
    bool           operator == ( const Vector3 &a ) const;   // v == a ?
    bool           operator != ( const Vector3 &a ) const;   // v != a ?


    // Arithmetic operators
    Vector3        operator +  ( const Vector3 &a ) const;   // v + a
    Vector3        operator -  ( const Vector3 &a ) const;   // v - a
    Vector3        operator -  ()                   const;   // -v
    Vector3        operator *  ( const Vector3 &a ) const;   // v * a (vx * ax, ...)
    Vector3        operator *  ( float s )          const;   // v * s
    Vector3        operator /  ( const Vector3 &a ) const;   // v / a (vx / ax, ...)
    Vector3        operator /  ( float s )          const;   // v / s

    // In-place mutators
    void           Normalize();                              // normalize vector

    // Accessor helper methods
    float          Length()        const;
    float          SquaredLength() const;

private:
    void           CheckRange( int index, int min, int max, const char* ) const;
    float          mElements[ 3 ];
};

Vector3 operator *  ( float s, const Vector3 &v );            // Left mult. by s
float   DotProduct  ( const Vector3 &a, const Vector3 &b );
Vector3 CrossProduct( const Vector3 &a, const Vector3 &b );


}

#endif
