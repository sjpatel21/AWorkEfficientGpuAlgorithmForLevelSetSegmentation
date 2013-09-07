#ifndef MATH_MATRIX44
#define MATH_MATRIX44

#include "math/Vector3.hpp"
#include "math/Vector4.hpp"
#include "math/Config.hpp"

#ifdef CONVERT_TO_AND_FROM_PHYSX
class NxMat34;
#endif

#ifdef CONVERT_TO_AND_FROM_FCOLLADA
class FMMatrix44;
#endif

namespace math
{

class Matrix44
{
public:

    // Constructors
    Matrix44();
    Matrix44( float a, float b, float c, float d,
              float e, float f, float g, float h,
              float i, float j, float k, float l,
              float m, float n, float o, float p );
    Matrix44( const Matrix44& m );

#ifdef CONVERT_TO_AND_FROM_PHYSX
    Matrix44( const NxMat34& m );
#endif

#ifdef CONVERT_TO_AND_FROM_FCOLLADA
    Matrix44( const FMMatrix44& m );
#endif

    // Accessor functions
    Vector4&       operator [] ( int i );
    const Vector4& operator [] ( int i ) const;

    const float*   Ref() const;

    // Assignment operators
    Matrix44&        operator =  ( const Matrix44 &m );
    Matrix44&        operator += ( const Matrix44 &m );
    Matrix44&        operator -= ( const Matrix44 &m );
    Matrix44&        operator *= ( const Matrix44 &m );
    Matrix44&        operator *= ( float s );
    Matrix44&        operator /= ( float s );

    // Comparison operators
    bool            operator == ( const Matrix44 &m ) const;  // M == N?
    bool            operator != ( const Matrix44 &m ) const;  // M != N?

    // Arithmetic operators
    Matrix44         operator + ( const Matrix44 &m ) const;   // M + N
    Matrix44         operator - ( const Matrix44 &m ) const;   // M - N
    Matrix44         operator - () const;                      // -M
    Matrix44         operator * ( const Matrix44 &m ) const;   // M * N
    Matrix44         operator * ( float s ) const;             // M * s
    Matrix44         operator / ( float s ) const;             // M / s

    // Initializers
    void SetToZero          ();                                                                          // Zero matrix
    void SetToIdentity      ();                                                                          // I
    void SetToRotation      ( float angleRadians, const Vector3& axis );                                 // Rotate by theta radians about axis
    void SetToRotation      ( const Vector4& q );                                                        // Rotate by quaternion
    void SetToScale         ( const Vector3& s );                                                        // Scale by components of s
    void SetToTranslate     ( const Vector3& t );                                                        // Translation by t
    void SetToLookAt        ( const Vector3& eye, const Vector3& center, const Vector3& up );            // look-at matrix
    void SetToPerspective   ( float fovYRadians, float aspectRatio, float nearPlane, float farPlane );   // perspective matrix
    void SetTo2DOrthographic( float left, float right, float bottom, float top );                        // 2D orthographic matrix

    // Accessor for LookAt matrices
    void GetLookAtVectors( Vector3& eye, Vector3& targetGuess, Vector3& up ) const;

    // In place mutators
    void Transpose();
    void InvertTranspose();

    // Transform points and vectors
    Vector4     Transform      ( const Vector4& v )                  const;
    Vector3     Transform      ( const Vector3& v )                  const;
    Vector3     TransformVector( const Vector3& v )                  const;
    void        Transform      ( Matrix44& out, const Matrix44& in ) const;

    // Conversion functions
#ifdef CONVERT_TO_AND_FROM_PHYSX
    NxMat34     ToNxMat34() const;
#endif

private:
    float              Trace       ( const Matrix44 &m )                  const; // Trace
    Matrix44           Adjoint     ( const Matrix44 &m )                  const; // Adjoint
    float              Determinant ( const Matrix44 &m )                  const; // Determinant
    Matrix44           OuterProduct( const Vector4 &a, const Vector4 &b ) const; // Outer product

    void               CheckRange  ( int index, int min, int max, const char* ) const;

    Vector4 mRows[4];
};


Vector4             operator *  ( const Matrix44 &m, const Vector4 &v );  // m * v
Vector4             operator *  ( const Vector4 &v, const Matrix44 &m );  // v * m
Vector4&            operator *= ( Vector4 &a, const Matrix44 &m );        // v *= m
Matrix44            operator *  ( float s, const Matrix44 &m );           // s * m

}

#endif
