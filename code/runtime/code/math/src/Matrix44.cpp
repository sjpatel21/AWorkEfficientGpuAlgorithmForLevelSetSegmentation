#include "math/Matrix44.hpp"

#ifdef CONVERT_TO_AND_FROM_PHYSX
#include <NxPhysics.h>
#endif

#ifdef CONVERT_TO_AND_FROM_FCOLLADA
#include <FCollada.h>
#include <FMath/FMMatrix44.h>
#endif

#include "core/Assert.hpp"

namespace math
{

Matrix44::Matrix44()
{
}

Vector4& Matrix44::operator [] ( int i )
{
    CheckRange(i, 0, 4, "(Matrix44::[i]) index out of range");
    return mRows[i];
}

const Vector4& Matrix44::operator [] ( int i ) const
{
    CheckRange(i, 0, 4, "(Matrix44::[i]) index out of range");
    return mRows[i];
}

const float* Matrix44::Ref() const
{
    return reinterpret_cast< const float* >( mRows );
}


Matrix44 operator * ( float s, const Matrix44& m )
{
    return(m * s);
}

Matrix44::Matrix44( float a, float b, float c, float d,
                  float e, float f, float g, float h,
                  float i, float j, float k, float l,
                  float m, float n, float o, float p )
{
    mRows[0][0] = a;  mRows[0][1] = b;  mRows[0][2] = c;  mRows[0][3] = d;
    mRows[1][0] = e;  mRows[1][1] = f;  mRows[1][2] = g;  mRows[1][3] = h;
    mRows[2][0] = i;  mRows[2][1] = j;  mRows[2][2] = k;  mRows[2][3] = l;
    mRows[3][0] = m;  mRows[3][1] = n;  mRows[3][2] = o;  mRows[3][3] = p;
}

Matrix44::Matrix44( const Matrix44 &m )
{
    mRows[0] = m[0];
    mRows[1] = m[1];
    mRows[2] = m[2];
    mRows[3] = m[3];
}

#ifdef CONVERT_TO_AND_FROM_PHYSX
Matrix44::Matrix44( const NxMat34& m )
{
    NxReal mRowMajor44[ 4 ][ 4 ];

    m.getRowMajor44( mRowMajor44 );

    mRows[0][0] = mRowMajor44[0][0];  mRows[0][1] = mRowMajor44[0][1];  mRows[0][2] = mRowMajor44[0][2];  mRows[0][3] = mRowMajor44[0][3];
    mRows[1][0] = mRowMajor44[1][0];  mRows[1][1] = mRowMajor44[1][1];  mRows[1][2] = mRowMajor44[1][2];  mRows[1][3] = mRowMajor44[1][3];
    mRows[2][0] = mRowMajor44[2][0];  mRows[2][1] = mRowMajor44[2][1];  mRows[2][2] = mRowMajor44[2][2];  mRows[2][3] = mRowMajor44[2][3];
    mRows[3][0] = mRowMajor44[3][0];  mRows[3][1] = mRowMajor44[3][1];  mRows[3][2] = mRowMajor44[3][2];  mRows[3][3] = mRowMajor44[3][3];
}
#endif

#ifdef CONVERT_TO_AND_FROM_FCOLLADA
Matrix44::Matrix44( const FMMatrix44& m )
{
    mRows[0][0] = m[0][0];  mRows[0][1] = m[0][1];  mRows[0][2] = m[0][2];  mRows[0][3] = m[0][3];
    mRows[1][0] = m[1][0];  mRows[1][1] = m[1][1];  mRows[1][2] = m[1][2];  mRows[1][3] = m[1][3];
    mRows[2][0] = m[2][0];  mRows[2][1] = m[2][1];  mRows[2][2] = m[2][2];  mRows[2][3] = m[2][3];
    mRows[3][0] = m[3][0];  mRows[3][1] = m[3][1];  mRows[3][2] = m[3][2];  mRows[3][3] = m[3][3];

    Transpose();
}
#endif

Matrix44& Matrix44::operator = ( const Matrix44 &m )
{
    mRows[0] = m[0];
    mRows[1] = m[1];
    mRows[2] = m[2];
    mRows[3] = m[3];

    return *this;
}

Matrix44& Matrix44::operator += ( const Matrix44 &m )
{
    mRows[0] += m[0];
    mRows[1] += m[1];
    mRows[2] += m[2];
    mRows[3] += m[3];

    return *this;
}

Matrix44& Matrix44::operator -= ( const Matrix44 &m )
{
    mRows[0] -= m[0];
    mRows[1] -= m[1];
    mRows[2] -= m[2];
    mRows[3] -= m[3];

    return *this;
}

Matrix44& Matrix44::operator *= ( const Matrix44 &m )
{
    *this = *this * m;

    return *this;
}

Matrix44& Matrix44::operator *= ( float s )
{
    mRows[0] *= s;
    mRows[1] *= s;
    mRows[2] *= s;
    mRows[3] *= s;

    return *this;
}

Matrix44& Matrix44::operator /= ( float s )
{
    mRows[0] /= s;
    mRows[1] /= s;
    mRows[2] /= s;
    mRows[3] /= s;

    return *this;
}


bool Matrix44::operator == ( const Matrix44 &m ) const
{
    return( mRows[0] == m[0] && mRows[1] == m[1] && mRows[2] == m[2] && mRows[3] == m[3] );
}

bool Matrix44::operator != ( const Matrix44 &m ) const
{
    return( mRows[0] != m[0] || mRows[1] != m[1] || mRows[2] != m[2] || mRows[3] != m[3] );
}


Matrix44 Matrix44::operator + ( const Matrix44 &m ) const
{
    Matrix44 result;

    result[0] = mRows[0] + m[0];
    result[1] = mRows[1] + m[1];
    result[2] = mRows[2] + m[2];
    result[3] = mRows[3] + m[3];

    return result;
}


Matrix44 Matrix44::operator - ( const Matrix44 &m ) const
{
    Matrix44 result;

    result[0] = mRows[0] - m[0];
    result[1] = mRows[1] - m[1];
    result[2] = mRows[2] - m[2];
    result[3] = mRows[3] - m[3];

    return result;
}


Matrix44 Matrix44::operator - () const
{
    Matrix44 result;

    result[0] = -mRows[0];
    result[1] = -mRows[1];
    result[2] = -mRows[2];
    result[3] = -mRows[3];

    return result;
}


Matrix44 Matrix44::operator * ( const Matrix44 &m ) const
{
#define N(x,y) mRows[x][y]
#define M(x,y) m[x][y]
#define R(x,y) result[x][y]

    Matrix44 result;

    R(0,0) = N(0,0) * M(0,0) + N(0,1) * M(1,0) + N(0,2) * M(2,0) + N(0,3) * M(3,0);
    R(0,1) = N(0,0) * M(0,1) + N(0,1) * M(1,1) + N(0,2) * M(2,1) + N(0,3) * M(3,1);
    R(0,2) = N(0,0) * M(0,2) + N(0,1) * M(1,2) + N(0,2) * M(2,2) + N(0,3) * M(3,2);
    R(0,3) = N(0,0) * M(0,3) + N(0,1) * M(1,3) + N(0,2) * M(2,3) + N(0,3) * M(3,3);

    R(1,0) = N(1,0) * M(0,0) + N(1,1) * M(1,0) + N(1,2) * M(2,0) + N(1,3) * M(3,0);
    R(1,1) = N(1,0) * M(0,1) + N(1,1) * M(1,1) + N(1,2) * M(2,1) + N(1,3) * M(3,1);
    R(1,2) = N(1,0) * M(0,2) + N(1,1) * M(1,2) + N(1,2) * M(2,2) + N(1,3) * M(3,2);
    R(1,3) = N(1,0) * M(0,3) + N(1,1) * M(1,3) + N(1,2) * M(2,3) + N(1,3) * M(3,3);

    R(2,0) = N(2,0) * M(0,0) + N(2,1) * M(1,0) + N(2,2) * M(2,0) + N(2,3) * M(3,0);
    R(2,1) = N(2,0) * M(0,1) + N(2,1) * M(1,1) + N(2,2) * M(2,1) + N(2,3) * M(3,1);
    R(2,2) = N(2,0) * M(0,2) + N(2,1) * M(1,2) + N(2,2) * M(2,2) + N(2,3) * M(3,2);
    R(2,3) = N(2,0) * M(0,3) + N(2,1) * M(1,3) + N(2,2) * M(2,3) + N(2,3) * M(3,3);

    R(3,0) = N(3,0) * M(0,0) + N(3,1) * M(1,0) + N(3,2) * M(2,0) + N(3,3) * M(3,0);
    R(3,1) = N(3,0) * M(0,1) + N(3,1) * M(1,1) + N(3,2) * M(2,1) + N(3,3) * M(3,1);
    R(3,2) = N(3,0) * M(0,2) + N(3,1) * M(1,2) + N(3,2) * M(2,2) + N(3,3) * M(3,2);
    R(3,3) = N(3,0) * M(0,3) + N(3,1) * M(1,3) + N(3,2) * M(2,3) + N(3,3) * M(3,3);

    return result;

#undef N
#undef M
#undef R
}


Matrix44 Matrix44::operator * ( float s ) const
{
    Matrix44 result;

    result[0] = mRows[0] * s;
    result[1] = mRows[1] * s;
    result[2] = mRows[2] * s;
    result[3] = mRows[3] * s;

    return result;
}


Matrix44 Matrix44::operator / ( float s ) const
{
    Matrix44 result;

    result[0] = mRows[0] / s;
    result[1] = mRows[1] / s;
    result[2] = mRows[2] / s;
    result[3] = mRows[3] / s;

    return result;
}


Vector4 operator * ( const Matrix44 &m, const Vector4 &v )          // m * v
{
    Vector4 result;

    result[0] = v[0] * m[0][0] + v[1] * m[0][1] + v[2] * m[0][2] + v[3] * m[0][3];
    result[1] = v[0] * m[1][0] + v[1] * m[1][1] + v[2] * m[1][2] + v[3] * m[1][3];
    result[2] = v[0] * m[2][0] + v[1] * m[2][1] + v[2] * m[2][2] + v[3] * m[2][3];
    result[3] = v[0] * m[3][0] + v[1] * m[3][1] + v[2] * m[3][2] + v[3] * m[3][3];

    return result;
}


Vector4 operator * ( const Vector4 &v, const Matrix44 &m )          // v * m
{
    Vector4 result;

    result[0] = v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0] + v[3] * m[3][0];
    result[1] = v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1] + v[3] * m[3][1];
    result[2] = v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2] + v[3] * m[3][2];
    result[3] = v[0] * m[0][3] + v[1] * m[1][3] + v[2] * m[2][3] + v[3] * m[3][3];

    return result;
}


Vector4& operator *= ( Vector4 &v, const Matrix44 &m )              // v *= m
{
    float    t0, t1, t2;

    t0   = v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0] + v[3] * m[3][0];
    t1   = v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1] + v[3] * m[3][1];
    t2   = v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2] + v[3] * m[3][2];
    v[3] = v[0] * m[0][3] + v[1] * m[1][3] + v[2] * m[2][3] + v[3] * m[3][3];
    v[0] = t0;
    v[1] = t1;
    v[2] = t2;

    return v;
}

void Matrix44::SetToZero()
{
    int i;

    for ( i = 0; i < 16; i++ )
        reinterpret_cast< float* >( mRows )[i] = 0;
}


void Matrix44::SetToIdentity()
{
    int i, j;

    for ( i = 0; i < 4; i++ )
        for ( j = 0; j < 4; j++ )
            if ( i == j )
                mRows[i][j] = 1;
            else
                mRows[i][j] = 0;
}


void Matrix44::SetToRotation( const Vector4& q )
{
    float   i2 =  2 * q[0],
            j2 =  2 * q[1],
            k2 =  2 * q[2],
            ij = i2 * q[1],
            ik = i2 * q[2],
            jk = j2 * q[2],
            ri = i2 * q[3],
            rj = j2 * q[3],
            rk = k2 * q[3];

    SetToIdentity();

    i2 *= q[0];
    j2 *= q[1];
    k2 *= q[2];

    mRows[0][0] = 1 - j2 - k2;  mRows[0][1] = ij - rk   ;  mRows[0][2] = ik + rj;
    mRows[1][0] = ij + rk    ;  mRows[1][1] = 1 - i2- k2;  mRows[1][2] = jk - ri;
    mRows[2][0] = ik - rj    ;  mRows[2][1] = jk + ri   ;  mRows[2][2] = 1 - i2 - j2;
}


void Matrix44::SetToRotation( float angleRadians, const Vector3& axis )
{
    float        s;
    Vector4      q;

    angleRadians /= 2.0;
    s             = sin( angleRadians );

    q[0] = s * axis[0];
    q[1] = s * axis[1];
    q[2] = s * axis[2];
    q[3] = cos( angleRadians );

    SetToRotation( q );
}


void Matrix44::SetToScale( const Vector3 &s )
{
    SetToIdentity();

    mRows[0][0] = s[0];
    mRows[1][1] = s[1];
    mRows[2][2] = s[2];
}


void Matrix44::SetToTranslate( const Vector3 &t )
{
    SetToIdentity();

    mRows[0][3] = t[0];
    mRows[1][3] = t[1];
    mRows[2][3] = t[2];
}

void Matrix44::SetToLookAt( const Vector3& eye, const Vector3& center, const Vector3& up )
{
    Matrix44 lookAtMatrix;

    //
    // variable names taken from the openGL specification for gluLookAt( ... )
    //
    Vector3 normalizedForward = center - eye;
    normalizedForward.Normalize();

    Vector3 normalizedUp = Vector3( up );
    normalizedUp.Normalize();

    Vector3 normalizedSide = CrossProduct( normalizedForward, normalizedUp );
    normalizedSide.Normalize();

    // recalculate up vector
    normalizedUp = CrossProduct( normalizedSide, normalizedForward );

    lookAtMatrix[0][0] = normalizedSide[0];
    lookAtMatrix[0][1] = normalizedSide[1];
    lookAtMatrix[0][2] = normalizedSide[2];
    lookAtMatrix[0][3] = 0;

    lookAtMatrix[1][0] = normalizedUp[0];
    lookAtMatrix[1][1] = normalizedUp[1];
    lookAtMatrix[1][2] = normalizedUp[2];
    lookAtMatrix[1][3] = 0;

    lookAtMatrix[2][0] = - normalizedForward[0];
    lookAtMatrix[2][1] = - normalizedForward[1];
    lookAtMatrix[2][2] = - normalizedForward[2];
    lookAtMatrix[2][3] = 0;

    lookAtMatrix[3][0] = 0;
    lookAtMatrix[3][1] = 0;
    lookAtMatrix[3][2] = 0;
    lookAtMatrix[3][3] = 1;

    Matrix44 translationMatrix;
    translationMatrix.SetToIdentity();
    translationMatrix.SetToTranslate( - eye );

    lookAtMatrix = lookAtMatrix * translationMatrix;

    *this = lookAtMatrix;
}

void Matrix44::SetToPerspective( float fovYRadians, float aspectRatio, float nearPlane, float farPlane )
{
    Matrix44 perspectiveMatrix;

    //
    // taken from the opengl spec...
    //
    float f = 1.0f / ( tan( fovYRadians / 2 ) );

    perspectiveMatrix[0][0] = f / aspectRatio;
    perspectiveMatrix[0][1] = 0;
    perspectiveMatrix[0][2] = 0;
    perspectiveMatrix[0][3] = 0;

    perspectiveMatrix[1][0] = 0;
    perspectiveMatrix[1][1] = f;
    perspectiveMatrix[1][2] = 0;
    perspectiveMatrix[1][3] = 0;

    perspectiveMatrix[2][0] = 0;
    perspectiveMatrix[2][1] = 0;
    perspectiveMatrix[2][2] = ( farPlane + nearPlane ) / ( nearPlane - farPlane );
    perspectiveMatrix[2][3] = ( 2 * nearPlane * farPlane ) / ( nearPlane - farPlane );

    perspectiveMatrix[3][0] = 0;
    perspectiveMatrix[3][1] = 0;
    perspectiveMatrix[3][2] = -1;
    perspectiveMatrix[3][3] = 0;

    *this = perspectiveMatrix;
}

void Matrix44::SetTo2DOrthographic( float left, float right, float bottom, float top )
{
    Matrix44 orthographicMatrix2D;

    //
    // taken from the opengl spec...
    //
    orthographicMatrix2D[0][0] = 2 / ( right - left );
    orthographicMatrix2D[0][1] = 0;
    orthographicMatrix2D[0][2] = 0;
    orthographicMatrix2D[0][3] = - ( right + left ) / ( right - left );

    orthographicMatrix2D[1][0] = 0;
    orthographicMatrix2D[1][1] = 2 / ( top - bottom );
    orthographicMatrix2D[1][2] = 0;
    orthographicMatrix2D[1][3] = - ( top + bottom ) / (top - bottom );

    orthographicMatrix2D[2][0] = 0;
    orthographicMatrix2D[2][1] = 0;
    orthographicMatrix2D[2][2] = -1;
    orthographicMatrix2D[2][3] = 0;

    orthographicMatrix2D[3][0] = 0;
    orthographicMatrix2D[3][1] = 0;
    orthographicMatrix2D[3][2] = 0;
    orthographicMatrix2D[3][3] = 1;

    *this = orthographicMatrix2D;
}

void Matrix44::GetLookAtVectors( Vector3& eye, Vector3& targetGuess, Vector3& up ) const
{
    math::Matrix44 lookAtInverseTranspose = *this;
    lookAtInverseTranspose.InvertTranspose();

    eye[ X ] = lookAtInverseTranspose[ 0 ][ 3 ];
    eye[ Y ] = lookAtInverseTranspose[ 1 ][ 3 ];
    eye[ Z ] = lookAtInverseTranspose[ 2 ][ 3 ];

    up[ X ] = mRows[ 1 ][ X ];
    up[ Y ] = mRows[ 1 ][ Y ];
    up[ Z ] = mRows[ 1 ][ Z ];

    math::Vector3 forward;

    forward[ X ] = - mRows[ 2 ][ X ];
    forward[ Y ] = - mRows[ 2 ][ Y ];
    forward[ Z ] = - mRows[ 2 ][ Z ];

    forward.Normalize();

    targetGuess = eye + forward;
}

void Matrix44::Transpose()
{
#define M(x,y) mRows[x][y]
#define R(x,y) result[x][y]

    Matrix44 result;

    R(0,0) = M(0,0); R(0,1) = M(1,0); R(0,2) = M(2,0); R(0,3) = M(3,0);
    R(1,0) = M(0,1); R(1,1) = M(1,1); R(1,2) = M(2,1); R(1,3) = M(3,1);
    R(2,0) = M(0,2); R(2,1) = M(1,2); R(2,2) = M(2,2); R(2,3) = M(3,2);
    R(3,0) = M(0,3); R(3,1) = M(1,3); R(3,2) = M(2,3); R(3,3) = M(3,3);

    *this = result;

#undef M
#undef R
}

void Matrix44::InvertTranspose()
{
    float     det;
    Matrix44  adjoint;
    Matrix44  result;

    adjoint = Adjoint( *this );
    det     = DotProduct( adjoint[0], mRows[0] );

    Assert( det != 0 ); // (Matrix44::inv) matrix is non-singular

    result  = adjoint;
    result.Transpose();

    result /= det;

    *this = result;
}

Vector3 Matrix44::Transform( const Vector3 &v ) const
{
    Vector4 transformedV = *this * Vector4( v[0], v[1], v[2], 1.0 );
    return transformedV.Projection();
}

Vector4 Matrix44::Transform( const Vector4 &v ) const
{
    return *this * v;
}

Vector3 Matrix44::TransformVector( const Vector3& v ) const
{
    Vector3 origin( 0, 0, 0 );

    return Transform( v ) - Transform( origin );
}

void Matrix44::Transform( Matrix44 &out, const Matrix44 &in ) const
{
    out = *this * in;
}

#ifdef CONVERT_TO_AND_FROM_PHYSX
NxMat34 Matrix44::ToNxMat34() const
{
    Vector3 rotationRow0( mRows[0][0], mRows[0][1], mRows[0][2] );  
    Vector3 rotationRow1( mRows[1][0], mRows[1][1], mRows[1][2] );  
    Vector3 rotationRow2( mRows[2][0], mRows[2][1], mRows[2][2] );  
    Vector3 translation ( mRows[0][3], mRows[1][3], mRows[2][3] );

    NxMat33 physXMat33( rotationRow0.ToNxVec3(), rotationRow1.ToNxVec3(), rotationRow2.ToNxVec3() );
    NxMat34 physXMat34( physXMat33, translation.ToNxVec3() );

    return physXMat34;
}
#endif

// returns outer product of a and b:  a * trans(b)
Matrix44 Matrix44::OuterProduct( const Vector4 &a, const Vector4 &b ) const
{
    Matrix44    result;

    result[0] = a[0] * b;
    result[1] = a[1] * b;
    result[2] = a[2] * b;
    result[3] = a[3] * b;

    return result;
}

Matrix44 Matrix44::Adjoint( const Matrix44 &m ) const
{
    Matrix44    result;

    result[0] =  CrossProduct( m[1], m[2], m[3] );
    result[1] = -CrossProduct( m[0], m[2], m[3] );
    result[2] =  CrossProduct( m[0], m[1], m[3] );
    result[3] = -CrossProduct( m[0], m[1], m[2] );

    return result;
}

float Matrix44::Trace( const Matrix44 &m ) const
{
    return( m[0][0] + m[1][1] + m[2][2] + m[3][3] );
}

float Matrix44::Determinant( const Matrix44 &m ) const
{
    return DotProduct( m[0], CrossProduct( m[1], m[2], m[3] ) );
}

void Matrix44::CheckRange( int index, int min, int max, const char* ) const
{
    Assert( index < max && index >= min );
}

}