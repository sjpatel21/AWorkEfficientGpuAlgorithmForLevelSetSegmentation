#include "rendering/Camera.hpp"

namespace rendering
{

Camera::Camera() :
mFovYRadians( 0.0f ),
mAspectRatio( 0.0f ),
mNearPlane  ( 0.0f ),
mFarPlane   ( 0.0f )
{
}

Camera::Camera( const math::Vector3& position, const math::Vector3& target, const math::Vector3& upHint ) :
mPosition   ( position ),
mTarget     ( target ),
mFovYRadians( 0.0f ),
mAspectRatio( 0.0f ),
mNearPlane  ( 0.0f ),
mFarPlane   ( 0.0f )
{
    // compute the stored up vector to be orthogonal to the other vectors    
    mUp = math::CrossProduct( math::CrossProduct( mTarget - mPosition, upHint ), mTarget - mPosition );
    mUp.Normalize();

    mLookAtMatrix.SetToLookAt( mPosition, mTarget, mUp );
}

Camera::~Camera()
{
}

const math::Vector3& Camera::GetPosition() const
{
    return mPosition;
}

void Camera::GetLookAtVectors( math::Vector3& position, math::Vector3& target, math::Vector3& up ) const
{
    position = mPosition;
    target   = mTarget;
    up       = mUp;
}

void Camera::SetLookAtVectors( const math::Vector3& position, const math::Vector3& target, const math::Vector3& upHint )
{
    mPosition = position;
    mTarget   = target;

    // compute the stored up vector to be orthogonal to the other vectors
    mUp = math::CrossProduct( math::CrossProduct( mTarget - mPosition, upHint ), mTarget - mPosition );
    mUp.Normalize();

    mLookAtMatrix.SetToLookAt( mPosition, mTarget, mUp );
}

void Camera::SetProjectionParameters( float fovYRadians, float aspectRatio, float nearPlane, float farPlane )
{
    mFovYRadians = fovYRadians;
    mAspectRatio = aspectRatio;
    mNearPlane   = nearPlane;
    mFarPlane    = farPlane;

    mProjectionMatrix.SetToPerspective( fovYRadians, aspectRatio, nearPlane, farPlane );
}

void Camera::GetProjectionParameters( float& fovYRadians, float& aspectRatio, float& nearPlane, float& farPlane )
{
    Assert( mFovYRadians != 0.0f );
    Assert( mAspectRatio != 0.0f );
    Assert( mNearPlane   != 0.0f );
    Assert( mFarPlane    != 0.0f );

    fovYRadians = mFovYRadians;
    aspectRatio = mAspectRatio;
    nearPlane   = mNearPlane;
    farPlane    = mFarPlane;
}

void Camera::GetLookAtMatrix( math::Matrix44& lookAtMatrix ) const
{
    lookAtMatrix = mLookAtMatrix;
}

void Camera::SetLookAtMatrix( const math::Matrix44& lookAtMatrix )
{
    mLookAtMatrix = lookAtMatrix;
    mLookAtMatrix.GetLookAtVectors( mPosition, mTarget, mUp );
}

void Camera::GetProjectionMatrix( math::Matrix44& projectionMatrix ) const
{
    projectionMatrix = mProjectionMatrix;
}

void Camera::SetProjectionMatrix( const math::Matrix44& projectionMatrix )
{
    mFovYRadians = 0.0f;
    mAspectRatio = 0.0f;
    mNearPlane   = 0.0f;
    mFarPlane    = 0.0f;

    mProjectionMatrix = projectionMatrix;
}

}