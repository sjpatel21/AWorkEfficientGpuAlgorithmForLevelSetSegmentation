#ifndef RENDERING_CAMERA_HPP
#define RENDERING_CAMERA_HPP

#include "content/Asset.hpp"

#include "math/Vector3.hpp"
#include "math/Matrix44.hpp"


namespace rendering
{

class Camera : public content::Asset
{
public:
    Camera();
    Camera( const math::Vector3& position, const math::Vector3& target, const math::Vector3& upHint );
    virtual ~Camera();

    const math::Vector3& GetPosition() const;

    void GetLookAtVectors( math::Vector3& position, math::Vector3& target, math::Vector3& up ) const;
    void SetLookAtVectors( const math::Vector3& position, const math::Vector3& target, const math::Vector3& upHint );

    void GetLookAtMatrix( math::Matrix44& lookAtMatrix ) const;
    void SetLookAtMatrix( const math::Matrix44& lookAtMatrix );

    void GetProjectionMatrix( math::Matrix44& projectionMatrix ) const;
    void SetProjectionMatrix( const math::Matrix44& projectionMatrix );

    void GetProjectionParameters( float& fovYRadians, float& aspectRatio, float& nearPlane, float& farPlane );
    void SetProjectionParameters( float fovYRadians, float aspectRatio, float nearPlane, float farPlane );
private:
    math::Matrix44 mProjectionMatrix;
    math::Matrix44 mLookAtMatrix;
    math::Vector3  mPosition, mTarget, mUp;
    float          mFovYRadians, mAspectRatio, mNearPlane, mFarPlane;
};

}

#endif