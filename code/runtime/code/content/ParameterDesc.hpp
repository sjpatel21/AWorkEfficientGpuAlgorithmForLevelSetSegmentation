#ifndef CONTENT_PARAMETER_DESC_HPP
#define CONTENT_PARAMETER_DESC_HPP

#include "math/Vector3.hpp"
#include "math/Vector4.hpp"
#include "math/Matrix44.hpp"

#include "content/ParameterType.hpp"

namespace content
{

struct ParameterDesc
{
    ParameterDesc();
    ParameterDesc( const ParameterDesc& p );

    ParameterDesc( const math::Matrix44& data, float min, float max );
    ParameterDesc( const math::Vector4&  data, float min, float max );
    ParameterDesc( const math::Vector3&  data, float min, float max );
    ParameterDesc( float                 data, float min, float max );

    ParameterDesc& operator = ( const ParameterDesc &p );

    ParameterType type;

    math::Matrix44 dataMatrix44;
    math::Vector4  dataVector4;
    math::Vector3  dataVector3;
    float          dataFloat;

    float         minimumValue;
    float         maximumValue;
};

}

#endif