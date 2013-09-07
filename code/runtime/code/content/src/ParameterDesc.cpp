#include "content/ParameterDesc.hpp"

#include "content/ParameterType.hpp"

namespace content
{

ParameterDesc::ParameterDesc()
{
}

ParameterDesc::ParameterDesc( const ParameterDesc& p )
{
    type         = p.type;
    dataFloat    = p.dataFloat;
    dataVector3  = p.dataVector3;
    dataVector4  = p.dataVector4;
    dataMatrix44  = p.dataMatrix44;
    minimumValue = p.minimumValue;
    maximumValue = p.maximumValue;
}


ParameterDesc& ParameterDesc::operator = ( const ParameterDesc &p )
{
    type        = p.type;
    dataFloat   = p.dataFloat;
    dataVector3 = p.dataVector3;
    dataVector4 = p.dataVector4;
    dataMatrix44 = p.dataMatrix44;
    minimumValue = p.minimumValue;
    maximumValue = p.maximumValue;

    return *this;
}


ParameterDesc::ParameterDesc( float data, float min, float max )
{
    Assert( data <= max && data >= min );

    type         = ParameterType_Float;
    dataFloat    = data;

    minimumValue = min;
    maximumValue = max;
}

ParameterDesc::ParameterDesc( const math::Vector3& data, float min, float max )
{
    Assert( data[ math::X ] <= max && data[ math::X ] >= min && 
        data[ math::Y ] <= max && data[ math::Y ] >= min &&
        data[ math::Z ] <= max && data[ math::Z ] >= min );

    type         = ParameterType_Vector3;
    dataVector3  = data;

    minimumValue = min;
    maximumValue = max;
}

ParameterDesc::ParameterDesc( const math::Vector4& data, float min, float max )
{
    Assert( data[ math::X ] <= max && data[ math::X ] >= min && 
        data[ math::Y ] <= max && data[ math::Y ] >= min &&
        data[ math::Z ] <= max && data[ math::Z ] >= min &&
        data[ math::H ] <= max && data[ math::H ] >= min );

    type         = ParameterType_Vector4;
    dataVector4  = data;

    minimumValue = min;
    maximumValue = max;
}

ParameterDesc::ParameterDesc( const math::Matrix44& data, float min, float max )
{
    Assert( 0 );

    type         = ParameterType_Matrix44;
    dataMatrix44  = data;

    minimumValue = min;
    maximumValue = max;
}


}