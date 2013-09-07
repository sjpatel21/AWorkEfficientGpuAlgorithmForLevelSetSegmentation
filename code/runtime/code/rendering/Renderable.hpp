#ifndef RENDERING_RENDERABLE_HPP
#define RENDERING_RENDERABLE_HPP

#include "math/Matrix44.hpp"

#include "container/List.hpp"

#include "content/Asset.hpp"
#include "content/Ref.hpp"

#include "rendering/Material.hpp"

namespace math
{
    class Matrix44;
}

namespace rendering
{

class MaterialStrip;

class Renderable : public content::Asset
{
public:
    Renderable();

    void Render() const;
    void Render( content::Ref< Material > overrideMaterial ) const;

    void SetTransform( const math::Matrix44& transform );
    void GetTransform( math::Matrix44& transform ) const;

    const container::List< MaterialStrip* >& GetMaterialStrips() const;
    void                                     SetMaterialStrips( const container::List< MaterialStrip* >& materialStrips );

protected:
    virtual ~Renderable();

    container::List< MaterialStrip* > mMaterialStrips;
    math::Matrix44                    mTransform;
};

}

#endif