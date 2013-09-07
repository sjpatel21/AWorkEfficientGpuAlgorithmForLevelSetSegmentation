#include "rendering/Renderable.hpp"

#include "core/Functor.hpp"

#include "rendering/MaterialStrip.hpp"
#include "rendering/Context.hpp"

namespace rendering
{

Renderable::Renderable()
{
    mTransform.SetToIdentity();
}

Renderable::~Renderable()
{
    foreach ( MaterialStrip* materialStrip, mMaterialStrips )
    {
        materialStrip->Release();
    }
}


void Renderable::Render() const
{
    Context::SetCurrentTransformMatrix( mTransform );

    foreach ( MaterialStrip* materialStrip, mMaterialStrips )
    {
        materialStrip->Render();
    }
}

void Renderable::Render( content::Ref< Material > overrideMaterial ) const
{
    Context::SetCurrentTransformMatrix( mTransform );

    foreach ( MaterialStrip* materialStrip, mMaterialStrips )
    {
        materialStrip->Render( overrideMaterial );
    }
}

void Renderable::SetTransform( const math::Matrix44& transform )
{
    mTransform = transform;
}

void Renderable::GetTransform( math::Matrix44& transform ) const
{
    transform = mTransform;
}

void Renderable::SetMaterialStrips( const container::List< MaterialStrip* >& materialStrips )
{
    Assert( !IsFinalized() );
    
    mMaterialStrips = materialStrips;

    foreach ( MaterialStrip* materialStrip, mMaterialStrips )
    {
        materialStrip->AddRef();
    }
}

const container::List< MaterialStrip* >& Renderable::GetMaterialStrips() const
{
    return mMaterialStrips;
}

}
