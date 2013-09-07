#include "rendering/Scene.hpp"

#include "rendering/rtgi/Config.hpp"

#ifdef CAPS_BASIC_EXTENSIONS_ONLY
#include "rendering/Context.hpp"
#endif

namespace rendering
{

Scene::Scene()
{
}

Scene::~Scene()
{
    foreach ( content::Ref< Renderable > renderable, mRenderables )
    {
        renderable->Release();
    }
}

void Scene::Render() const
{
#ifdef CAPS_BASIC_EXTENSIONS_ONLY
    Render( Context::GetDebugMaterial() );
#else
    foreach ( content::Ref< Renderable > renderable, mRenderables )
    {
        renderable->Render();
    }
#endif

}

void Scene::Render( content::Ref< Material > overrideMaterial ) const
{
    foreach ( content::Ref< Renderable > renderable, mRenderables )
    {
#ifdef CAPS_BASIC_EXTENSIONS_ONLY
        renderable->Render( Context::GetDebugMaterial() );
#else
        renderable->Render( overrideMaterial );
#endif
    }
}

const container::List< content::Ref< Renderable > >& Scene::GetRenderables() const
{
    return mRenderables;
}

void Scene::SetRenderables( container::List< content::Ref< Renderable > >& renderables )
{
    Assert( !IsFinalized() );

    // release the old renderables
    foreach ( content::Ref< Renderable > renderable, mRenderables )
    {
        renderable->Release();
    }

    // assign new renderables
    mRenderables = renderables;

    // now addref the new renderables
    foreach ( content::Ref< Renderable > renderable, mRenderables )
    {
        renderable->AddRef();
    }
}

}