#include "rendering/renderstrategies/RenderStrategy.hpp"

#include "core/Assert.hpp"
#include "core/String.hpp"
#include "core/Functor.hpp"

#include "rendering/DebugDraw.hpp"
#include "rendering/TextConsole.hpp"

#include "rendering/rtgi/RTGI.hpp"
#include "rendering/rtgi/Color.hpp"

namespace rendering
{

RenderStrategy::RenderStrategy() :
mRenderCallback( NULL ),
mClearColor    ( rtgi::ColorRGBA( 0, 0, 0, 1 ) )
{
}

RenderStrategy::~RenderStrategy()
{
    AssignRef( mRenderCallback, NULL );
}

void RenderStrategy::Update( content::Ref< Scene > scene, content::Ref< Camera > camera, double timeDeltaSeconds )
{
}

void RenderStrategy::Render( content::Ref< Scene > scene, content::Ref< Camera > camera )
{
    rtgi::ClearColorBuffer( GetClearColor() );
    rtgi::ClearDepthBuffer();
    rtgi::ClearStencilBuffer();

    scene->Render();

    CallRenderCallback( scene );
}

rtgi::ColorRGBA RenderStrategy::GetClearColor() const
{
    return mClearColor;
}

void RenderStrategy::SetClearColor( const rtgi::ColorRGBA& clearColor )
{
    mClearColor = clearColor;
}

core::IFunctor* RenderStrategy::GetRenderCallback() const
{
    return mRenderCallback;
}

void RenderStrategy::SetRenderCallback( core::IFunctor* renderCallback )
{
    AssignRef( mRenderCallback, renderCallback );
}

void RenderStrategy::CallRenderCallback( content::Ref< Scene > scene )
{
    if ( GetRenderCallback() != NULL )
    {
        if ( !RefIsNull( scene ) )
        {
            GetRenderCallback()->Call( reinterpret_cast< void* >( const_cast< Scene* >( scene.GetRawPointer() ) ) );
        }
        else
        {
            GetRenderCallback()->Call( NULL );
        }
    } 
}

}