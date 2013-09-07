#ifndef RENDERING_RENDERSTRATEGIES_RENDERSTRATEGY_HPP
#define RENDERING_RENDERSTRATEGIES_RENDERSTRATEGY_HPP

#include "core/RefCounted.hpp"
#include "core/Functor.hpp"

#include "content/Ref.hpp"

#include "rendering/rtgi/Color.hpp"

#include "rendering/Scene.hpp"

namespace core
{
    class IFunctor;
    class String;
}

namespace rendering
{

class Camera;

class RenderStrategy : public core::RefCounted
{

public:
    RenderStrategy();
    virtual ~RenderStrategy();

    virtual void Update( content::Ref< Scene > scene, content::Ref< Camera > camera, double timeDeltaSeconds );
    virtual void Render( content::Ref< Scene > scene, content::Ref< Camera > camera );

    core::IFunctor* GetRenderCallback() const                           CORE_GET_CALLBACK_METHOD( GetRenderCallback );
    void            SetRenderCallback( core::IFunctor* renderCallback ) CORE_SET_CALLBACK_METHOD( SetRenderCallback );

    rtgi::ColorRGBA GetClearColor() const;
    void            SetClearColor( const rtgi::ColorRGBA& clearColor );

protected:
    void CallRenderCallback( content::Ref< Scene > scene );

private:
    rtgi::ColorRGBA mClearColor;
    core::IFunctor* mRenderCallback;
};

}

#endif