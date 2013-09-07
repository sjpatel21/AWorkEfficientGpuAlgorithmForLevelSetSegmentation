#include "python/Macros.hpp"

#include "core/RefCounted.hpp"

#include "rendering/rtgi/RTGI.hpp"

#include "rendering/Scene.hpp"
#include "rendering/Camera.hpp"
#include "rendering/Context.hpp"
#include "rendering/TextConsole.hpp"
#include "rendering/renderstrategies/RenderStrategy.hpp"
#include "rendering/renderstrategies/BasicRenderStrategy.hpp"

//
// base class wrappers
//
PYTHON_DECLARE_CLASS_WRAPPER_BASE_BEGIN( RenderStrategy, rendering::RenderStrategy )

    PYTHON_CLASS_WRAPPER_BASE_METHOD_VOID_SIGNATURE( Update, content::Ref< rendering::Scene > scene, content::Ref< rendering::Camera > camera, double timeDeltaSeconds )
    PYTHON_CLASS_WRAPPER_BASE_METHOD_VIRTUAL_VOID_ARGUMENTS( rendering::RenderStrategy::Update, "Update", scene, camera, timeDeltaSeconds )

    PYTHON_CLASS_WRAPPER_BASE_METHOD_HELPER_VOID_SIGNATURE( Update, content::Ref< rendering::Scene > scene, content::Ref< rendering::Camera > camera, double timeDeltaSeconds )
    PYTHON_CLASS_WRAPPER_BASE_METHOD_HELPER_VOID_ARGUMENTS( rendering::RenderStrategy::Update, scene, camera, timeDeltaSeconds )

    PYTHON_CLASS_WRAPPER_BASE_METHOD_VOID_SIGNATURE( Render, content::Ref< rendering::Scene > scene, content::Ref< rendering::Camera > camera )
    PYTHON_CLASS_WRAPPER_BASE_METHOD_VIRTUAL_VOID_ARGUMENTS( rendering::RenderStrategy::Render, "Render", scene, camera )

    PYTHON_CLASS_WRAPPER_BASE_METHOD_HELPER_VOID_SIGNATURE( Render, content::Ref< rendering::Scene > scene, content::Ref< rendering::Camera > camera )
    PYTHON_CLASS_WRAPPER_BASE_METHOD_HELPER_VOID_ARGUMENTS( rendering::RenderStrategy::Render, scene, camera )

PYTHON_DECLARE_CLASS_WRAPPER_BASE_END


//
// factories
//
PYTHON_DECLARE_CLASS_FACTORY( CreateBasicRenderStrategy, rendering::BasicRenderStrategy )


//
// rendering module
//
PYTHON_MODULE_BEGIN( rendering )

    PYTHON_CLASS_FACTORY( "CreateBasicRenderStrategy", CreateBasicRenderStrategy, rendering::BasicRenderStrategy )

    PYTHON_CLASS_BEGIN_BASE( "RenderStrategy", RenderStrategy )
        PYTHON_CLASS_METHOD_VIRTUAL(      "Update",            rendering::RenderStrategy::Update,        RenderStrategy, Update )
        PYTHON_CLASS_METHOD_VIRTUAL(      "Render",            rendering::RenderStrategy::Render,        RenderStrategy, Render )
        PYTHON_CLASS_METHOD(              "SetClearColor",     rendering::RenderStrategy::SetClearColor )
        PYTHON_CLASS_METHOD_SET_CALLBACK( "SetRenderCallback", rendering::RenderStrategy::SetRenderCallback )
    PYTHON_CLASS_END

    PYTHON_CLASS_BEGIN_DERIVED( "BasicRenderStrategy", rendering::BasicRenderStrategy, rendering::RenderStrategy )
    PYTHON_CLASS_END

    PYTHON_CLASS_BEGIN( "Camera", rendering::Camera )
        PYTHON_CLASS_METHOD( "SetLookAtVectors",        rendering::Camera::SetLookAtVectors )
        PYTHON_CLASS_METHOD( "SetProjectionParameters", rendering::Camera::SetProjectionParameters )
        PYTHON_CLASS_METHOD( "GetLookAtVectors",        rendering::Camera::GetLookAtVectors )
    PYTHON_CLASS_END

    PYTHON_SHARED_CLASS( rendering::Camera )

    PYTHON_CLASS_BEGIN_NON_COPYABLE( "Context", rendering::Context )
        PYTHON_CLASS_METHOD_STATIC( "SetCurrentCamera",         rendering::Context::SetCurrentCamera )
        PYTHON_CLASS_METHOD_STATIC( "SetCurrentScene",          rendering::Context::SetCurrentScene )
        PYTHON_CLASS_METHOD_STATIC( "SetCurrentRenderStrategy", rendering::Context::SetCurrentRenderStrategy )
        PYTHON_CLASS_METHOD_STATIC( "GetCurrentCamera"        , rendering::Context::GetCurrentCamera )
    PYTHON_CLASS_END

    PYTHON_CLASS_BEGIN_NON_COPYABLE( "TextConsole", rendering::TextConsole )
#if defined(PLATFORM_WIN32)
        PYTHON_CLASS_METHOD_STATIC(
            "PrintToStaticConsole",
            PYTHON_DISAMBIGUATE_OVERLOADED_METHOD(
                rendering::TextConsole::PrintToStaticConsole, void, const core::String&, const core::String& ) )
#elif defined(PLATFORM_OSX)
        PYTHON_CLASS_METHOD_STATIC_FROM_POINTER(
            "PrintToStaticConsole",
            PYTHON_DISAMBIGUATE_OVERLOADED_METHOD(
                rendering::TextConsole::PrintToStaticConsole, void, const core::String&, const core::String& ) )
#endif
    PYTHON_CLASS_END

    PYTHON_NAMESPACE_BEGIN( "rtgi", rendering::rtgi )
        PYTHON_CLASS_BEGIN( "ColorRGBA", rendering::rtgi::ColorRGBA )
            PYTHON_CLASS_METHOD_CONSTRUCTOR( float, float, float, float )
        PYTHON_CLASS_END
    PYTHON_NAMESPACE_END

PYTHON_MODULE_END
