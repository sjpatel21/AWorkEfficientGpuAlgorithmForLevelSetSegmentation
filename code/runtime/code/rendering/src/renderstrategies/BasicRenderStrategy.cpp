#include "rendering/renderstrategies/BasicRenderStrategy.hpp"

#include "core/Assert.hpp"
#include "core/String.hpp"
#include "core/Functor.hpp"

#include "math/Vector3.hpp"
#include "math/Matrix44.hpp"

#include "content/Ref.hpp"
#include "content/Inventory.hpp"
#include "content/LoadManager.hpp"

#include "rendering/Camera.hpp"
#include "rendering/TextConsole.hpp"
#include "rendering/DebugDraw.hpp"
#include "rendering/Renderable.hpp"
#include "rendering/Scene.hpp"
#include "rendering/Material.hpp"

#include "rendering/rtgi/RTGI.hpp"
#include "rendering/rtgi/Effect.hpp"
#include "rendering/rtgi/ShaderProgram.hpp"
#include "rendering/rtgi/Texture.hpp"
#include "rendering/rtgi/IndexBuffer.hpp"
#include "rendering/rtgi/VertexBuffer.hpp"

#include "rendering/Context.hpp"

namespace rendering
{

    BasicRenderStrategy::BasicRenderStrategy()
    {
        mInventory = new content::Inventory();
        mInventory->AddRef();

        content::LoadManager::Load( "runtime/art/BasicRenderStrategyOverrideMaterials.dae", mInventory );

        mSolidWireFrameMaterial = mInventory->Find< Material >( "solidWireFrameMaterial" );
        mDebugTexCoordMaterial  = mInventory->Find< Material >( "debugTexCoordMaterial" );
    }

    BasicRenderStrategy::~BasicRenderStrategy()
    {
        mDebugTexCoordMaterial  = content::Ref< Material >();
        mSolidWireFrameMaterial = content::Ref< Material >();

        content::LoadManager::Unload( mInventory );

        mInventory->Release();
        mInventory = NULL;
    }

    void BasicRenderStrategy::Update( content::Ref< Scene > scene, content::Ref< Camera > camera, double timeDeltaSeconds )
    {
    }

    void BasicRenderStrategy::Render( content::Ref< Scene > scene, content::Ref< Camera > camera )
    {
        rtgi::ClearColorBuffer( GetClearColor() );
        rtgi::ClearDepthBuffer();
        rtgi::ClearStencilBuffer();

        scene->Render();
        //scene->Render( mDebugTexCoordMaterial );
        //scene->Render( mSolidWireFrameMaterial );

        if ( GetRenderCallback() != NULL )
        {
            GetRenderCallback()->Call( reinterpret_cast< void* >( const_cast< Scene* >( scene.GetRawPointer() ) ) );
        }
    }

}