#include "rendering/loaders/SceneLoader.hpp"

#include <FCollada.h> 
#include <FCDocument/FCDocument.h>
#include <FCDocument/FCDLibrary.h>
#include <FCDocument/FCDSceneNode.h>
#include <FCDocument/FCDSceneNodeIterator.h>
#include <FCDocument/FCDEntityInstance.h>
#include <FCDocument/FCDGeometryInstance.h>
#include <FCDocument/FCDMaterialInstance.h>
#include <FCDocument/FCDEntity.h>
#include <FCDocument/FCDGeometry.h>
#include <FCDocument/FCDMaterial.h>
#include <FCDocument/FCDCamera.h>
#include <FUtils/FUDaeEnum.h>

#include "core/Printf.hpp"

#include "math/Utility.hpp"

#include "content/Inventory.hpp"
#include "content/Ref.hpp"

#include "rendering/Context.hpp"
#include "rendering/Camera.hpp"
#include "rendering/Renderable.hpp"
#include "rendering/Scene.hpp"
#include "rendering/MaterialStrip.hpp"
#include "rendering/Material.hpp"

namespace rendering
{

void SceneLoader::Load( content::Inventory* inventory, FCDocument* document )
{
    FCDVisualSceneNodeLibrary* visualSceneNodeLibrary = document->GetVisualSceneLibrary();

    for ( size_t i = 0; i < visualSceneNodeLibrary->GetEntityCount(); i++ )
    {
        FCDSceneNode* visualSceneNode = visualSceneNodeLibrary->GetEntity( i );
        core::String  sceneName       = visualSceneNode->GetName().c_str();
        
        container::List< content::Ref< Renderable > > renderables;

        FCDSceneNodeIterator visualSceneNodeIterator( visualSceneNode );

        FCDSceneNode* node = visualSceneNodeIterator.Next();

        while( !visualSceneNodeIterator.IsDone() )
        {
            for ( size_t j = 0; j < node->GetInstanceCount(); j++ )
            {
                FCDEntityInstance*      instance     = node->GetInstance( j );
                FCDEntityInstance::Type instanceType = instance->GetType();

                if ( instanceType == FCDEntityInstance::GEOMETRY )
                {
                    FCDGeometryInstance*              geometryInstance = dynamic_cast< FCDGeometryInstance* >( instance );
                    FCDGeometry*                      geometry         = dynamic_cast< FCDGeometry* >( geometryInstance->GetEntity() );
                    core::String                      geometryName     = geometry->GetName().c_str();
                    content::Ref< Renderable >        renderable       = inventory->Find< Renderable >( geometryName );
                    container::List< MaterialStrip* > materialStrips   = renderable->GetMaterialStrips();

                    renderable->SetTransform( node->CalculateWorldTransform() );

                    for ( size_t k = 0; k < geometryInstance->GetMaterialInstanceCount(); k++ )
                    {
                        FCDMaterialInstance* materialInstance = geometryInstance->GetMaterialInstance( k );
                        FCDMaterial*         fcdMaterial      = dynamic_cast< FCDMaterial* >( materialInstance->GetEntity() );
                        core::String         materialSemantic = materialInstance->GetSemantic().c_str();
                        core::String         materialName     = fcdMaterial->GetName().c_str();

                        content::Ref< Material > material;

                        if ( inventory->Contains( materialName ) )
                        {
                            material = inventory->Find< Material >( materialName );
                        }
                        else
                        {
                            material = Context::GetDebugMaterial();
                            core::Printf(
                                "Warning - Scene Loader - Couldn't find material " + materialName +
                                " in inventory when assigning a material to "      + geometryName +
                                ".  Assigning the debug material instead.\n\n" );
                        }

                        foreach ( MaterialStrip* materialStrip, materialStrips )
                        {
                            if ( materialStrip->GetMaterialSemantic() == materialSemantic )
                            {
                                SetFinalized( materialStrip, false );
                                materialStrip->SetMaterial( material );
                                SetFinalized( materialStrip, true );
                            }
                        }
                    }

                    renderables.Append( renderable );

                }

                if ( instanceType == FCDEntityInstance::SIMPLE )
                {
                    FCDEntity*      entity     = instance->GetEntity();
                    FCDEntity::Type entityType = entity->GetType();
                    
                    if ( entityType == FCDEntity::CAMERA )
                    {
#if defined(PLATFORM_WIN32)
                        core::String cameraName = entity->GetName();
#elif defined(PLATFORM_OSX)
                        core::String cameraName( entity->GetName().c_str() );
#endif
                        
                        FCDCamera* fcdCamera    = dynamic_cast< FCDCamera* >( entity );
                        float      fovYRadians  = math::DegreesToRadians( *( fcdCamera->GetFovY() ) );
                        float      aspectRatio  = 1;
                        float      nearPlane    = *( fcdCamera->GetNearZ() );
                        float      farPlane     = *( fcdCamera->GetFarZ()  );

                        if ( fcdCamera->HasAspectRatio() )
                        {
                            aspectRatio = fcdCamera->GetAspectRatio();
                        }

                        FCDTLookAt* lookAt = NULL;

                        for ( size_t k = 0; k < node->GetTransformCount(); k++ )
                        {
                            FCDTransform* transform = node->GetTransform( k );

                            if ( transform->GetType() == FCDTransform::LOOKAT )
                            {
                                lookAt = dynamic_cast< FCDTLookAt* >( transform );
                            }
                        }

                        Camera* camera = new Camera();
                        camera->SetProjectionParameters( fovYRadians, aspectRatio, nearPlane, farPlane );

                        if ( lookAt != NULL )
                        {
                            camera->SetLookAtVectors( lookAt->GetPosition(), lookAt->GetTarget(), lookAt->GetUp() );
                        }
                        else
                        {
                            math::Matrix44 cameraTransform = node->CalculateLocalTransform();
                            cameraTransform.InvertTranspose();
                            camera->SetLookAtMatrix( cameraTransform );
                        }

                        SetDebugName( camera, cameraName );
                        Insert( inventory, cameraName, camera );
                    }
                }
            }

            node = visualSceneNodeIterator.Next();
        }

        Scene* scene = new Scene();
        SetDebugName( scene, sceneName );
        SetFinalized( scene, false );

        scene->SetRenderables( renderables );

        SetFinalized( scene, true );

        Insert( inventory, sceneName, scene );
    }
}

}