#ifndef RENDERING_RENDERSTRATEGIES_BASIC_RENDERSTRATEGY_HPP
#define RENDERING_RENDERSTRATEGIES_BASIC_RENDERSTRATEGY_HPP

#include "content/Ref.hpp"

#include "rendering/renderstrategies/RenderStrategy.hpp"

namespace content
{
class Inventory;
}

namespace rendering
{

class Scene;
class Camera;
class Material;

class BasicRenderStrategy : public RenderStrategy
{

public:
    BasicRenderStrategy();
    virtual ~BasicRenderStrategy();

    virtual void Update( content::Ref< Scene > scene, content::Ref< Camera > camera, double timeDeltaSeconds );
    virtual void Render( content::Ref< Scene > scene, content::Ref< Camera > camera );

private:
    content::Inventory*      mInventory;
    content::Ref< Material > mDebugTexCoordMaterial;
    content::Ref< Material > mSolidWireFrameMaterial;
};

}

#endif