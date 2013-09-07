#ifndef GPUSEG_RENDERSTRATEGY_HPP
#define GPUSEG_RENDERSTRATEGY_HPP

#include "content/Ref.hpp"

#include "rendering/renderstrategies/RenderStrategy.hpp"

#include "VolumeDesc.hpp"

namespace content
{
class Inventory;
}

namespace rendering
{

class Scene;
class Camera;
class Material;

}

class Engine;

class GPUSegRenderStrategy : public rendering::RenderStrategy
{

public:
    GPUSegRenderStrategy();
    virtual ~GPUSegRenderStrategy();

    virtual void SetEngine                        ( Engine* engine );
    virtual void SetSourceVolumeTexture           ( rendering::rtgi::Texture* texture );
    virtual void SetCurrentLevelSetVolumeTexture  ( rendering::rtgi::Texture* texture );
    virtual void SetFrozenLevelSetVolumeTexture   ( rendering::rtgi::Texture* texture );
    virtual void SetActiveElementsVolumeTexture   ( rendering::rtgi::Texture* texture );

    virtual void LoadVolume( const VolumeDesc& volumeDesc );
    virtual void UnloadVolume();

    virtual void Update( content::Ref< rendering::Scene > scene, content::Ref< rendering::Camera > camera, double timeDeltaSeconds );
    virtual void Render( content::Ref< rendering::Scene > scene, content::Ref< rendering::Camera > camera );

    virtual void BeginPlaceSeed();
    virtual void AddSeedPoint( int screenX, int screenY );
    virtual void EndPlaceSeed();

private:
    math::Vector3 ComputeIntersectionWithCuttingPlane( math::Vector3 virtualScreenCoordinates );
};

#endif