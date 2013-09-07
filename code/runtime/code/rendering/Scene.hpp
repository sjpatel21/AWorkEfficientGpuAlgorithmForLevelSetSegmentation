#ifndef RENDERING_SCENE_HPP
#define RENDERING_SCENE_HPP

#include "container/List.hpp"

#include "content/Asset.hpp"
#include "content/Ref.hpp"

#include "rendering/Renderable.hpp"

namespace rendering
{

class Scene : public content::Asset
{
public:
    Scene();

    void Render() const;
    void Render( content::Ref< Material > overrideMaterial ) const;

    const container::List< content::Ref< Renderable > >& GetRenderables() const;
    void                                                 SetRenderables( container::List< content::Ref< Renderable > >& renderables );

protected:
    virtual ~Scene();

private:
    container::List< content::Ref< Renderable > > mRenderables;
};

}

#endif