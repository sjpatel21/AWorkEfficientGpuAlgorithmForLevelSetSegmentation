#ifndef RENDERING_LOADERS_GEOMETRY_LOADER_HPP
#define RENDERING_LOADERS_GEOMETRY_LOADER_HPP

#include "content/Loader.hpp"

class FCDocument;

namespace content
{
    class Inventory;
}

namespace rendering
{

class GeometryLoader : public content::Loader
{
public:
    virtual void Load( content::Inventory* inventory, FCDocument* document );
};

}

#endif