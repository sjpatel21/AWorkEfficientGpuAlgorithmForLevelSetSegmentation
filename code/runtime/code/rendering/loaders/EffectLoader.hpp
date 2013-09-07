#ifndef RENDERING_LOADERS_EFFECT_LOADER_HPP
#define RENDERING_LOADERS_EFFECT_LOADER_HPP

#include "content/Loader.hpp"

class FCDocument;

namespace content
{
    class Inventory;
}

namespace rendering
{

class EffectLoader : public content::Loader
{
public:
    virtual void Load( content::Inventory* inventory, FCDocument* document );
};

}

#endif