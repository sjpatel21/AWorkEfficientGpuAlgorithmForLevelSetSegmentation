#ifndef RENDERING_RENDERSTRATEGIES_SKETCH_RENDERSTRATEGY_HPP
#define RENDERING_RENDERSTRATEGIES_SKETCH_RENDERSTRATEGY_HPP

#include "rendering/renderstrategies/RenderStrategy.hpp"

namespace rendering
{

class SketchRenderStrategy : public RenderStrategy
{
public:
    virtual void BeginSketch()                                                                                                   = 0;
    virtual void EndSketch()                                                                                                     = 0;
    virtual void ClearSketches()                                                                                                 = 0;
    virtual void AddSketchPoint   ( int virtualScreenX,    int virtualScreenY )                                                  = 0;
    virtual void RemoveSketchPoint( int virtualScreenX,    int virtualScreenY )                                                  = 0;
    virtual void MoveSketchPoint  ( int oldVirtualScreenX, int oldVirtualScreenY, int newVirtualScreenX, int newVirtualScreenY ) = 0;
};

}

#endif