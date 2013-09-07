#ifndef GPUSEG_USER_INPUT_HANDLERS_USER_INPUT_HANDLER_HPP
#define GPUSEG_USER_INPUT_HANDLERS_USER_INPUT_HANDLER_HPP

#include "core/RefCounted.hpp"

class QMouseEvent;
class QWheelEvent;
class QKeyEvent;
class RenderingWindow;
class Engine;

class UserInputHandler : public core::RefCounted
{
public:
    
    UserInputHandler( RenderingWindow* renderingWindow, Engine* engine );

    virtual void renderingWindowMousePressEvent  ( QMouseEvent*  mouseEvent ) = 0;
    virtual void renderingWindowMouseReleaseEvent( QMouseEvent*  mouseEvent ) = 0;
    virtual void renderingWindowMouseMoveEvent   ( QMouseEvent*  mouseEvent ) = 0;
    virtual void renderingWindowMouseWheelEvent  ( QWheelEvent*  wheelEvent ) = 0;
    virtual void renderingWindowKeyPressEvent    ( QKeyEvent* keyEvent ) = 0;
    virtual void renderingWindowKeyReleaseEvent  ( QKeyEvent* keyEvent ) = 0;

    Engine*         GetEngine();
    RenderingWindow* GetRenderingWindow();

protected:
    virtual ~UserInputHandler();

private:
    Engine*         mEngine;
    RenderingWindow* mRenderingWindow;
};

#endif