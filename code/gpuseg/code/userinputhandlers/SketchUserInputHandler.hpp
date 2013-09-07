#ifndef GPUSEG_USER_INPUT_HANDLERS_SKETCH_USER_INPUT_HANDLER_HPP
#define GPUSEG_USER_INPUT_HANDLERS_SKETCH_USER_INPUT_HANDLER_HPP

#include <QtCore/QPoint>

#include "userinputhandlers/UserInputHandler.hpp"

class QMouseEvent;
class QWheelEvent;

class SketchUserInputHandler : public UserInputHandler
{
public:
    SketchUserInputHandler( RenderingWindow* renderingWindow, Engine* engine );

    virtual void renderingWindowMousePressEvent  ( QMouseEvent*  mouseEvent );
    virtual void renderingWindowMouseReleaseEvent( QMouseEvent*  mouseEvent );
    virtual void renderingWindowMouseMoveEvent   ( QMouseEvent*  mouseEvent );
    virtual void renderingWindowMouseWheelEvent  ( QWheelEvent*  wheelEvent );

    virtual void renderingWindowKeyPressEvent    ( QKeyEvent* keyEvent );
    virtual void renderingWindowKeyReleaseEvent  ( QKeyEvent* keyEvent );

protected:
    virtual ~SketchUserInputHandler() {};

private:
    QPoint mPreviousMousePosition;
};

#endif