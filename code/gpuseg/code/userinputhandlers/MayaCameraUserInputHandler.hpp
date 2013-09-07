#ifndef GPUSEG_USER_INPUT_HANDLERS_MAYA_CAMERA_USER_INPUT_HANDLER_HPP
#define GPUSEG_USER_INPUT_HANDLERS_MAYA_CAMERA_USER_INPUT_HANDLER_HPP

#include <QtCore/QPoint>

#include "userinputhandlers/UserInputHandler.hpp"

class QMouseEvent;
class QWheelEvent;
class QKeyEvent;

class MayaCameraUserInputHandler : public UserInputHandler
{
public:

    MayaCameraUserInputHandler( RenderingWindow* renderingWindow, Engine* engine );

    virtual void renderingWindowMousePressEvent  ( QMouseEvent*  mouseEvent );
    virtual void renderingWindowMouseReleaseEvent( QMouseEvent*  mouseEvent );
    virtual void renderingWindowMouseMoveEvent   ( QMouseEvent*  mouseEvent );
    virtual void renderingWindowMouseWheelEvent  ( QWheelEvent*  wheelEvent );

    virtual void renderingWindowKeyPressEvent    ( QKeyEvent* keyEvent );
    virtual void renderingWindowKeyReleaseEvent  ( QKeyEvent* keyEvent );

protected:
    virtual ~MayaCameraUserInputHandler() {};

private:
    QPoint mPreviousMousePosition;
};

#endif