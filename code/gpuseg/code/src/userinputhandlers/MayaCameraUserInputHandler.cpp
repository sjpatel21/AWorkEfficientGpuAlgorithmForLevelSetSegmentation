#include "userinputhandlers/MayaCameraUserInputHandler.hpp"

#include <math.h>

#include <QtGui/QMouseEvent>
#include <QtGui/QKeyEvent>

#include "math/Utility.hpp"

#include "Engine.hpp"

#include "widgets/RenderingWindow.hpp"

static const float MOUSE_SENSITIVITY_X     = 1.0f;
static const float MOUSE_SENSITIVITY_Y     = 1.5f;
static const float MOUSE_WHEEL_SENSITIVITY = 0.1f;

MayaCameraUserInputHandler::MayaCameraUserInputHandler(
    RenderingWindow* renderingWindow,
    Engine*          engine ) :
UserInputHandler(
    renderingWindow,
    engine )
{
}

void MayaCameraUserInputHandler::renderingWindowMousePressEvent( QMouseEvent* mouseEvent )
{
    mPreviousMousePosition = mouseEvent->pos();
}

void MayaCameraUserInputHandler::renderingWindowMouseReleaseEvent( QMouseEvent* mouseEvent )
{
}

void MayaCameraUserInputHandler::renderingWindowMouseMoveEvent( QMouseEvent* mouseEvent )
{
    QPoint currentMousePosition = mouseEvent->pos();
    QPoint mouseDelta           = currentMousePosition - mPreviousMousePosition;
    mPreviousMousePosition      = currentMousePosition;

    float verticalTrackBallRadius = GetRenderingWindow()->height() / 2;
    float verticalAngleRadians    = atan( mouseDelta.y() * MOUSE_SENSITIVITY_Y / verticalTrackBallRadius );
    float horizontalAngleRadians  = atan( mouseDelta.x() * MOUSE_SENSITIVITY_X / verticalTrackBallRadius );

    float verticaldegrees = math::RadiansToDegrees( verticalAngleRadians );

    GetEngine()->RotateCamera( horizontalAngleRadians, verticalAngleRadians );
}

void MayaCameraUserInputHandler::renderingWindowMouseWheelEvent( QWheelEvent* wheelEvent )
{
    float distance = MOUSE_WHEEL_SENSITIVITY * wheelEvent->delta();

    GetEngine()->MoveCameraAlongViewVector( distance );
}

void MayaCameraUserInputHandler::renderingWindowKeyPressEvent( QKeyEvent* keyEvent )
{
}

void MayaCameraUserInputHandler::renderingWindowKeyReleaseEvent( QKeyEvent* )
{
}
