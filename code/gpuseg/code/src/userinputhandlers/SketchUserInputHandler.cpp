#include "userinputhandlers/SketchUserInputHandler.hpp"

#include <QtGui/QMouseEvent>

#include "Engine.hpp"

#include "widgets/RenderingWindow.hpp"

SketchUserInputHandler::SketchUserInputHandler(
    RenderingWindow* renderingWindow,
    Engine*          engine ) :
UserInputHandler(
    renderingWindow,
    engine )
{
}

void SketchUserInputHandler::renderingWindowMousePressEvent( QMouseEvent* mouseEvent )
{
    GetEngine()->BeginPlaceSeed();
    GetEngine()->AddSeedPoint( mouseEvent->pos().x(), GetRenderingWindow()->height() - mouseEvent->pos().y() );
}

void SketchUserInputHandler::renderingWindowMouseReleaseEvent( QMouseEvent* mouseEvent )
{
    GetEngine()->AddSeedPoint( mouseEvent->pos().x(), GetRenderingWindow()->height() - mouseEvent->pos().y() );
    GetEngine()->EndPlaceSeed();
}

void SketchUserInputHandler::renderingWindowMouseMoveEvent( QMouseEvent* mouseEvent )
{
    GetEngine()->AddSeedPoint( mouseEvent->pos().x(), GetRenderingWindow()->height() - mouseEvent->pos().y() );
}

void SketchUserInputHandler::renderingWindowMouseWheelEvent( QWheelEvent* wheelEvent )
{
}

void SketchUserInputHandler::renderingWindowKeyPressEvent( QKeyEvent* keyEvent )
{
}

void SketchUserInputHandler::renderingWindowKeyReleaseEvent( QKeyEvent* keyEvent )
{
}
