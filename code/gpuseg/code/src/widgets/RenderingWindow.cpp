#include "widgets/RenderingWindow.hpp"

#include <QtCore/QSize>
#include <QtGui/QPaintEvent>
#include <QtGui/QResizeEvent>

#include "core/Assert.hpp"

#include "Engine.hpp"
#include "widgets/MainWindow.hpp"

RenderingWindow::RenderingWindow( MainWindow* parent ) :
QWidget    ( parent ),
mMainWindow( parent )
{
    // set up drawing to work nicely with openGL
    setAttribute( Qt::WA_PaintOnScreen,      true );
    setAttribute( Qt::WA_NoSystemBackground, true );
    setAttribute( Qt::WA_OpaquePaintEvent,   true );

    // set window attributes
    setMinimumSize( 600, 600 );
}

//
// user event handling.  pass these events to the main window so
// the main window can decide what to do.  the main window will
// have a better idea of what mode (2D paint, 3D camera, etc.)
// the user is in, so defer event handling logic to the main
// window.
//
void RenderingWindow::resizeEvent( QResizeEvent* resizeEvent )
{
    mMainWindow->renderingWindowResizeEvent( resizeEvent );
}

void RenderingWindow::mousePressEvent( QMouseEvent* mouseEvent )
{
    mMainWindow->renderingWindowMousePressEvent( mouseEvent );
}

void RenderingWindow::mouseReleaseEvent( QMouseEvent* mouseEvent )
{
    mMainWindow->renderingWindowMouseReleaseEvent( mouseEvent );
}

void RenderingWindow::mouseMoveEvent( QMouseEvent* mouseEvent )
{
    mMainWindow->renderingWindowMouseMoveEvent( mouseEvent );
}

void RenderingWindow::wheelEvent( QWheelEvent* wheelEvent )
{
    mMainWindow->renderingWindowMouseWheelEvent( wheelEvent );
}