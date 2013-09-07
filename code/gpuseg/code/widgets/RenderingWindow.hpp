#ifndef GPUSEG_WIDGETS_RENDERING_WINDOW_HPP
#define GPUSEG_WIDGETS_RENDERING_WINDOW_HPP

#include <QtGui/QWidget>

class MainWindow;

class RenderingWindow : public QWidget
{
    Q_OBJECT

public:
    RenderingWindow( MainWindow* parent = NULL );

protected:
    virtual void resizeEvent      ( QResizeEvent* resizeEvent );
    virtual void mouseMoveEvent   ( QMouseEvent*  mouseEvent );
    virtual void mousePressEvent  ( QMouseEvent*  mouseEvent );
    virtual void mouseReleaseEvent( QMouseEvent*  mouseEvent );
    virtual void wheelEvent       ( QWheelEvent*  wheelEvent );

    MainWindow* mMainWindow;
};

#endif
