#include "customwidgets/customapplication.h"

#include <QtCore/QFile>
#include <QtCore/QTimer>
#include <QtGui/QCloseEvent>
#include <QtGui/QMainWindow>
#include <QtGui/QDockWidget>
#include <QtGui/QFrame>
#include <QtGui/QPaintEvent>
#include <QtUiTools/QUiLoader>

#include "core/assert.h"

CustomApplication::CustomApplication( int argc, char** argv ) :
QApplication( argc, argv )
{
}

bool CustomApplication::quit()
{
	Assert( 0 );

	return true;
}

void CustomApplication::closeEvent( QCloseEvent* closeEvent )
{
	Assert( 0 );
}