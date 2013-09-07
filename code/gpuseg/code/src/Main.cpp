#if defined PLATFORM_WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <QtGui/QApplication>

#include "widgets/MainWindow.hpp"


//
//platform dependent main function
//
#if   defined PLATFORM_WIN32
int WINAPI WinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow )
#else  // any other platform
int main( int numberOfArguments, char** pointersToArguments )
#endif
{
//
// we have to do extra work and allocate extra memory on win32
//
#if defined PLATFORM_WIN32
    //
    // split arguments
    //
    QString     rawArgumentString( lpCmdLine );
    QStringList argumentList( rawArgumentString.split( " ", QString::SkipEmptyParts ) );
    
    int         numberOfArguments   = argumentList.size();
    char**      pointersToArguments = new char*[ numberOfArguments ];

    //
    // convert to char array, since the QApplication constructor only takes
    // arguments in the form of c strings and not QStrings.
    //
    for ( int i = 0; i < numberOfArguments; i++ )
    {
        QString string         = argumentList.at( i );
        int numberOfCharacters = string.length() + 1;

        pointersToArguments[ i ] = new char[ numberOfCharacters ];
        
        memcpy( pointersToArguments[ i ], const_cast< char* >( string.toStdString().c_str() ), numberOfCharacters );
    }
#endif

    
    //
    // init qt
    //
    QApplication application( numberOfArguments, pointersToArguments );
    application.setQuitOnLastWindowClosed( true );

    MainWindow mainWindow;
    mainWindow.show();

    application.exec();


//
// cleanup the extra memory we allocated on win32
//
#if defined PLATFORM_WIN32
    for ( int i = 0; i < numberOfArguments; i++ )
    {
        delete[] pointersToArguments[ i ];
    }

    delete[] pointersToArguments;
#endif

    return 0;
}