#include "content/Document.hpp"

#include <QtCore/QString>
#include <QtCore/QFile>
#include <QtXml/QDomElement>

namespace content
{

Document::Document( const core::String& fileName, DocumentOpenMode fileOpenMode ) :
mFileName    ( fileName ),
mFileOpenMode( fileOpenMode )
{
    QFile qFile( fileName.ToQString() );
    qFile.open( QFile::ReadOnly );

    mDomDocument.setContent( &qFile );

    qFile.close();
}

Document::~Document()
{
}

Element Document::GetRootElement() const
{
    Element rootNode( mDomDocument.firstChildElement(), mDomDocument );

    Assert( rootNode.GetName() == "Root" );

    return rootNode;
}

void Document::Save()
{
    Assert( mFileOpenMode == DocumentOpenMode_ReadWrite );

    if ( mFileOpenMode == DocumentOpenMode_ReadWrite )
    {
        QFile saveFile( mFileName.ToQString() );
        saveFile.open( QFile::WriteOnly );

        saveFile.write( mDomDocument.toString( 4 ).toAscii() );

        saveFile.close();
    }
}

}