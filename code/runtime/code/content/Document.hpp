#ifndef CONTENT_FILE_HPP
#define CONTENT_FILE_HPP

#include <QtXml/QDomDocument>

#include "content/Element.hpp"

namespace core
{
    class String;
}

namespace content
{

enum DocumentOpenMode
{
    DocumentOpenMode_ReadOnly,
    DocumentOpenMode_ReadWrite
};

class Document
{
public:
    Document( const core::String& fileName, DocumentOpenMode fileOpenMode = DocumentOpenMode_ReadOnly );
    ~Document();

    Element GetRootElement() const;

    void Save();

private:
    DocumentOpenMode mFileOpenMode;
    core::String     mFileName;
    QDomDocument     mDomDocument;
};

}

#endif