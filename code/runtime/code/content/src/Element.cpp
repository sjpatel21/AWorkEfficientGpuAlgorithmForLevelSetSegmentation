#include "content/Element.hpp"

#include <QtXml/QDomElement>
#include <QtXml/QDomNode>

namespace content
{

Element::Element()
{
}

Element::Element( QDomElement domElement, QDomDocument domDocument ) :
mDomElement      ( domElement ),
mDomDocument     ( domDocument )
{
    mCurrentChildDomElement = domElement.firstChildElement(); 
}

Element::~Element()
{
}

Element Element::GetCurrentChildElement()
{
    Element currentChild( mCurrentChildDomElement, mDomDocument );

    mCurrentChildDomElement = mCurrentChildDomElement.nextSiblingElement();

    return currentChild;
}

Element Element::GetCurrentChildElement( const core::String& nodeName )
{
    if( !mCurrentChildDomElement.isNull() )
    {
        QString actualNodeName   = mCurrentChildDomElement.tagName();
        QString expectedNodeName = nodeName.ToQString();

        Assert( actualNodeName == expectedNodeName );
    }

    Element currentChild( mCurrentChildDomElement, mDomDocument );

    mCurrentChildDomElement = mCurrentChildDomElement.nextSiblingElement();

    return currentChild;
}

core::String Element::GetName() const
{
    return core::String( mDomElement.tagName() );
}

bool Element::IsNull() const
{
    return mDomElement.isNull();
}

void Element::SetValue( float value )
{
    // only set value from nodes with no children
    Assert( mDomElement.firstChildElement().isNull() );

    // get old node
    QDomNode oldTextNode = mDomElement.firstChild();
    Assert( oldTextNode.isText() );

    // create new node
    QDomText newTextNode = mDomDocument.createTextNode( QString( "%1" ).arg( value ) );

    mDomElement.replaceChild( newTextNode, oldTextNode );
}

void Element::SetValue( const core::String& value )
{
    // only set value from nodes with no children
    Assert( mDomElement.firstChildElement().isNull() );

    // get old node
    QDomNode oldTextNode = mDomElement.firstChild();
    Assert( oldTextNode.isText() );

    // create new node
    QDomText newTextNode = mDomDocument.createTextNode( value.ToQString() );

    mDomElement.replaceChild( newTextNode, oldTextNode );
}

void Element::SetAttributeValue( const core::String& attributeName, float value )
{
    Assert( mDomElement.hasAttribute( attributeName.ToQString() ) );

    // create new node
    QDomAttr newAttributeNode = mDomDocument.createAttribute( attributeName.ToQString() );
    newAttributeNode.setValue( QString( "%1" ).arg( value ) );

    mDomElement.removeAttribute( attributeName.ToQString() );
    mDomElement.setAttributeNode( newAttributeNode );
}

void Element::SetAttributeValue( const core::String& attributeName, const core::String& value )
{
    Assert( mDomElement.hasAttribute( attributeName.ToQString() ) );

    // create new node
    QDomAttr newAttributeNode = mDomDocument.createAttribute( attributeName.ToQString() );
    newAttributeNode.setValue( value.ToQString() );

    mDomElement.removeAttribute( attributeName.ToQString() );
    mDomElement.setAttributeNode( newAttributeNode );
}

}