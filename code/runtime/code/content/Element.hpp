#ifndef CONTENT_NODE_HPP
#define CONTENT_NODE_HPP

#include <QtXml/QDomElement>
#include <QtXml/QDomDocument>

#include "core/String.hpp"

#include "math/Vector3.hpp"
#include "math/Vector4.hpp"

namespace core
{
    class String;
}

namespace content
{

class Document;

class Element
{
public:
    Element( QDomElement domElement, QDomDocument domDocument );
    ~Element();

    Element                              GetCurrentChildElement();
    Element                              GetCurrentChildElement( const core::String& nodeName );

    core::String                         GetName() const;
    bool                                 IsNull()  const;

    template< typename T > inline T      GetValue() const;
    template< typename T > inline T      GetAttributeValue( const core::String& attributeName ) const;
#if defined(PLATFORM_WIN32)
    template <>            float         GetValue() const;
    template <>            core::String  GetValue() const;
    template <>            float         GetAttributeValue( const core::String& attributeName ) const;
    template <>            core::String  GetAttributeValue( const core::String& attributeName ) const;
#endif
    
    void SetValue         ( float value );
    void SetValue         ( const core::String& value );
    void SetAttributeValue( const core::String& attributeName, float value );
    void SetAttributeValue( const core::String& attributeName, const core::String& value );

private:
    Element();

    QDomDocument mDomDocument;
    QDomElement  mDomElement;
    QDomElement  mCurrentChildDomElement;
};

template< typename T > inline T Element::GetValue() const
{
    Assert( 0 );

    T t;
    return t;
}


template< typename T > inline T Element::GetAttributeValue( const core::String& attributeName ) const
{
    Assert( 0 );

    T t;
    return t;
}
    
template <> inline float Element::GetAttributeValue( const core::String& attributeName ) const
{
    Assert( mDomElement.hasAttribute( attributeName.ToQString() ) );
        
    bool conversionSuccessful = false;
    float convertedFloat      = mDomElement.attribute( attributeName.ToQString() ).toFloat( &conversionSuccessful );
        
    Assert( conversionSuccessful );
        
    return convertedFloat;
}
    
template <> inline core::String Element::GetAttributeValue( const core::String& attributeName ) const
{
    Assert( mDomElement.hasAttribute( attributeName.ToQString() ) );
        
    return core::String( mDomElement.attribute( attributeName.ToQString() ) );
} 
    
template <> inline float Element::GetValue() const
{
    Assert( mDomElement.firstChildElement().isNull() );
        
    bool conversionSuccessful = false;
    float convertedFloat      = mDomElement.text().toFloat( &conversionSuccessful );
        
    Assert( conversionSuccessful );
    return convertedFloat;
}
    
template <> inline core::String Element::GetValue() const
{
        // only get value from nodes with no children
    Assert( mDomElement.firstChildElement().isNull() );
        
    return core::String( mDomElement.text() );
}    

    
}

#endif