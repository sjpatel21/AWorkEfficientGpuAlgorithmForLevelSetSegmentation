#ifndef CORE_STRING_HPP
#define CORE_STRING_HPP

#include <string>

#include <QtCore/QString>
#include <QtCore/QHash>

#include "core/Assert.hpp"

namespace core
{

class String
{
public:
    // has to be inlined according to c++ standard 11.4 [class.friend] paragraph 5
    friend String operator+ ( const char* s1, const String& s2 ) { return String( s1 + s2.mString ); };

    // for qt hashmap compatibility
    friend uint   qHash     ( const String& key );

    class ConstIterator
    {
        friend class String;

    public:
        char                   operator *  () const;
        char                   operator -> () const;
        bool                   operator == ( String::ConstIterator& other ) const;
        bool                   operator != ( String::ConstIterator& other ) const;
        String::ConstIterator& operator ++ ( int ); // the int means post-increment (e.g. i++)

    private:
        QString::ConstIterator mConstIterator;
    };

    String();
    String( const String&      s );
    String( const char*        s );
    String( const QString&     s );
    String( const std::string& s );

    String arg ( int    a, int fieldWidth = 0 ) const;
    String arg ( uint   a, int fieldWidth = 0 ) const;
    String arg ( long   a, int fieldWidth = 0 ) const;
    String arg ( ulong  a, int fieldWidth = 0 ) const;
    String arg ( short  a, int fieldWidth = 0 ) const;
    String arg ( ushort a, int fieldWidth = 0 ) const;
    String arg ( char   a, int fieldWidth = 0 ) const;
    String arg ( uchar  a, int fieldWidth = 0 ) const;
    String arg ( double a, int fieldWidth = 0 ) const; 

    String& Remove( const core::String& substring );

    String::ConstIterator ConstBegin()  const;
    String::ConstIterator ConstEnd()    const;

    const char*           ToAscii()                               const;
    std::string           ToStdString()                           const;
    QString               ToQString()                             const;
#if defined(PLATFORM_WIN32)
    unsigned __int64      ToUnsignedInt64( bool* success = NULL ) const;
#elif defined(PLATFORM_OSX)
    uint64_t      ToUnsignedInt64( bool* success = NULL ) const;
#endif
    
    bool    operator != ( const String& s  ) const;
    bool    operator == ( const String& s  ) const;
    bool    operator >  ( const String& s  ) const;
    bool    operator <  ( const String& s  ) const;
    String  operator +  ( const String&  s ) const;
    String  operator +  ( const QString& s ) const;
    String  operator +  ( const char*    s ) const;
    String& operator += ( const QString& s );

private:
    QString    mString;
    QByteArray mByteArray;
};

inline uint qHash( const String& key )
{
    return ::qHash( key.mString );
}

}

#endif