#include "core/String.hpp"

namespace core
{

char String::ConstIterator::operator * () const
{
    return mConstIterator->toAscii();
}

char String::ConstIterator::operator -> () const
{
    return mConstIterator->toAscii();
}

bool String::ConstIterator::operator == ( String::ConstIterator& other ) const
{
    return mConstIterator == other.mConstIterator;
}

bool String::ConstIterator::operator != ( String::ConstIterator& other ) const
{
    return mConstIterator != other.mConstIterator;
}

String::ConstIterator& String::ConstIterator::operator ++ ( int ) // the int means post-increment (e.g. i++)
{
    mConstIterator++;
    return *this;
}

String::String()
{
}

String::String( const String& s ) :
mString   ( s.mString ),
mByteArray( s.mString.toAscii() )
{
}

String::String( const char* s ) :
mString   ( s ),
mByteArray( s )
{
}

String::String( const QString& s ) :
mString   ( s ),
mByteArray( s.toAscii() )
{
}

String::String( const std::string& s ) :
mString   ( s.c_str() ),
mByteArray( s.c_str() )
{
}

String String::arg ( int a, int fieldWidth ) const
{
    return String( mString.arg( a, fieldWidth ) );
}

String String::arg ( uint a, int fieldWidth ) const
{
    return String( mString.arg( a, fieldWidth ) );
}

String String::arg ( long a, int fieldWidth ) const
{
    return String( mString.arg( a, fieldWidth ) );
}

String String::arg ( ulong a, int fieldWidth ) const
{
    return String( mString.arg( a, fieldWidth ) );
}

String String::arg ( short a, int fieldWidth ) const
{
    return String( mString.arg( a, fieldWidth ) );
}

String String::arg ( ushort a, int fieldWidth ) const
{
    return String( mString.arg( a, fieldWidth ) );
}

String String::arg ( char a, int fieldWidth ) const
{
    return String( mString.arg( a, fieldWidth ) );
}

String String::arg ( uchar a, int fieldWidth ) const
{
    return String( mString.arg( a, fieldWidth ) );
}

String String::arg ( double a, int fieldWidth ) const
{
    return String( mString.arg( a, fieldWidth ) );
} 

String& String::Remove( const core::String& substring )
{
    mString    = mString.remove( substring.mString );
    mByteArray = mString.toAscii();

    return *this;
} 

String::ConstIterator String::ConstBegin() const
{
    ConstIterator constIterator;
    constIterator.mConstIterator = mString.constBegin();

    return constIterator;
}

String::ConstIterator String::ConstEnd() const
{
    ConstIterator constIterator;
    constIterator.mConstIterator = mString.constEnd();

    return constIterator;
}

const char* String::ToAscii() const
{
    return mByteArray.data();
}

std::string String::ToStdString() const
{
    return mString.toStdString();
}

QString String::ToQString() const
{
    return mString;
}

#if defined(PLATFORM_WIN32)
    unsigned __int64 String::ToUnsignedInt64( bool* success ) const
#elif defined(PLATFORM_OSX)
    uint64_t String::ToUnsignedInt64( bool* success ) const
#endif
    {
    return mString.toULongLong( success );
}

bool String::operator == ( const String& s  ) const
{
    return mString == s.mString;
}

bool String::operator >  ( const String& s  ) const
{
    return mString >  s.mString;
}

bool String::operator < ( const String& s  ) const
{
    return mString <  s.mString;
}

bool String::operator != ( const String& s ) const
{
    return mString != s.mString;
}

String String::operator + ( const String&  s ) const
{
    QString tmp = mString + s.mString;
    return core::String( tmp );
}

String String::operator + ( const QString& s ) const
{
    QString tmp = mString + s;
    return core::String( tmp );
}

String String::operator + ( const char* s ) const
{
    QString tmp = mString + s; 
    return core::String( tmp );
}

String& String::operator += ( const QString& s )
{
    mString = mString + s;
    mByteArray = mString.toAscii();
    return *this;
}

}