#ifndef PYTHON_MACROS_HPP
#define PYTHON_MACROS_HPP

#include "python/BoostPythonPrefix.hpp"
#include <boost/python.hpp>
#include "python/BoostPythonSuffix.hpp"

#include "core/NameSpaceID.hpp"
#include "core/Assert.hpp"
#include "core/RefCounted.hpp"

//
// wrapping base classes 
//
#define PYTHON_CLASS_WRAPPER_BASE_NAME Wrapper
#define PYTHON_CLASS_WRAPPER_BASE_FORWARDING_METHOD ForwardingMethod

#define PYTHON_DECLARE_CLASS_WRAPPER_BASE_BEGIN(baseClassName,fullyQualifiedClass)                                                        \
class baseClassName##PYTHON_CLASS_WRAPPER_BASE_NAME : public fullyQualifiedClass , public boost::python::wrapper< fullyQualifiedClass >   \
    {                                                                                                                                     \
    public:

#define PYTHON_DECLARE_CLASS_WRAPPER_BASE_END \
    };

#define PYTHON_CLASS_WRAPPER_BASE_METHOD_SIGNATURE(returnType,methodName,...) \
    returnType methodName ( __VA_ARGS__ )                                     \
    {

#define PYTHON_CLASS_WRAPPER_BASE_METHOD_VOID_SIGNATURE(methodName,...) \
    void methodName ( __VA_ARGS__ )                                     \
    {

#define PYTHON_CLASS_WRAPPER_BASE_METHOD_PURE_VIRTUAL_ARGUMENTS(methodName,...)   \
        return this->get_override( methodName )( __VA_ARGS__ );                   \
    };

#define PYTHON_CLASS_WRAPPER_BASE_METHOD_PURE_VIRTUAL_VOID_ARGUMENTS(methodName,...)   \
        this->get_override( methodName )( __VA_ARGS__ );                               \
    };


#define PYTHON_CLASS_WRAPPER_BASE_METHOD_VIRTUAL_ARGUMENTS(fullyQualifiedMethod,methodName,...)  \
        if ( boost::python::override overrideFunction = this->get_override( methodName ) )       \
        {                                                                                        \
            return overrideFunction( __VA_ARGS__ );                                              \
        }                                                                                        \
        else                                                                                     \
        {                                                                                        \
            return fullyQualifiedMethod ( __VA_ARGS__ );                                         \
        }                                                                                        \
    };

#define PYTHON_CLASS_WRAPPER_BASE_METHOD_VIRTUAL_VOID_ARGUMENTS(fullyQualifiedMethod,methodName,...) \
        if ( boost::python::override overrideFunction = this->get_override( methodName ) )           \
        {                                                                                            \
            overrideFunction( __VA_ARGS__ );                                                         \
        }                                                                                            \
        else                                                                                         \
        {                                                                                            \
            fullyQualifiedMethod ( __VA_ARGS__ );                                                    \
        }                                                                                            \
    };


#define PYTHON_CLASS_WRAPPER_BASE_METHOD_HELPER_SIGNATURE(returnType,methodName,...)   \
    returnType methodName##PYTHON_CLASS_WRAPPER_BASE_FORWARDING_METHOD ( __VA_ARGS__ ) \
    {

#define PYTHON_CLASS_WRAPPER_BASE_METHOD_HELPER_VOID_SIGNATURE(methodName,...)   \
    void methodName##PYTHON_CLASS_WRAPPER_BASE_FORWARDING_METHOD ( __VA_ARGS__ ) \
    {

#define PYTHON_CLASS_WRAPPER_BASE_METHOD_HELPER_ARGUMENTS(fullyQualifiedMethod,...) \
        return this-> fullyQualifiedMethod ( __VA_ARGS__ );                         \
    };

#define PYTHON_CLASS_WRAPPER_BASE_METHOD_HELPER_VOID_ARGUMENTS(fullyQualifiedMethod,...) \
        this-> fullyQualifiedMethod ( __VA_ARGS__ );                                     \
    };


//
// class factories
//
#define PYTHON_DECLARE_CLASS_FACTORY(factoryName,fullyQualifiedClass)                                                       \
    boost::shared_ptr< fullyQualifiedClass > factoryName ()                                                                 \
{                                                                                                                           \
    fullyQualifiedClass * factoryName##object = new fullyQualifiedClass ();                                                 \
    factoryName##object->AddRef();                                                                                          \
    return boost::shared_ptr< fullyQualifiedClass >( factoryName##object , boost::mem_fn( &core::RefCounted::Release ) );   \
};


//
// module definition
//
#define PYTHON_MODULE_BEGIN(moduleName)            \
    BOOST_PYTHON_MODULE( moduleName )              \
    {

#define PYTHON_MODULE_END \
    }


//
// static method definition
//
#define PYTHON_METHOD_STATIC(methodName,fullyQualifiedMethod) \
    boost::python::def( methodName, fullyQualifiedMethod, boost::python::return_value_policy< boost::python::return_by_value >() );


#define PYTHON_CLASS_FACTORY(methodName,fullyQualifiedMethod,fullyQualifiedClass) \
    PYTHON_METHOD_STATIC( methodName, fullyQualifiedMethod )                      \
    PYTHON_SHARED_CLASS( fullyQualifiedClass )

//
// class definition
//
#define PYTHON_CLASS_BEGIN(className,fullyQualifiedClass) \
    {                                                     \
        boost::python::scope scope##__COUNT__ = boost::python::class_< fullyQualifiedClass >( className, boost::python::no_init )

#define PYTHON_CLASS_BEGIN_NON_COPYABLE(className,fullyQualifiedClass)                            \
    {                                                                                             \
        boost::python::scope scope##__COUNT__ =                                                   \
        boost::python::class_< fullyQualifiedClass, boost::noncopyable >( className, boost::python::no_init )

#define PYTHON_CLASS_BEGIN_DERIVED(className,fullyQualifiedClass,...)                             \
    {                                                                                             \
        boost::python::scope scope##__COUNT__ =                                                   \
        boost::python::class_< fullyQualifiedClass, boost::python::bases< __VA_ARGS__ > >( className, boost::python::no_init )

#define PYTHON_CLASS_BEGIN_BASE(className,nonQualifiedClass)                                      \
    {                                                                                             \
        boost::python::scope scope##__COUNT__ =                                                   \
        boost::python::class_< nonQualifiedClass##PYTHON_CLASS_WRAPPER_BASE_NAME, boost::noncopyable >( className, boost::python::no_init )

#define PYTHON_CLASS_END \
        ;                \
    }

//
// namespace definition
//
#if defined(PLATFORM_WIN32)

#define PYTHON_NAMESPACE_BEGIN(namespaceName,fullyQualifiedNamespace) \
    {                                                                 \
        boost::python::scope scope##__COUNT__ = boost::python::class_< fullyQualifiedNamespace##::CORE_GET_NAMESPACE_ID >( namespaceName );

#elif defined(PLATFORM_OSX)

#define PYTHON_NAMESPACE_BEGIN(namespaceName,fullyQualifiedNamespace) \
{                                                                 \
boost::python::scope scope##__COUNT__ = boost::python::class_< fullyQualifiedNamespace::CORE_GET_NAMESPACE_ID >( namespaceName );

#endif

#define PYTHON_NAMESPACE_END \
    }

//
// enums
//
#define PYTHON_ENUM_BEGIN(enumName,fullyQualifiedEnum) \
    {                                                  \
        boost::python::scope scope##__COUNT__ = boost::python::enum_< fullyQualifiedEnum >( enumName )

#define PYTHON_ENUM_END \
        ;               \
    }


#define PYTHON_ENUM_VALUE(enumValueName,enumValue) \
        .value( enumValueName , enumValue )

            
//
// share
//
#define PYTHON_SHARED_CLASS( type ) \
    boost::python::register_ptr_to_python< boost::shared_ptr< type > >();


//
// method definition
//
#define PYTHON_CLASS_METHOD_CONSTRUCTOR( ... ) \
        .def( boost::python::init< __VA_ARGS__ >()[ boost::python::return_value_policy< boost::python::manage_new_object >() ] )

#define PYTHON_CLASS_METHOD(methodName,fullyQualifiedMethod) \
        .def( methodName , & fullyQualifiedMethod )

#define PYTHON_CLASS_METHOD_STATIC(methodName,fullyQualifiedMethod) \
        .def( methodName , & fullyQualifiedMethod ).staticmethod( methodName )

#define PYTHON_CLASS_METHOD_STATIC_FROM_POINTER(methodName,fullyQualifiedMethodPointer) \
        .def( methodName , fullyQualifiedMethodPointer ).staticmethod( methodName )

#define PYTHON_CLASS_METHOD_PURE_VIRTUAL(methodName,fullyQualifiedMethod) \
        .def( methodName , boost::python::pure_virtual( & fullyQualifiedMethod ) )

#if defined(PLATFORM_WIN32)

#define PYTHON_CLASS_METHOD_VIRTUAL(methodName,fullyQualifiedMethod,nonQualifiedClassName,nonQualifiedMethodName) \
.def( methodName , & fullyQualifiedMethod , & nonQualifiedClassName##PYTHON_CLASS_WRAPPER_BASE_NAME##::##nonQualifiedMethodName##PYTHON_CLASS_WRAPPER_BASE_FORWARDING_METHOD )

#elif defined(PLATFORM_OSX)

#define PYTHON_CLASS_METHOD_VIRTUAL(methodName,fullyQualifiedMethod,nonQualifiedClassName,nonQualifiedMethodName) \
    .def( methodName , & fullyQualifiedMethod , & nonQualifiedClassName##PYTHON_CLASS_WRAPPER_BASE_NAME::nonQualifiedMethodName##PYTHON_CLASS_WRAPPER_BASE_FORWARDING_METHOD )

#endif

#define PYTHON_CLASS_METHOD_SET_CALLBACK(methodName,fullyQualifiedMethod) \
    .def( methodName , & fullyQualifiedMethod##SET_CALLBACK_METHOD_HELPER )

#define PYTHON_CLASS_METHOD_GET_CALLBACK(methodName,fullyQualifiedMethod) \
    .def( methodName , & fullyQualifiedMethod##GET_CALLBACK_METHOD_HELPER , boost::python::return_value_policy< boost::python::return_by_value >() )

#define PYTHON_CLASS_METHOD_CALLBACK(methodName,fullyQualifiedMethod) \
    .def( methodName , & fullyQualifiedMethod##CALLBACK_METHOD_HELPER , boost::python::return_value_policy< boost::python::return_by_value >() )

#define PYTHON_CLASS_METHOD_GET_RAW_POINTER(methodName,fullyQualifiedClass) \
    .def( methodName , & content::Ref< fullyQualifiedClass >::GetSharedPtr , boost::python::return_value_policy< boost::python::return_by_value >() )
    
#define PYTHON_NAMESPACE_METHOD(methodName,fullyQualifiedMethod) \
    .def( methodName , & fullyQualifiedMethod ).staticmethod( methodName )

#define PYTHON_CLASS_FIELD_READONLY(fieldName,fullyQualifiedField) \
    .def_readonly( fieldName , & fullyQualifiedField )

#define PYTHON_CLASS_FIELD_READWRITE(fieldName,fullyQualifiedField) \
    .def_readwrite( fieldName , & fullyQualifiedField )

//
// overloading
//
#define PYTHON_DISAMBIGUATE_OVERLOADED_METHOD(fullyQualifiedMethod,returnType,...) \
    ( returnType (*)( __VA_ARGS__ ) )( & fullyQualifiedMethod )


//
// initialization
//
#if defined(PLATFORM_WIN32)

#define PYTHON_FORWARD_DECLARE_MODULE( moduleName ) \
    extern "C" void init##moduleName##();

#define PYTHON_INITIALIZE_MODULE( moduleName )                                          \
    {                                                                                   \
        init##moduleName##();                                                           \
        const char*                importString  = "import " #moduleName "\n";          \
        boost::python::object      mainModule    = boost::python::import( "__main__" ); \
        boost::python::api::object mainNamespace = mainModule.attr( "__dict__" );       \
        core::String s = importString;                                                  \
        boost::python::exec( importString, mainNamespace, mainNamespace );              \
    }

#elif defined(PLATFORM_OSX)

#define PYTHON_FORWARD_DECLARE_MODULE( moduleName ) \
extern "C" void init##moduleName();

#define PYTHON_INITIALIZE_MODULE( moduleName )                                          \
    {                                                                                   \
        init##moduleName();                                                           \
        const char*                importString  = "import " #moduleName "\n";          \
        boost::python::object      mainModule    = boost::python::import( "__main__" ); \
        boost::python::api::object mainNamespace = mainModule.attr( "__dict__" );       \
        core::String s = importString;                                                  \
        boost::python::exec( importString, mainNamespace, mainNamespace );              \
    }

#endif


#define PYTHON_TERMINATE_MODULE( moduleName )

#endif // include guard