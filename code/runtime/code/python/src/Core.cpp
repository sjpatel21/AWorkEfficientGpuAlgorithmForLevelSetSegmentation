#include "core/Printf.hpp"
#include "core/Functor.hpp"
#include "core/String.hpp"

#include "python/Macros.hpp"

namespace python
{

//
// core string
//
struct CoreStringToPython
{
    static PyObject* convert( core::String const& coreString );
};

struct CoreStringFromPython
{
    CoreStringFromPython();

    static void* PyObjectIsConvertable ( PyObject* pyObject );
    static void  PyObjectFromCoreString( PyObject* pyObject, boost::python::converter::rvalue_from_python_stage1_data* data );
};

//
// core functor
//
struct CoreFunctorContainerFromPython
{
    CoreFunctorContainerFromPython();

    static void* PyObjectIsConvertable           ( PyObject* pyObject );
    static void  PyObjectFromCoreFunctorContainer( PyObject* pyObject, boost::python::converter::rvalue_from_python_stage1_data* data );
};


PYTHON_MODULE_BEGIN( core )

    // custom to-python converter code
    boost::python::to_python_converter< core::String, CoreStringToPython >();

    // custom from-python converter code
    CoreStringFromPython();
    CoreFunctorContainerFromPython();

    // required to overload python's sys.stderr
    PYTHON_METHOD_STATIC( "write",  core::Printf )

    // normal definitions
    PYTHON_CLASS_BEGIN( "FunctorContainer", core::FunctorContainer )
    PYTHON_CLASS_END

    PYTHON_SHARED_CLASS( core::FunctorContainer )

    PYTHON_METHOD_STATIC( "Printf", core::Printf )

PYTHON_MODULE_END

//
// core string implementation
//
PyObject* CoreStringToPython::convert( core::String const& coreString )
    {
        return boost::python::incref( boost::python::object( coreString.ToAscii() ).ptr() );
    }

    CoreStringFromPython::CoreStringFromPython()
    {
        boost::python::converter::registry::push_back(
            &PyObjectIsConvertable,
            &PyObjectFromCoreString,
            boost::python::type_id< core::String >() );
    }

    void* CoreStringFromPython::PyObjectIsConvertable( PyObject* pyObject )
    {
        if ( !PyString_Check( pyObject ) )
        {
            return 0;
        }
        else
        {
            return pyObject;
        }
    }

    void CoreStringFromPython::PyObjectFromCoreString( PyObject* pyObject, boost::python::converter::rvalue_from_python_stage1_data* data )
    {
        const char* value = PyString_AsString( pyObject );
        if ( value == 0 )
        {
            boost::python::throw_error_already_set();
        }

        void* storage = ( ( boost::python::converter::rvalue_from_python_storage< core::String >* ) data )->storage.bytes;
        new ( storage ) core::String( value );
        data->convertible = storage;
    }



    //
    // core functor implementation
    //
    CoreFunctorContainerFromPython::CoreFunctorContainerFromPython()
    {
        boost::python::converter::registry::push_back(
            &PyObjectIsConvertable,
            &PyObjectFromCoreFunctorContainer,
            boost::python::type_id< boost::shared_ptr< core::FunctorContainer > >() );
    }

    void* CoreFunctorContainerFromPython::PyObjectIsConvertable( PyObject* pyObject )
    {
        if ( !PyMethod_Check( pyObject ) )
        {
            return 0;
        }
        else
        {
            return pyObject;
        }
    }

    void CoreFunctorContainerFromPython::PyObjectFromCoreFunctorContainer(
        PyObject* pyObject,
        boost::python::converter::rvalue_from_python_stage1_data* data )
    {
        PyObject* selfPyObject         = PyMethod_Self( pyObject );
        PyObject* functionPyObject     = PyMethod_Function( pyObject );

        boost::python::handle<> selfHandle         ( selfPyObject );
        boost::python::handle<> functionHandle     ( functionPyObject );

        boost::python::object self        ( selfHandle );
        boost::python::object function    ( functionHandle );
        boost::python::object functionName( function.attr( "__name__" ) );

        boost::python::object  functorContainer = self.attr( PyString_AsString( functionName.ptr() ) )();

        boost::shared_ptr< core::FunctorContainer > functorContainerRaw = 
            boost::python::extract< boost::shared_ptr< core::FunctorContainer > >( functorContainer );

        void* storage =
            ( ( boost::python::converter::rvalue_from_python_storage< boost::shared_ptr< core::FunctorContainer > >* ) data )->storage.bytes;

        new ( storage ) boost::shared_ptr< core::FunctorContainer >( functorContainerRaw );
        data->convertible = storage;
    }
}