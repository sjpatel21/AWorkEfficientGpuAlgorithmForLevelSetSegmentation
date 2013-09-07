#include "python/Macros.hpp"

#include "math/Vector3.hpp"
#include "math/Utility.hpp"

namespace python
{

PYTHON_MODULE_BEGIN( math )

    PYTHON_METHOD_STATIC( "DegreesToRadians", math::DegreesToRadians )
    
    PYTHON_CLASS_BEGIN( "Vector3", math::Vector3 )
        PYTHON_CLASS_METHOD_CONSTRUCTOR( float, float, float )
    PYTHON_CLASS_END

PYTHON_MODULE_END

}