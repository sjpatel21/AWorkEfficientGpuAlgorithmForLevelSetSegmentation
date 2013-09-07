#ifndef PYTHON_BOOST_PYTHON_PREFIX_HPP
#define PYTHON_BOOST_PYTHON_PREFIX_HPP

//
// this is a bunch of stuff to get boost building properly
//
#define BOOST_PYTHON_STATIC_LIB
#define BOOST_PYTHON_STATIC_MODULE
#define BOOST_PYTHON_SOURCE
#define Py_NO_ENABLE_SHARED

#ifdef BUILD_DEBUG
// we need to hardcode the library name because the generalized boost autolink system doesn't know about
// the special -y flag specific to boost python that denotes the use of the debug build of the python interpreter
#define BOOST_AUTO_LINK_NOMANGLE
#if defined(PLATFORM_WIN32)
#define BOOST_LIB_NAME libboost_python-vc80-mt-sgyd-1_35
#elif defined(PLATFORM_OSX)
#define BOOST_LIB_NAME libboost_python-gcc42-mt-sd-1_35
#endif
#define BOOST_DEBUG_PYTHON
#endif

#ifdef BUILD_RELEASE
#define BOOST_LIB_NAME boost_python
#endif

#include <boost/config/auto_link.hpp>

#endif