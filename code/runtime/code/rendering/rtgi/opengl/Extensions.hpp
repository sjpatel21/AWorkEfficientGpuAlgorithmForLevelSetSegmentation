#ifndef RENDERING_RTGI_OPENGL_EXTENSIONS_HPP
#define RENDERING_RTGI_OPENGL_EXTENSIONS_HPP

#if defined(PLATFORM_WIN32)

#define NOMINMAX
#include <windows.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/wglext.h>
#include <GL/glext.h>

#elif defined(PLATFORM_OSX)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <OpenGL/glext.h>

#endif

#if defined(PLATFORM_WIN32)

extern PFNGLSTRINGMARKERGREMEDYPROC                    glStringMarkerGREMEDY;

extern PFNGLBINDBUFFERARBPROC                          glBindBufferARB;
extern PFNGLGENBUFFERSARBPROC                          glGenBuffersARB;
extern PFNGLBUFFERDATAARBPROC                          glBufferDataARB;
extern PFNGLDELETEBUFFERSARBPROC                       glDeleteBuffersARB;
extern PFNGLMAPBUFFERARBPROC                           glMapBufferARB;
extern PFNGLUNMAPBUFFERARBPROC                         glUnmapBufferARB;
                                
extern PFNGLACTIVETEXTUREARBPROC                       glActiveTextureARB;
extern PFNGLCLIENTACTIVETEXTUREARBPROC                 glClientActiveTextureARB;

extern PFNGLDRAWBUFFERSARBPROC                         glDrawBuffersARB;

extern PFNWGLSWAPINTERVALEXTPROC                       wglSwapIntervalEXT;

extern PFNGLISRENDERBUFFEREXTPROC                      glIsRenderbufferEXT;
extern PFNGLBINDRENDERBUFFEREXTPROC                    glBindRenderbufferEXT;
extern PFNGLDELETERENDERBUFFERSEXTPROC                 glDeleteRenderbuffersEXT;
extern PFNGLGENRENDERBUFFERSEXTPROC                    glGenRenderbuffersEXT;
extern PFNGLRENDERBUFFERSTORAGEEXTPROC                 glRenderbufferStorageEXT;
extern PFNGLGETRENDERBUFFERPARAMETERIVEXTPROC          glGetRenderbufferParameterivEXT;
extern PFNGLISFRAMEBUFFEREXTPROC                       glIsFramebufferEXT;
extern PFNGLBINDFRAMEBUFFEREXTPROC                     glBindFramebufferEXT;
extern PFNGLDELETEFRAMEBUFFERSEXTPROC                  glDeleteFramebuffersEXT;
extern PFNGLGENFRAMEBUFFERSEXTPROC                     glGenFramebuffersEXT;
extern PFNGLCHECKFRAMEBUFFERSTATUSEXTPROC              glCheckFramebufferStatusEXT;
extern PFNGLFRAMEBUFFERTEXTURE1DEXTPROC                glFramebufferTexture1DEXT;
extern PFNGLFRAMEBUFFERTEXTURE2DEXTPROC                glFramebufferTexture2DEXT;
extern PFNGLFRAMEBUFFERTEXTURE3DEXTPROC                glFramebufferTexture3DEXT;
extern PFNGLFRAMEBUFFERRENDERBUFFEREXTPROC             glFramebufferRenderbufferEXT;
extern PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC glGetFramebufferAttachmentParameterivEXT;
extern PFNGLGENERATEMIPMAPEXTPROC                      glGenerateMipmapEXT;

extern PFNGLTEXIMAGE3DPROC                             glTexImage3D;      
extern PFNGLTEXSUBIMAGE3DPROC                          glTexSubImage3D;      

extern PFNGLTEXBUFFEREXTPROC                           glTexBufferEXT;

extern PFNGLFRAMEBUFFERTEXTUREEXTPROC                  glFramebufferTextureEXT;

#elif defined(PLATFORM_OSX)

#define GL_TEXTURE_BUFFER_EXT 0x8C2A
void glTexBufferEXT( GLenum, GLenum, unsigned int );

#endif

#endif // RENDERING_RTGI_OPENGL_EXTENSIONS_HPP
