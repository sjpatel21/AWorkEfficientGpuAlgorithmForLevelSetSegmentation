#include "rendering/rtgi/opengl/Extensions.hpp"

#if defined(PLATFORM_WIN32)
// debug
PFNGLSTRINGMARKERGREMEDYPROC                    glStringMarkerGREMEDY           = NULL;

// buffer functions
PFNGLBINDBUFFERARBPROC                          glBindBufferARB                 = NULL;
PFNGLGENBUFFERSARBPROC                          glGenBuffersARB                 = NULL;
PFNGLBUFFERDATAARBPROC                          glBufferDataARB                 = NULL;
PFNGLDELETEBUFFERSARBPROC                       glDeleteBuffersARB              = NULL;
PFNGLMAPBUFFERARBPROC                           glMapBufferARB                  = NULL;
PFNGLUNMAPBUFFERARBPROC                         glUnmapBufferARB                = NULL;
PFNGLDRAWBUFFERSARBPROC                         glDrawBuffersARB                = NULL;

// multi-texture functions
PFNGLACTIVETEXTUREARBPROC                       glActiveTextureARB              = NULL;
PFNGLCLIENTACTIVETEXTUREARBPROC                 glClientActiveTextureARB        = NULL;
                                                
// vsync control                                
PFNWGLSWAPINTERVALEXTPROC                       wglSwapIntervalEXT              = NULL;

// frame buffer object
PFNGLISRENDERBUFFEREXTPROC                      glIsRenderbufferEXT             = NULL;
PFNGLBINDRENDERBUFFEREXTPROC                    glBindRenderbufferEXT           = NULL;
PFNGLDELETERENDERBUFFERSEXTPROC                 glDeleteRenderbuffersEXT        = NULL;
PFNGLGENRENDERBUFFERSEXTPROC                    glGenRenderbuffersEXT           = NULL;
PFNGLRENDERBUFFERSTORAGEEXTPROC                 glRenderbufferStorageEXT        = NULL;
PFNGLGETRENDERBUFFERPARAMETERIVEXTPROC          glGetRenderbufferParameterivEXT = NULL;
PFNGLISFRAMEBUFFEREXTPROC                       glIsFramebufferEXT              = NULL;
PFNGLBINDFRAMEBUFFEREXTPROC                     glBindFramebufferEXT            = NULL;
PFNGLDELETEFRAMEBUFFERSEXTPROC                  glDeleteFramebuffersEXT         = NULL;
PFNGLGENFRAMEBUFFERSEXTPROC                     glGenFramebuffersEXT            = NULL;
PFNGLCHECKFRAMEBUFFERSTATUSEXTPROC              glCheckFramebufferStatusEXT     = NULL;
PFNGLFRAMEBUFFERTEXTURE1DEXTPROC                glFramebufferTexture1DEXT       = NULL;
PFNGLFRAMEBUFFERTEXTURE2DEXTPROC                glFramebufferTexture2DEXT       = NULL;
PFNGLFRAMEBUFFERTEXTURE3DEXTPROC                glFramebufferTexture3DEXT       = NULL;
PFNGLFRAMEBUFFERRENDERBUFFEREXTPROC             glFramebufferRenderbufferEXT    = NULL;
PFNGLGENERATEMIPMAPEXTPROC                      glGenerateMipmapEXT             = NULL;
PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC glGetFramebufferAttachmentParameterivEXT = NULL;

// 3d textures
PFNGLTEXIMAGE3DPROC                             glTexImage3D                    = NULL;
PFNGLTEXSUBIMAGE3DPROC                          glTexSubImage3D                 = NULL;

// buffer textures
PFNGLTEXBUFFEREXTPROC                           glTexBufferEXT                  = NULL;

// geometry shader with 3D texture bound to drame buffer
PFNGLFRAMEBUFFERTEXTUREEXTPROC                  glFramebufferTextureEXT         = NULL;

#elif defined(PLATFORM_OSX)

#if !defined(GL_EXT_texture_buffer_object) && !GL_EXT_texture_buffer_object
void glTexBufferEXT( GLenum, GLenum, unsigned int ) { /* no-op */ };
#endif

#endif
