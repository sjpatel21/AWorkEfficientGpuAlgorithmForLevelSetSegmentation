#include "userinputhandlers/UserInputHandler.hpp"

#include "core/Assert.hpp"

#include "Engine.hpp"

#include "widgets/RenderingWindow.hpp"

UserInputHandler::UserInputHandler(
    RenderingWindow* renderingWindow,
    Engine*          engine ) :
mRenderingWindow( renderingWindow ),
mEngine        ( engine )
{
    Assert( mRenderingWindow != NULL );
    Assert( mEngine          != NULL );

    mEngine->AddRef();
}

UserInputHandler::~UserInputHandler()
{
    mEngine->Release();
    mEngine = NULL;
}

Engine* UserInputHandler::GetEngine()
{
    return mEngine;
}

RenderingWindow* UserInputHandler::GetRenderingWindow()
{
    return mRenderingWindow;
}