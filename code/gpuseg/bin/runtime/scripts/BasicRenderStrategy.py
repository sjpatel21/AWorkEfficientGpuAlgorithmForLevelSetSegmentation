def Initialize( engine ):

    global gInventory

    print ''
    print ''
    print 'Initializing BasicRenderStrategy...'
    print ''
    print ''
        
    gInventory = content.CreateInventory()

    content.ParameterManager.Initialize()
    content.LoadManager.Load( 'runtime/art/BasicRenderStrategyScene.dae', gInventory )

    camera = gInventory.FindRenderingCamera( 'cameraShape1' )
    scene  = gInventory.FindRenderingScene( 'untitled' )
    
    rendering.Context.SetCurrentCamera( camera )
    rendering.Context.SetCurrentScene( scene )
    
    basicRenderStrategy = rendering.CreateBasicRenderStrategy()
    basicRenderStrategy.SetRenderCallback( engine.RenderCallback() )
    basicRenderStrategy.SetClearColor( rendering.rtgi.ColorRGBA( 0.5, 0.5, 0.5, 1 ) )
    
    rendering.Context.SetCurrentRenderStrategy( basicRenderStrategy )

def Terminate():
    
    global gInventory    
    
    print ''
    print ''
    print 'Terminating BasicRenderStrategy...'
    print ''
    print ''
    
    rendering.Context.SetCurrentRenderStrategy( None )
    rendering.Context.SetCurrentScene( content.RefRenderingScene() )
    rendering.Context.SetCurrentCamera( content.RefRenderingCamera() )

    content.LoadManager.Unload( gInventory )

    gInventory = None

    content.ParameterManager.Terminate()
    

def Update( timeDeltaSeconds ):
    pass
    
    
def RenderCallback():
    pass
    
    
