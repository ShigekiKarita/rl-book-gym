import numpy


class SimpleImageViewer:
    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display

    def imshow(self, arr):
        import pyglet
        if self.window is None:
            height, width, _channels = arr.shape
            self.window = pyglet.window.Window(width=width, height=height,
                                               display=self.display, vsync=False,
                                               resizable=True)
            self.width = width
            self.height = height
            self.isopen = True

            @self.window.event
            def on_resize(width, height):
                self.width = width
                self.height = height

            @self.window.event
            def on_close():
                self.isopen = False

        assert len(arr.shape) == 3, "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(arr.shape[1], arr.shape[0],
                                       'RGB', arr.tobytes(), pitch=arr.shape[1]*-3)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0, 0, width=self.window.width, height=self.window.height)
        self.window.flip()

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()
