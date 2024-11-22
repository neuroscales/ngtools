from ngtools.fileserver import LocalFileServer


server = LocalFileServer(1234)
server.serve_forever()
