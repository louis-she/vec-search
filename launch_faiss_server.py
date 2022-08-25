from distributed_faiss.server import IndexServer
import sys

server = IndexServer(int(sys.argv[1]), "/home/featurize/faiss_server_index")
server.start_blocking(9977, load_index=False)
