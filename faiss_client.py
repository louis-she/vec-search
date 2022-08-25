from distributed_faiss.client import IndexClient, IndexCfg
import numpy as np
from tqdm import tqdm

client = IndexClient(
    "./faiss_distribute_config.txt",
)

index = client.create_index("test_index", IndexCfg(
    faiss_factory="OPQ16_64,IMI2x8,PQ8+16",
    dim=512,
    index_storage_dir="/home/featurize/faiss_client_index"
))

def random_float32_array(num, dimention):
    ret = np.zeros((num, dimention), np.float32)
    step = 100000
    for offset in tqdm(range(0, num, step)):
        n = min(num - offset, step)
        ret[offset:offset + n] = np.random.rand(n, dimention).astype(np.float32)
    return ret

