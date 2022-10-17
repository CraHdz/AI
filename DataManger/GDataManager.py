from  torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data



class GDataSet(InMemoryDataset):
    def __init__(self):
        pass

    @property
    def raw_file_names(self):
        return None


    def download(self):
        #如果需要下载，在这定义下载过程
        pass