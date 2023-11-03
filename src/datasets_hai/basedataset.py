from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """Abstract method for learning to defer methods"""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """must at least have data_dir"""
        pass

    @abstractmethod
    def get_length(self):
        # should be able to call it before generate_data
        pass
    
    @abstractmethod
    def generate_data(self):
        """generates the data loader, called on init
        
        should generate the following must:
            self.d (dimension)
            self.k_classes (number of classes in target)
            self.ids
            self.data_x
            self.data_embs # 3d tensor first index captures which embedding
            self.which_emb # which embedding to index
            self.data_y
            self.ai_preds
            self.human_preds
            self.captions
            self.metric_y
            self.captions_embs
        """
        pass