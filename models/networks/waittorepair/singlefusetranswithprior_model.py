from .singlefuse_model import SingleFuseModel


class SingleFuseTransWithPriorModel(SingleFuseModel):
    def __init__(self, opt):
        super(SingleFuseTransWithPriorModel, self).__init__(opt)

    def backward(self):
        self.backword_single()

