from util import Utils


class Trainer:
    def __init__(self, params, utils, dl):
        self.params = params
        self.utils = utils
        self.dl = dl
        self.log_time = {}

    def train(self):
        embeddings = Utils(params=self.params, dl=self.dl).get_pre_trained_embeddings()
        print('-----------{}-------------'.format(self.params.config))
        training_time = self.utils.train(pretrained_emb=embeddings, save_plots_as=self.params.config)
        self.log_time[self.params.config] = training_time
        print('-----------------------------------------')
