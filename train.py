from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config


def main():
    # create instance of config
    config = Config()
    
    # build model
    model = NERModel(config)
    model.build()
    # model.restore_session(config.dir_model) # optional, restore weights
    # model.reinitialize_weights("proj")
    
    # create datasets
    dev = CoNLLDataset(config.filename_dev, config.processing_word,
                       config.processing_tag, config.max_iter,config.max_len)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter,config.max_len)
    # train model
    model.train(train, dev)


if __name__ == "__main__":
    main()
