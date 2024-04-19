import os
import hydra
import torch
import numpy as np
import random
from hydra import utils
from torch.utils.data import DataLoader
from deepke.name_entity_re.multimodal.models.IFA_model import IFANERCRFModel
from deepke.name_entity_re.multimodal.modules.dataset import MMPNERProcessor, MMPNERDataset
from deepke.name_entity_re.multimodal.modules.train import Trainer

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# import wandb
# writer = wandb.init(project="DeepKE_NER_MM")
writer=None

DATA_PATH = {
    'twitter15': {'train': '/home/oyhj/IdeaProjects/DeepKE-main/example/ner/multimodal/data/twitter2015/train.txt',
                  'dev': '/home/oyhj/IdeaProjects/DeepKE-main/example/ner/multimodal/data/twitter2015/valid.txt',
                  'test': '/home/oyhj/IdeaProjects/DeepKE-main/example/ner/multimodal/data/twitter2015/test.txt',
                  'train_auximgs': '/home/oyhj/IdeaProjects/DeepKE-main/example/ner/multimodal/data/twitter2015/twitter2015_train_dict.pth',
                  'dev_auximgs': '/home/oyhj/IdeaProjects/DeepKE-main/example/ner/multimodal/data/twitter2015/twitter2015_val_dict.pth',
                  'test_auximgs': '/home/oyhj/IdeaProjects/DeepKE-main/example/ner/multimodal/data/twitter2015/twitter2015_test_dict.pth',
                  'rcnn_img_path': '/home/oyhj/IdeaProjects/DeepKE-main/example/ner/multimodal/data/twitter2015',
                  'img2crop': '/home/oyhj/IdeaProjects/DeepKE-main/example/ner/multimodal/data/twitter2015/twitter15_detect/twitter15_img2crop.pth'},

    'twitter17': {'train': '/home/oyhj/IdeaProjects/DeepKE-main/example/ner/multimodal/data/twitter2017/train.txt',
                  'dev': '/home/oyhj/IdeaProjects/DeepKE-main/example/ner/multimodal/data/twitter2017/valid.txt',
                  'test': '/home/oyhj/IdeaProjects/DeepKE-main/example/ner/multimodal/data/twitter2017/test.txt',
                  'train_auximgs': '/home/oyhj/IdeaProjects/DeepKE-main/example/ner/multimodal/data/twitter2017/twitter2017_train_dict.pth',
                  'dev_auximgs': '/home/oyhj/IdeaProjects/DeepKE-main/example/ner/multimodal/data/twitter2017/twitter2017_val_dict.pth',
                  'test_auximgs': '/home/oyhj/IdeaProjects/DeepKE-main/example/ner/multimodal/data/twitter2017/twitter2017_test_dict.pth',
                  'rcnn_img_path': '/home/oyhj/IdeaProjects/DeepKE-main/example/ner/multimodal/data/twitter2017',
                  'img2crop': '/home/oyhj/IdeaProjects/DeepKE-main/example/ner/multimodal/data/twitter2017/twitter17_detect/twitter17_img2crop.pth'}
}

IMG_PATH = {
    'twitter15': '/home/oyhj/IdeaProjects/DeepKE-main/example/ner/multimodal/data/twitter2015/twitter2015_images',
    'twitter17': '/home/oyhj/IdeaProjects/DeepKE-main/example/ner/multimodal/data/twitter2017/twitter2017_images'
}

AUX_PATH = {
    'twitter15': {'train': '/home/oyhj/IdeaProjects/DeepKE-main/example/ner/multimodal/data/twitter2015/twitter2015_aux_images/train/crops',
                  'dev': '/home/oyhj/IdeaProjects/DeepKE-main/example/ner/multimodal/data/twitter2015/twitter2015_aux_images/val/crops',
                  'test': '/home/oyhj/IdeaProjects/DeepKE-main/example/ner/multimodal/data/twitter2015/twitter2015_aux_images/test/crops'},

    'twitter17': {'train': '/home/oyhj/IdeaProjects/DeepKE-main/example/ner/multimodal/data/twitter2017/twitter2017_aux_images/train/crops',
                  'dev': '/home/oyhj/IdeaProjects/DeepKE-main/example/ner/multimodal/data/twitter2017/twitter2017_aux_images/val/crops',
                  'test': '/home/oyhj/IdeaProjects/DeepKE-main/example/ner/multimodal/data/twitter2017/twitter2017_aux_images/test/crops'}
}
# #twitter2015最好-74.99
# # label_lr = [1e-5]  # 1e-3
# # weight_decay = [5e-3]  # 1e-3
# # infer_lr = [5e-4]  # 1e-2
# # weight_decay_infer = [5e-3]  # 1e-4

# #twitter2017最好-87.92
# label_lr = [5e-5]  # 1e-3
# weight_decay = [5e-4]  # 1e-3
# infer_lr = [1e-3]  # 1e-2
# weight_decay_infer = [5e-4]  # 1e-4


LABEL_LIST = ["[CLS]", "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[SEP]", "X"]

def set_seed(seed=2024):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

@hydra.main(config_path="./conf", config_name='config.yaml')
def main(cfg):
    cwd = utils.get_original_cwd()
    cfg.cwd = cwd
    print(cfg)
    weight_decay = [1e-4,2e-4,3e-4]  # 1e-3
    label_lr = [2e-4,1e-4,9e-5]  # 1e-3
    weight_decay_infer = [1e-5,2e-5,3e-5]  # 1e-4
    infer_lr = [2e-5,1e-5,9e-4]  # 1e-2
    # #twitter-2017-choose-bestxiang
    # label_lr = [5e-5,6e-5]  # 1e-3    1
    # weight_decay = [4e-4,5e-4,6e-4]  # 1e-3    3
    # infer_lr = [9e-4,1e-3,2e-3]  # 1e-2    2
    # weight_decay_infer = [4e-4,5e-4,6e-4]  # 1e-4    2

    # label_lr = [1e-5]  # 1e-3
    # weight_decay = [5e-3]  # 1e-3
    # infer_lr = [5e-4]  # 1e-2
    # weight_decay_infer = [5e-3]  # 1e-4

    # label_lr = [5e-5]  # 1e-3
    # weight_decay = [5e-4]  # 1e-3
    # infer_lr = [1e-3]  # 1e-2
    # weight_decay_infer = [5e-4]  # 1e-4
    pmtlen=[8]
    routnum=[1,2,4,5,6,7]
    # rates=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # pmts=["I have seven classes tomorrow. ","kosad iksjd uhsjd","","Distinguish the type of each entity as ","Can you choose a category from the following types to classify each entity? ","Which of the following types is the type of each entity? "]
    #twitter2015选参数
    # pmts = ["Your task is to choose one of these four types as the entity type in the text: ",
    #         "Which of the following types is the type of each entity? ",
    #         "Choose one of the following types as the type for each entity: "]

    # label_lr = [1e-3,5e-4,1e-4,5e-3]  # 1e-3
    # weight_decay = [1e-3,1e-2,1e-4]  # 1e-3
    # infer_lr = [1e-2,1e-3,1e-4]  # 1e-2
    # weight_decay_infer = [3e-4,1e-3,1e-2]  # 1e-4

    # #twitter2017选参数
    # label_lr = [9e-5,1e-4,2e-4]  # 1e-3
    # weight_decay = [1e-3,2e-3,9e-4]  # 1e-3
    # infer_lr = [2e-4,1e-4,9e-5]  # 1e-2
    # weight_decay_infer = [9e-4,2e-3,1e-3]  # 1e-4

    cfg.bert_name = "/home/oyhj/IdeaProjects/DeepKE-main/src/deepke/name_entity_re/multimodal/modules/bert-base-uncased"
    cfg.vit_name = "/home/oyhj/IdeaProjects/DeepKE-main/src/deepke/name_entity_re/multimodal/modules/clip-vit-base-patch32"
    cfg.dataset_name = 'twitter15'

    for sh_dc in weight_decay:
        for label_lrx in label_lr:
            for share_dc in weight_decay_infer:
                for lr_cmd in infer_lr:

                    # for rate in rates:
                    # for pmt in pmts:
                    set_seed(cfg.seed)  # set seed, default is 1
                    if cfg.save_path is not None:  # make save_path dir
                        if not os.path.exists(cfg.save_path):
                            os.makedirs(cfg.save_path, exist_ok=True)
                    cfg.label_lr = label_lrx
                    cfg.label_weight_decay = sh_dc
                    cfg.infer_lr = lr_cmd
                    cfg.infer_weight_decay = share_dc
                    # cfg.rate = rate
                    # cfg.pmtlen=len
                    # cfg.pmts=pmt
                    print(cfg)
                    label_mapping = {label: idx for idx, label in enumerate(LABEL_LIST, 0)}
                    # label_mapping["PAD"] = 0
                    data_path, img_path, aux_path = DATA_PATH[cfg.dataset_name], IMG_PATH[cfg.dataset_name], AUX_PATH[
                        cfg.dataset_name]
                    rcnn_img_path = DATA_PATH[cfg.dataset_name]['rcnn_img_path']

                    processor = MMPNERProcessor(data_path, cfg)
                    train_dataset = MMPNERDataset(processor, label_mapping, img_path, aux_path, rcnn_img_path,
                                                  max_seq=cfg.max_seq, ignore_idx=cfg.ignore_idx, aux_size=cfg.aux_size,
                                                  rcnn_size=cfg.rcnn_size, mode='train', cwd=cwd)
                    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4,
                                                  pin_memory=True)

                    dev_dataset = MMPNERDataset(processor, label_mapping, img_path, aux_path, rcnn_img_path,
                                                max_seq=cfg.max_seq, ignore_idx=cfg.ignore_idx, aux_size=cfg.aux_size,
                                                rcnn_size=cfg.rcnn_size, mode='dev', cwd=cwd)
                    dev_dataloader = DataLoader(dev_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4,
                                                pin_memory=True)

                    test_dataset = MMPNERDataset(processor, label_mapping, img_path, aux_path, rcnn_img_path,
                                                 max_seq=cfg.max_seq, ignore_idx=cfg.ignore_idx, aux_size=cfg.aux_size,
                                                 rcnn_size=cfg.rcnn_size, mode='test', cwd=cwd)
                    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4,
                                                 pin_memory=True)

                    prompt_label_token = processor.initPrompt()

                    model = IFANERCRFModel(LABEL_LIST, cfg)

                    trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader,
                                      test_data=test_dataloader, model=model,
                                      label_map=label_mapping, args=cfg,
                                      logger=logger, writer=writer, prompt_label_token=prompt_label_token)
                    trainer.train()


if __name__ == '__main__':
    main()