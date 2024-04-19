import torch
from torch import nn
from torchcrf import CRF
import torch.nn.functional as F
from .modeling_IFA import IFAModel
from transformers import BertConfig, BertModel, CLIPConfig, CLIPModel

class IFANERCRFModel(nn.Module):
    def __init__(self, label_list, args):
        super(IFANERCRFModel, self).__init__()
        self.args = args
        self.vision_config = CLIPConfig.from_pretrained(self.args.vit_name).vision_config
        self.text_config = BertConfig.from_pretrained(self.args.bert_name)
        self.text_config.pmtlen=args.pmtlen
        self.text_config.routnum = args.routnum
        self.vision_config.type = 'vision'
        self.text_config.type = 'text'

        clip_model_dict = CLIPModel.from_pretrained(self.args.vit_name).vision_model.state_dict()
        bert_model_dict = BertModel.from_pretrained(self.args.bert_name).state_dict()

        my_model_dict = torch.load("/home/oyhj/DeepKE/example/ner/multimodal/logs/实验数据/twitter-2015数据/twitter-15-74.99/checkpoints/twitter15/best_model.pth")

        print(self.vision_config)
        print(self.text_config)



        self.vision_config.device = args.device
        self.model = IFAModel(self.vision_config, self.text_config)

        self.num_labels  = len(label_list)  # pad
        self.crf = CRF(self.num_labels, batch_first=True)
        self.fc = nn.Linear(self.text_config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)

        model_dict = self.model.state_dict()
        model_dict_crf = self.crf.state_dict()
        model_dict_fc = self.fc.state_dict()
        for name in my_model_dict:
            if 'model' in name:
                name2 = name.replace('model.','')
                model_dict[name2] = my_model_dict[name]
            elif 'crf' in name:
                name2 = name.replace('crf.', '')
                model_dict_crf[name2] = my_model_dict[name]
            else:
                name2 = name.replace('fc.', '')
                model_dict_fc[name2] = my_model_dict[name]


        # load:
        # vision_names, text_names = [], []
        # model_dict = self.model.state_dict()
        # for name in model_dict:
        #     if 'vision' in name:
        #         clip_name = name.replace('vision_', '').replace('model.', '')
        #         if clip_name in clip_model_dict:
        #             vision_names.append(clip_name)
        #             model_dict[name] = clip_model_dict[clip_name]
        #     elif 'text' in name:
        #         text_name = name.replace('text_', '').replace('model.', '')
        #         if text_name in bert_model_dict:
        #             text_names.append(text_name)
        #             model_dict[name] = bert_model_dict[text_name]
        # assert len(vision_names) == len(clip_model_dict) and len(text_names) == len(bert_model_dict), \
        #             (len(vision_names), len(text_names), len(clip_model_dict), len(bert_model_dict))
        self.model.load_state_dict(model_dict)

        self.fc.load_state_dict(model_dict_fc)
        print(2)

    def forward(
            self, 
            input_ids=None, 
            attention_mask=None, 
            token_type_ids=None, 
            labels=None, 
            images=None, 
            aux_imgs=None,
            rcnn_imgs=None,
            prompt_label_token=None
    ):
        bsz = input_ids.size(0)

        last_hidden_state = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            prompt_label_token=prompt_label_token,
                            pixel_values=images,
                            aux_values=aux_imgs, 
                            rcnn_values=rcnn_imgs,
                            return_dict=True,)

        sequence_output = last_hidden_state              # bsz, len, hidden
        sequence_output = self.dropout(sequence_output)  # bsz, len, hidden
        emissions = self.fc(sequence_output)             # bsz, len, labels
        
        logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')  # 去掉CLS
            # loss = loss + cmdloss
            return logits, loss
        return logits, None