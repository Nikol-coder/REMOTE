import sys
sys.path.append("..")
import os
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertConfig, CLIPConfig, BertModel
from .modeling_unimo_moe_opt import UnimoModel      
from .modeling_clip import CLIPModel                    
import numpy as np
# from utils import *


class UnimoREModel(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(UnimoREModel, self).__init__()
        self.args = args
        # print(vision_config)
        # print(text_config)
        clip_model = CLIPModel.from_pretrained(args.vit_name, ignore_mismatched_sizes=True)
        clip_vit = clip_model.vision_model
        vision_config = CLIPConfig.from_pretrained(args.vit_name).vision_config
        text_config = BertConfig.from_pretrained(args.bert_name)
        bert = BertModel.from_pretrained(args.bert_name)
        clip_model_dict = clip_vit.state_dict()
        bert_model_dict = bert.state_dict()

        self.vision_config = vision_config
        self.text_config = text_config

        # for re
        vision_config.device = args.device
        self.model = UnimoModel(vision_config, text_config)

        vision_names, text_names = [], []
        model_dict = self.model.state_dict()

        avg_conv_weight = torch.mean(clip_model_dict['embeddings.patch_embedding.weight'], dim=1).unsqueeze(1)
        clip_model_dict['embeddings.depth_embedding.weight'] = torch.tensor(avg_conv_weight.data.clone())
        for name in model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]
                else:
                    print(f"Vision weight not found: {clip_name}")
            elif 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '')
                if text_name in bert_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = bert_model_dict[text_name]
                else:
                    print(f"Text weight not found: {text_name}")

        missing_vision_weights = set(clip_model_dict.keys()) - set(vision_names)
        missing_text_weights = set(bert_model_dict.keys()) - set(text_names)
        if missing_vision_weights:
            print(f"Missing vision weights: {missing_vision_weights}")
        if missing_text_weights:
            print(f"Missing text weights: {missing_text_weights}")
            
        assert len(vision_names) == len(clip_model_dict) and len(text_names) == len(bert_model_dict), \
            (len(vision_names), len(text_names), len(clip_model_dict), len(bert_model_dict))
        self.model.load_state_dict(model_dict)

        self.model.resize_token_embeddings(len(tokenizer))
        self.args = args

        # self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.text_config.hidden_size * 2, num_labels)
        self.mm_linear = nn.Linear(self.text_config.hidden_size * 2, self.text_config.hidden_size)

        #The entity identifiers at the beginning and end of the text, used for extracting entity features. Among them, the text features and the caption text features of the image can both be represented by <s> and <o>.
        self.head_entity_start = tokenizer.convert_tokens_to_ids("<s1>")
        self.tail_entity_start = tokenizer.convert_tokens_to_ids("<s2>")
        self.head_object_start = tokenizer.convert_tokens_to_ids("<o1>")
        self.tail_object_start = tokenizer.convert_tokens_to_ids("<o2>")
        self.head_knowledge_start = tokenizer.convert_tokens_to_ids("<k1>")
        self.tail_knowledge_start = tokenizer.convert_tokens_to_ids("<k2>")
        self.tokenizer = tokenizer

        self.output_attentions = True
        self.output_hidden_states = True

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            org_image=None,
            ent_imgs=None,
            ent_idx_head=None,
            ent_idx_tail=None,
            position=None,
    ):
        # dataset
        output, vision_output = self.model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        aux_values=ent_imgs,
                                        position=position,
                                        output_attentions=True, 
                                        output_hidden_states=True,
                                        return_dict=True, )
        bsz, seq_len, hidden_size = output.shape
    
        entity_hidden_state = torch.Tensor(bsz, hidden_size * 2)  # batch, 2*hidden
        for i in range(bsz):
        
            head_entity_idx = input_ids[i].eq(self.head_entity_start).nonzero()
            if head_entity_idx.numel() > 0:
                head_entity_idx = head_entity_idx.item()
            else:
                head_entity_idx = -1  
        

            tail_entity_idx = input_ids[i].eq(self.tail_entity_start).nonzero()
            if tail_entity_idx.numel() > 0:
                tail_entity_idx = tail_entity_idx.item()
            else:
                tail_entity_idx = -1  
                
            if head_entity_idx != -1:
                head_entity_hidden = output[i, head_entity_idx, :].squeeze()
            else:
                head_entity_hidden = torch.zeros_like(output[i, 0, :])
            if tail_entity_idx != -1:
                tail_entity_hidden = output[i, tail_entity_idx, :].squeeze()
            else:
                tail_entity_hidden = torch.zeros_like(output[i, 0, :])
            
            if ent_idx_head[i] != -1:
                vision_hidden_head = vision_output[i, ent_idx_head[i], :]
            else:
                
                vision_hidden_head = torch.zeros_like(vision_output[i, 0, :])  
            if ent_idx_tail[i] != -1:
                vision_hidden_tail = vision_output[i, ent_idx_tail[i], :]
            else:
                vision_hidden_tail = torch.zeros_like(vision_output[i, 0, :])  
        
            head_knowledge_idx = input_ids[i].eq(self.head_knowledge_start).nonzero()
            tail_knowledge_idx = input_ids[i].eq(self.tail_knowledge_start).nonzero()
            head_knowledge_idx = head_knowledge_idx.item() if head_knowledge_idx.numel() > 0 else -1
            tail_knowledge_idx = tail_knowledge_idx.item() if tail_knowledge_idx.numel() > 0 else -1
            
            head_knowledge_hidden = output[i, head_knowledge_idx, :].squeeze() if head_knowledge_idx != -1 else torch.zeros_like(output[i, 0, :])
            tail_knowledge_hidden = output[i, tail_knowledge_idx, :].squeeze() if tail_knowledge_idx != -1 else torch.zeros_like(output[i, 0, :])
            

            count_ones = (head_entity_idx == -1) + (tail_entity_idx == -1) + (ent_idx_head[i] == -1) + (ent_idx_tail[i] == -1)
            
            if count_ones > 2:
                # print("ent_idx_head:",ent_idx_head)
                # print("ent_idx_tail:",ent_idx_tail)
                print(head_entity_idx ,tail_entity_idx ,ent_idx_head[i] ,ent_idx_tail[i])
                print("input_ids:", input_ids[i])
                print("wrong data format, please check the dataset!")
                exit()

            if self.args.use_cap:
            
                head_object_idx = input_ids[i].eq(self.head_object_start).nonzero()
                if head_object_idx.numel() > 0:
                    head_object_idx = head_object_idx.item()
                else:
                    head_object_idx = -1  
                tail_object_idx = input_ids[i].eq(self.tail_object_start).nonzero()
                if tail_object_idx.numel() > 0:
                    tail_object_idx = tail_object_idx.item()
                else:
                    tail_object_idx = -1  
                
                
                if head_object_idx != -1:
                    head_object_hidden = output[i, head_object_idx, :].squeeze()
                else:
                    head_object_hidden = torch.zeros_like(output[i, 0, :])
                if tail_object_idx != -1:
                    tail_object_hidden = output[i, tail_object_idx, :].squeeze()
                else:
                    tail_object_hidden = torch.zeros_like(output[i, 0, :])    
                
                mm_head_hidden = []
                mm_tail_hidden = []
        
                #head entity
                if head_entity_idx != -1 and ent_idx_head[i] == -1 and head_object_idx == -1: #text-text
                    head_entity_hidden = self.mm_linear(torch.cat([head_entity_hidden, head_knowledge_hidden], dim=-1))
                    mm_head_hidden = []
                elif head_entity_idx != -1 and ent_idx_head[i] == -1 and head_object_idx == -1: #text-image
                    head_entity_hidden = self.mm_linear(torch.cat([head_entity_hidden, head_knowledge_hidden], dim=-1))
                    mm_head_hidden = []
                elif head_entity_idx == -1 and ent_idx_head[i] != -1 and head_object_idx != -1: #image-text
                    mm_head_hidden = self.mm_linear(torch.cat([vision_hidden_head, head_object_hidden], dim=-1))
                elif head_entity_idx == -1 and ent_idx_head[i] != -1 and head_object_idx != -1: #image-image
                    mm_head_hidden = self.mm_linear(torch.cat([vision_hidden_head, head_object_hidden], dim=-1))
                    
                #tail entity
                if tail_entity_idx != -1 and ent_idx_tail[i] == -1 and tail_object_idx == -1:#text-text
                    tail_entity_hidden = self.mm_linear(torch.cat([tail_entity_hidden, tail_knowledge_hidden], dim=-1))
                    mm_tail_hidden = []
                elif tail_entity_idx != -1 and ent_idx_tail[i] == -1 and tail_object_idx == -1:#text-image
                    tail_entity_hidden = self.mm_linear(torch.cat([tail_entity_hidden, tail_knowledge_hidden], dim=-1))
                    mm_tail_hidden = []
                elif tail_entity_idx == -1 and ent_idx_tail[i] != -1 and tail_object_idx != -1:#image-text
                    mm_tail_hidden = self.mm_linear(torch.cat([vision_hidden_tail, tail_object_hidden], dim=-1))
                elif tail_entity_idx == -1 and ent_idx_tail[i] != -1 and tail_object_idx != -1:#image-image
                    mm_tail_hidden = self.mm_linear(torch.cat([vision_hidden_tail, tail_object_hidden], dim=-1))
                    
                if isinstance(mm_head_hidden, list):
                    mm_head_hidden = torch.tensor(mm_head_hidden)
                if isinstance(mm_tail_hidden, list):
                    mm_tail_hidden = torch.tensor(mm_tail_hidden)
                    
                if mm_head_hidden.numel() > 0 and mm_tail_hidden.numel() > 0:  # 图-图
                    entity_hidden_state[i] = torch.cat([mm_head_hidden, mm_tail_hidden], dim=-1)
                elif mm_head_hidden.numel() == 0 and mm_tail_hidden.numel() > 0:  # 文-图
                    entity_hidden_state[i] = torch.cat([head_entity_hidden, mm_tail_hidden], dim=-1)
                elif mm_head_hidden.numel() > 0 and mm_tail_hidden.numel() == 0:  # 图-文
                    entity_hidden_state[i] = torch.cat([mm_head_hidden, tail_entity_hidden], dim=-1)
                elif mm_head_hidden.numel() == 0 and mm_tail_hidden.numel() == 0:  # 文-文
                    entity_hidden_state[i] = torch.cat([head_entity_hidden, tail_entity_hidden], dim=-1)
                else:
                    print("wrong data format, please check the dataset!")
                    exit()
                    
            else:
                if head_entity_idx != -1 and tail_entity_idx != -1 and ent_idx_head[i] == -1 and ent_idx_tail[i] == -1: 
                    entity_hidden_state[i] = torch.cat([head_entity_hidden, tail_entity_hidden], dim=-1)
                elif head_entity_idx != -1 and tail_entity_idx == -1 and ent_idx_head[i] == -1 and ent_idx_tail[i] != -1: 
                    entity_hidden_state[i] = torch.cat([head_entity_hidden, vision_hidden_tail], dim=-1)
                elif head_entity_idx == -1 and tail_entity_idx != -1 and ent_idx_head[i] != -1 and ent_idx_tail[i] == -1: 
                    entity_hidden_state[i] = torch.cat([vision_hidden_head, tail_entity_hidden], dim=-1)
                elif head_entity_idx == -1 and tail_entity_idx == -1 and ent_idx_head[i] != -1 and ent_idx_tail[i] != -1: 
                    entity_hidden_state[i] = torch.cat([vision_hidden_head, vision_hidden_tail], dim=-1)  
                else:
                    #nocap
                    print(head_entity_idx ,tail_entity_idx ,ent_idx_head[i] ,ent_idx_tail[i])
                    print("wrong dataset! nocap")
                    exit()

                # entity_hidden_state[i] = torch.cat([head_hidden, vision_hidden], dim=-1)
                
        entity_hidden_state = entity_hidden_state.to(self.args.device)
        logits = self.classifier(entity_hidden_state)
        # print("output:",output.shape)
        # print("vision_output:",vision_output.shape)
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(logits, labels.view(-1)), logits
        return logits
