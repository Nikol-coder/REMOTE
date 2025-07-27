import random
import os
import numpy
import torch
import json
import ast
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import logging
from transformers.models.clip import CLIPProcessor
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class MOREProcessor(object):
    def __init__(self, data_path, re_path, bert_name, vit_name):
        self.data_path = data_path
        self.re_path = re_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)
        
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<s1>', '</s1>', '<s2>', '</s2>', '<k1>', '</k1>', '<k2>', '</k2>', '<o1>', '</o1>', '<o2>', '</o2>']})
        self.clip_processor = CLIPProcessor.from_pretrained(vit_name)
        self.ent_processor = CLIPProcessor.from_pretrained(vit_name)

    def load_from_file(self, mode="train"):
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))
        # txt
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            words, relations, heads, tails, imgids, dataid = [], [], [], [], [], []
            for i, line in enumerate(lines):
                line = ast.literal_eval(line)  # str to dict
                words.append(line['token'])
                relations.append(line['relation'])
                heads.append(line['h'])  # text {name, pos} 或 image {name, box}
                tails.append(line['t'])  # text {name, pos} 或 image {name, box}
                imgids.append(line['img_id'])
                dataid.append(i)

        assert len(words) == len(relations) == len(heads) == len(tails) == (len(imgids))
        # print((len(imgids)))
        # exit()
        
        #ent_dict 
        ent_path = self.data_path[mode + "_ent_dict"]
        
        with open(ent_path, 'r') as f:
            ent_imgs = json.load(f)

        """
        ent_imgs[01ba096b-11c5-51d6-adb0-0ba0a1cbe2c7.jpg]: {'[OBJ0]': (0.0158, 0.625, 0.0316, 0.395), '[OBJ1]': (0.5558, 0.6375, 0.1083, 0.35)}
        ent_imgs,即ent_dict 为，图片中的实体位置
        """
        
        return {'words': words, 'relations': relations, 'heads': heads, 'tails': tails, 'imgids': imgids,
                'dataid': dataid, 'ent_dict': ent_imgs}

    def get_relation_dict(self):
        with open(self.re_path, 'r', encoding="utf-8") as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        return re_dict

    def get_rel2id(self, train_path):
        with open(self.re_path, 'r', encoding="utf-8") as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        re2id = {key: [] for key in re_dict.keys()}
        with open(train_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = ast.literal_eval(line)  # str to dict
                assert line['relation'] in re2id
                re2id[line['relation']].append(i)
        return re2id


class MOREDataset(Dataset):
    def __init__(self, processor, img_path=None, dep_path=None, cap_path=None, args=None, mode="train") -> None:
        self.processor = processor
        self.max_seq = args.max_seq
        self.args = args
        self.img_path = img_path[mode] if img_path is not None else img_path
        self.dep_path = dep_path[mode] if dep_path is not None else dep_path
        self.cap_path = cap_path[mode] if cap_path is not None else cap_path
        self.cap_dict = json.load(open(self.cap_path, 'r'))
        # self.img_num = 10
        self.img_num = 12 #OBJ is range in 0~11
        self.mode = mode
        self.data_dict = self.processor.load_from_file(mode)

        self.re_dict = self.processor.get_relation_dict()
        self.tokenizer = self.processor.tokenizer
        self.clip_processor = self.processor.clip_processor
        self.ent_processor = self.processor.ent_processor
        self.ent_processor.feature_extractor.size, self.ent_processor.feature_extractor.crop_size = 64, 64
        self.crop_size = self.ent_processor.feature_extractor.crop_size

    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        word_list, relation, head_d, tail_d, imgid = self.data_dict['words'][idx], self.data_dict['relations'][idx], \
                                                    self.data_dict['heads'][idx], self.data_dict['tails'][idx], self.data_dict['imgids'][idx]
        
        head_pos, tail_pos = head_d['pos'], tail_d['pos']

        extend_word_list = []
        
        for i in range(len(word_list) + 1):
            if len(head_pos) == 2:
                if i == head_pos[0]:
                    extend_word_list.append('<s1>')
                if i == head_pos[1]:
                    extend_word_list.append('</s1>')
            if len(tail_pos) == 2:
                if i == tail_pos[0]:
                    extend_word_list.append('<s2>')
                if i == tail_pos[1]:
                    extend_word_list.append('</s2>')
            if i < len(word_list):
                extend_word_list.append(word_list[i])
                

        extend_word_list = " ".join(extend_word_list)
        
        # org image  
        img_path = os.path.join(self.img_path, imgid)
        org_image = Image.open(img_path).convert('RGB')
        dw, dh = org_image.size

        
        # ent image  
        visual_ents = self.data_dict['ent_dict'][imgid]  # {name:(box) , ...}
        ent_names = list(visual_ents.keys())  # all visual entity names

        """
        visual_ents: {'[OBJ0]': (0.8317, 0.6625, 0.28, 0.66), '[OBJ1]': (0.3842, 0.5587, 0.3683, 0.8675)}
        ent_names: ['[OBJ0]', '[OBJ1]']
        """
        assert len(ent_names) <= self.img_num
        ent_idx_head = -1
        ent_idx_tail = -1
        if len(head_pos) == 4:
            try:
                ent_idx_head = ent_names.index(head_d['name'])
            except ValueError:
                print("ent_names:", ent_names)
                print("visual_ents:", visual_ents)
        if len(tail_pos) == 4:
            try:
                ent_idx_tail = ent_names.index(tail_d['name'])
            except ValueError:
                print("ent_names:", ent_names)
                print("visual_ents:", visual_ents)
            # ent_idx = ent_names.index(tail_d['name'])
        
        
        # Attribute Features Fusion  caption
        if self.args.use_cap:
            cap_head = ""
            cap_tail = ""

            head_name = ' '.join(head_d['name']) if isinstance(head_d['name'], list) else head_d['name']
            tail_name = ' '.join(tail_d['name']) if isinstance(tail_d['name'], list) else tail_d['name']

            if len(head_pos) == 4:
                cap_head = '<o1> ' + self.cap_dict[imgid][head_d['name']] + ' </o1>'
            if len(tail_pos) == 4:
                cap_tail = '<o2> ' + self.cap_dict[imgid][tail_d['name']] + ' </o2>'
            if len(head_pos) == 2:
                cap_head = '<k1> ' + self.cap_dict[imgid][head_name] + ' </k1>'
            if len(tail_pos) == 2:
                cap_tail = '<k2> ' + self.cap_dict[imgid][tail_name] + ' </k2>'

            cap = ""
            if cap_head and cap_tail:
                cap = cap_head + " " + cap_tail
            elif cap_head:
                cap = cap_head
            elif cap_tail:
                cap = cap_tail
            # print("cap:",cap)
            # exit()
            if cap == "":
                encode_dict = self.tokenizer.encode_plus(text=extend_word_list,
                                                    max_length=self.max_seq,
                                                    truncation=True,
                                                    padding='max_length')
            else:
                encode_dict = self.tokenizer.encode_plus(text=extend_word_list,
                                                        text_pair=cap,
                                                        max_length=self.max_seq,
                                                        truncation=True,
                                                        padding='max_length')
        else:
            encode_dict = self.tokenizer.encode_plus(text=extend_word_list,
                                                    max_length=self.max_seq,
                                                    truncation=True,
                                                    padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], encode_dict['attention_mask']
        input_ids, token_type_ids, attention_mask = torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask)

        
        #depth_map
        if self.args.use_dep:
            depth_data = np.load(os.path.join(self.dep_path, imgid.split('.')[0] + '.npz'))
            depth_map = depth_data['depth_map'] 
            depth_map_w, depth_map_h = depth_map.shape[1], depth_map.shape[0]

            

        ent_imgs = []
        position = []
        for name, box in visual_ents.items():
            left, top, right, bottom = box[0], box[1], box[2], box[3]

            ent_img = org_image.crop((left * dw, top * dh, right * dw, bottom * dh))
            ent_img = self.ent_processor(images=ent_img, return_tensors='pt')['pixel_values'].squeeze()
            # print("ent_imgs:",ent_img.shape)
            
            if self.args.use_dep:
                it = (int(depth_map_w * left), int(depth_map_w * right), int(depth_map_h * top), int(depth_map_h * bottom))
                #左到右，上到下，从深度图中截取一个矩形区域
                depth_box = depth_map[it[2]:it[3], it[0]:it[1]]
                depth_box = cv2.resize(depth_box, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)
                ent_img = torch.cat([ent_img, torch.tensor(depth_box).unsqueeze(0)], dim=0)
                # print("depth_box:",depth_box.shape)
            if self.args.use_box:
                # position.append([box[0], box[1], box[2], box[3], box[2] * box[3]])
                position.append([box[0], box[1], box[2], box[3], (right - left) * (bottom - top)])
            #ent_imgs: visual object images
            # print("ent_img:",ent_img.shape)
            ent_imgs.append(ent_img)
        
        # padding  
        for i in range(self.img_num - len(ent_imgs)):
            if self.args.use_dep:
                ent_imgs.append(torch.zeros((4, self.crop_size, self.crop_size)))
            else:
                ent_imgs.append(torch.zeros((3, self.crop_size, self.crop_size)))
            if self.args.use_box:
                position.append([0.0] * 5)

        assert len(ent_imgs) == self.img_num
        assert ent_idx_head < len(ent_imgs)
        assert ent_idx_tail < len(ent_imgs)
        
        re_label = torch.tensor(self.re_dict[relation])  # label to id
        # print("re_label:",re_label)
        # exit()
        
        org_image = self.clip_processor(images=org_image, return_tensors='pt')['pixel_values'].squeeze()
        ret_dict = {
            'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask,
            'org_image': org_image, 'ent_imgs': torch.stack(ent_imgs, dim=0), 'ent_idx_head': torch.tensor(ent_idx_head),
            'ent_idx_tail': torch.tensor(ent_idx_tail), 'labels': re_label
        }

        if self.args.use_box:
            ret_dict.update({'position': torch.tensor(position)})
            
        return ret_dict, re_label
