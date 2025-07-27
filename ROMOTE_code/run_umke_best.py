import os
import argparse
import logging
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from models_best.unimo_model import UnimoREModel
from processor.dataset_umke_best import MOREProcessor, MOREDataset
from modules.train import BertTrainer
import warnings
from datetime import datetime
import transformers
import gc
import resource

transformers.logging.set_verbosity_error()

# from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    )
logger = logging.getLogger(__name__)

DATA_PATH = {
    'UMKE': {
        'train': '/data/umke_partner/train_set.json',
        'valid': '/data/umke_partner/val_set.json',
        'test': '/data/umke_partner/test_set.json',
        'train_ent_dict': '/data/umke_partner/pos_umke.json',
        'valid_ent_dict': '/data/umke_partner/pos_umke.json',
        'test_ent_dict': '/data/umke_partner/pos_umke.json',
    }
}


IMG_PATH = {
    'UMKE': {
        'train': '/data/UMKE_IMG',
        'valid': '/data/UMKE_IMG',
        'test': '/data/UMKE_IMG',
    }
}


DEP_PATH = {
    'UMKE': {
        'train': '/data/depth_data_umke',
        'valid': '/data/depth_data_umke',
        'test': '/data/depth_data_umke',
    }
}


#qwen2.5 vl
CAP_PATH = {
    'UMKE': {
        'train': '/data/umke_partner/cap_qwenvl_simply.json',
        'valid': '/data/umke_partner/cap_qwenvl_simply.json',
        'test': '/data/umke_partner/cap_qwenvl_simply.json',
    }
}


#relation ID key-value
re_path = '/data/lxk2019/UMREF/MORE_test/rel2id_umke_partner.json'

#seed
def set_seed(seed=2024):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vit_name', default='vit', type=str, help="The name of vit.")
    parser.add_argument('--dataset_name', default='UMKE', type=str, help="The name of dataset.")
    parser.add_argument('--bert_name', default='bert-base', type=str,
                        help="Pretrained language model name, bart-base or bart-large")
    parser.add_argument('--num_epochs', default=30, type=int, help="Training epochs")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--batch_size', default=16, type=int, help="batch size")
    parser.add_argument('--lr', default=2e-5, type=float, help="learning rate")
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--eval_begin_epoch', default=16, type=int)
    parser.add_argument('--seed', default=1, type=int, help="random seed, default is 1")
    parser.add_argument('--load_path', default=None, type=str, help="Load model from load_path")
    parser.add_argument('--save_path', default=None, type=str, help="save model at save_path")
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--max_seq', default=128, type=int)
    parser.add_argument('--use_box', action='store_true')
    parser.add_argument('--use_cap', action='store_true')
    parser.add_argument('--use_dep', action='store_true')

    args = parser.parse_args()
    set_seed(args.seed)  # set seed, default is 1

    
    data_path, img_path, dep_path, cap_path = DATA_PATH[args.dataset_name], IMG_PATH[args.dataset_name], DEP_PATH[args.dataset_name], CAP_PATH[args.dataset_name]
    # print(data_path, img_path, dep_path, cap_path)
    # exit()
    data_process, dataset_class = MOREProcessor, MOREDataset
    logger.info(data_path)

    if args.save_path is not None:  # make save_path dir
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # save path
        args.save_path = os.path.join(args.save_path, args.dataset_name + "_" + str(args.batch_size) + "_" + str(args.lr) + "_" + args.notes + "_" + current_time)
        if args.do_train:
            # args.save_path = os.path.join(args.save_path, args.dataset_name + "_" + str(args.batch_size) + "_" + str(args.lr) + "_" + args.notes)
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path, exist_ok=True)
    logger.info(args)

    # logdir = "logs/" + args.dataset_name + "_" + str(args.batch_size) + "_" + str(args.lr) + args.notes
    # writer = SummaryWriter(logdir=logdir),
    writer = None
    
    # test
    if args.do_test:
        processor = data_process(data_path, re_path, args.bert_name, args.vit_name)

        train_dataset = dataset_class(processor, img_path, dep_path, cap_path, args, mode='train')
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        valid_dataset = dataset_class(processor, img_path, dep_path, cap_path, args, mode='valid')
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        test_dataset = dataset_class(processor, img_path, dep_path, cap_path, args, mode='test')
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        re_dict = processor.get_relation_dict()
        num_labels = len(re_dict)
        tokenizer = processor.tokenizer

        # train 
        model = UnimoREModel(num_labels, tokenizer, args)
        model = torch.nn.DataParallel(model)
        model = model.to(args.device)
        trainer = BertTrainer(train_data=train_dataloader, dev_data=valid_dataloader, test_data=test_dataloader,
                              re_dict=re_dict, model=model, args=args, logger=logger, writer=writer)
        
        trainer.best(0)
        torch.cuda.empty_cache()    
        # writer.close()
        exit()    
    
    #train 
    if args.do_train:
        
        processor = data_process(data_path, re_path, args.bert_name, args.vit_name)

        train_dataset = dataset_class(processor, img_path, dep_path, cap_path, args, mode='train')
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        valid_dataset = dataset_class(processor, img_path, dep_path, cap_path, args, mode='valid')
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        test_dataset = dataset_class(processor, img_path, dep_path, cap_path, args, mode='test')
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        
        re_dict = processor.get_relation_dict()
        # print("re_dict:",re_dict)
        # exit()
        num_labels = len(re_dict)
        tokenizer = processor.tokenizer

        # train 
        model = UnimoREModel(num_labels, tokenizer, args)
        model = torch.nn.DataParallel(model)
        model = model.to(args.device)
        trainer = BertTrainer(train_data=train_dataloader, dev_data=valid_dataloader, test_data=test_dataloader,
                              re_dict=re_dict, model=model, args=args, logger=logger, writer=writer)
        trainer.train()
        torch.cuda.empty_cache()
        # writer.close()


if __name__ == "__main__":
    main()
