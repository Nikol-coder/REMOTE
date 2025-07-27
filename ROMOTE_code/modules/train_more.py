import torch
from torch import optim
from tqdm import tqdm
from sklearn.metrics import classification_report
from transformers.optimization import get_linear_schedule_with_warmup
from .metrics import eval_result
import gc
import resource

class BertTrainer(object):
    def __init__(self, train_data=None, dev_data=None, test_data=None, re_dict=None, model=None, process=None,
                 args=None, logger=None, writer=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.re_dict = re_dict
        self.model = model
        self.num_gpus = torch.cuda.device_count()
        self.process = process
        self.logger = logger
        self.writer = writer
        self.refresh_step = 2

        self.best_dev_epoch = None
        self.best_test_epoch = None

        self.best_test_metric = 0
        self.test_acc = 0
        self.best_dev_metric = 0
        self.dev_acc = 0

        self.optimizer = None

        self.step = 0
        self.args = args

        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
            self.before_multimodal_train()

    def train(self):
        self.step = 0
        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data) * self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")

        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True,
                  initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0
            for epoch in range(1, self.args.num_epochs + 1):
                # self.test(epoch)
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                for batch in self.train_data:
                    self.step += 1
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    params , labels = batch

                    (loss, logits), labels = self._step(params, labels)                 
                    avg_loss += loss.detach().cpu().item()

                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        print_output = "loss:{:<6.5f}".format(avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)
                        if self.writer is not None:
                            self.writer.add_scalar(tag='train_loss', scalar_value=avg_loss, global_step=self.step)
                        avg_loss = 0
                        # clear_memory()
                        
                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)  # generator to dev.
                    self.test(epoch)

            pbar.close()
            self.pbar = None
            self.logger.info(
                "Get best dev performance at epoch {}, best dev f1 score is {}, acc = {}".format(self.best_dev_epoch,
                                                                                                 self.best_dev_metric,
                                                                                                 self.dev_acc))
            self.logger.info(
                "Get best test performance at epoch {}, best test f1 score is {}, acc = {}".format(self.best_test_epoch,
                                                                                                   self.best_test_metric,
                                                                                                   self.test_acc))

    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        step = 0
        true_labels, pred_labels = [], []
        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Evaluating")
                total_loss = 0
                for batch in self.dev_data:
                    step += 1
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    params , labels = batch
                    # (loss, logits), labels = self._step(batch)  
                    (loss, logits), labels = self._step(params, labels)
                    # (loss, logits), labels  = self._step(batch)
                    total_loss += loss.detach().cpu().item()

                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())

                    pbar.update()
                # evaluate done
                pbar.close()
                sk_result = classification_report(y_true=true_labels, y_pred=pred_labels,
                                                  labels=list(self.re_dict.values())[1:],
                                                  target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                acc, micro_f1 = round(result['acc'] * 100, 4), round(result['micro_f1'] * 100, 4)
                if self.writer is not None:
                    self.writer.add_scalar(tag='dev_acc', scalar_value=acc, global_step=epoch)
                    self.writer.add_scalar(tag='dev_f1', scalar_value=micro_f1, global_step=epoch)
                    self.writer.add_scalar(tag='dev_loss', scalar_value=total_loss / len(self.dev_data),
                                           global_step=epoch)

                self.logger.info("Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}, acc: {}." \
                                 .format(epoch, self.args.num_epochs, self.best_dev_metric, self.best_dev_epoch,
                                         micro_f1, acc))
                if micro_f1 >= self.best_dev_metric:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metric = micro_f1  # update best metric(f1 score)
                    self.dev_acc = acc


        self.model.train()

    def test(self, epoch):
        self.model.eval()
        self.logger.info("\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        true_labels, pred_labels = [], []
        input_ids_list = []
        position_list = []
        ent_idx_list = []
        # position_idx_list = []
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    params , labels = batch
                    input_ids = params['input_ids']
                    position = params['position']
                    ent_idx = params['ent_idx']
                    (loss, logits), labels = self._step(params, labels)
                    # (loss, logits), labels = self._step(batch)  # logits: batch, 3
                    total_loss += loss.detach().cpu().item()

                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())
                    input_ids_list.extend(input_ids.detach().cpu().tolist())
                    position_list.extend(position.detach().cpu().tolist())
                    ent_idx_list.extend(ent_idx.detach().cpu().tolist())
                    # position_idx_list.extend(position_idx.detach().cpu().tolist())
                    pbar.update()
                # evaluate done
                pbar.close()
                
                sk_result = classification_report(y_true=true_labels, y_pred=pred_labels,
                                                  labels=list(self.re_dict.values())[1:],
                                                  target_names=list(self.re_dict.keys())[1:],
                                                  digits=8)
                self.logger.info("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                acc, micro_f1 = round(result['acc'] * 100, 4), round(result['micro_f1'] * 100, 4)
                if self.writer is not None:
                    self.writer.add_scalar(tag='test_acc', scalar_value=acc, global_step=epoch)
                    self.writer.add_scalar(tag='test_f1', scalar_value=micro_f1, global_step=epoch)
                    self.writer.add_scalar(tag='test_loss', scalar_value=total_loss / len(self.test_data),
                                           global_step=epoch)
                self.logger.info("Epoch {}/{}, best test f1: {}, best epoch: {}, current test f1 score: {}, acc: {}" \
                                 .format(epoch, self.args.num_epochs, self.best_test_metric, self.best_test_epoch,
                                         micro_f1, acc))
                if micro_f1 >= self.best_test_metric:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_test_epoch = epoch
                    self.best_test_metric = micro_f1  # update best metric(f1 score)
                    self.test_acc = acc
                    if self.args.save_path is not None:  # save model
                        torch.save(self.model.state_dict(), self.args.save_path + "/best_model.pth")
                        self.logger.info("Save best model at {}".format(self.args.save_path))
                    
                    fout = open("./test.txt", 'w')
                    for i in range(len(pred_labels)):
                        samp_input_ids = input_ids_list[i]
                        samp_pred_label = pred_labels[i]
                        samp_true_label = true_labels[i]
                        samp_position = position_list[i]
                        samp_ent_idx = ent_idx_list[i]
                        
                        
                        if isinstance(samp_ent_idx, int):
                            samp_position_values = [samp_position[samp_ent_idx]]
                        else:
                            samp_position_values = samp_position[samp_ent_idx]
                        

                        formatted_position = []
                        for pos_list in samp_position_values:
                            if isinstance(pos_list, list):
                                formatted_position.extend(['{:.4f}'.format(pos) for pos in pos_list])
                            else:
                                print(f"Unsupported type: {type(pos_list)}")
                                formatted_position.append(str(pos_list))
                        
                        fout.write(f"input_ids: " + ' '.join(map(str, samp_input_ids)) + '\n')
                        fout.write(f"position: " + ' '.join(formatted_position) + '\n')
                        fout.write(f"pred_label: " + ' '.join(map(str, [samp_pred_label])) + '\n')  
                        fout.write(f"true_label: " + ' '.join(map(str, [samp_true_label])) + '\n' + '\n')  
                    fout.close()
                    print("save at test.txt")
                    
        self.model.train()

    def best(self, epoch):
        
        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")
        
        self.model.eval()
        self.logger.info("\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        true_labels, pred_labels = [], []
        input_ids_list = []
        position_list = []
        ent_idx_list = []
        logits_list = []
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    params , labels = batch
                    input_ids = params['input_ids']
                    position = params['position']
                    ent_idx = params['ent_idx']
                    
                    (loss, logits), labels = self._step(params, labels)

                    total_loss += loss.detach().cpu().item()

                    sorted_logits_indices = torch.argsort(logits, dim=-1, descending=True)
    
                    logits_list.extend(sorted_logits_indices.detach().cpu().tolist())
                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())
                    input_ids_list.extend(input_ids.detach().cpu().tolist())
                    position_list.extend(position.detach().cpu().tolist())
                    ent_idx_list.extend(ent_idx.detach().cpu().tolist())
                    # position_idx_list.extend(position_idx.detach().cpu().tolist())
                    pbar.update()
                # evaluate done
                pbar.close()
                
                sk_result = classification_report(y_true=true_labels, y_pred=pred_labels,
                                                  labels=list(self.re_dict.values())[1:],
                                                  target_names=list(self.re_dict.keys())[1:],
                                                  digits=8)
                self.logger.info("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                acc, micro_f1 = round(result['acc'] * 100, 4), round(result['micro_f1'] * 100, 4)
                if self.writer is not None:
                    self.writer.add_scalar(tag='test_acc', scalar_value=acc, global_step=epoch)
                    self.writer.add_scalar(tag='test_f1', scalar_value=micro_f1, global_step=epoch)
                    self.writer.add_scalar(tag='test_loss', scalar_value=total_loss / len(self.test_data),
                                           global_step=epoch)
                self.logger.info("Epoch {}/{}, best test f1: {}, best epoch: {}, current test f1 score: {}, acc: {}" \
                                 .format(epoch, self.args.num_epochs, self.best_test_metric, self.best_test_epoch,
                                         micro_f1, acc))
                if micro_f1 >= self.best_test_metric:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_test_epoch = epoch
                    self.best_test_metric = micro_f1  # update best metric(f1 score)
                    self.test_acc = acc
                    if self.args.save_path is not None:  # save model
                        torch.save(self.model.state_dict(), self.args.save_path + "/best_model.pth")
                        self.logger.info("Save best model at {}".format(self.args.save_path))
                    
                    fout = open("./test_best_encoder.txt", 'w')
                    for i in range(len(pred_labels)):
                        samp_input_ids = input_ids_list[i]
                        samp_pred_label = pred_labels[i]
                        samp_true_label = true_labels[i]
                        samp_position = position_list[i]
                        samp_ent_idx = ent_idx_list[i]
                        samp_logits = logits_list[i]
                        
                        if isinstance(samp_ent_idx, int):
                            samp_position_values = [samp_position[samp_ent_idx]]
                        else:
                            samp_position_values = samp_position[samp_ent_idx]
                        
                        formatted_position = []
                        for pos_list in samp_position_values:
                            if isinstance(pos_list, list):
                                formatted_position.extend(['{:.4f}'.format(pos) for pos in pos_list])
                            else:
                                print(f"Unsupported type: {type(pos_list)}")
                                formatted_position.append(str(pos_list))
                        
                        fout.write(f"input_ids: " + ' '.join(map(str, samp_input_ids)) + '\n')
                        fout.write(f"position: " + ' '.join(formatted_position) + '\n')
                        fout.write(f"pred_label: " + ' '.join(map(str, [samp_pred_label])) + '\n')  
                        fout.write(f"true_label: " + ' '.join(map(str, [samp_true_label])) + '\n')  
                        fout.write(f"logits: " + ' '.join(map(str, [samp_logits])) + '\n' + '\n')  
                    fout.close()
                    print("save at test_best_encoder.txt")

    def _step(self, params, labels):
        outputs = self.model(**params)
        return outputs, labels

    def before_multimodal_train(self):
        optimizer_grouped_parameters = []
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'model' in name:
                params['params'].append(param)
        optimizer_grouped_parameters.append(params)

        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)
