import time
import os
import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.FLNM_Net import FLNM_Net
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm

class FLNM_NetTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        # self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.dropout_rate = config.MODEL.DROP_RATE
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.FS = config.TRAIN.DATA.FS
        self.gamma = config.MODEL.FLNM_Net.gamma
        self.use_att = config.MODEL.FLNM_Net.use_att
        self.use_FAG = config.MODEL.FLNM_Net.use_FAG

        self.time_vec = []    # FPS + TIME
        self.time_per_vec = []# FPS + TIME
        self.frame_count = 0  # FPS + TIME
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH  # FPS + TIME

        if torch.cuda.is_available() and config.NUM_OF_GPU_TRAIN > 0:
            dev_list = [int(d) for d in config.DEVICE.replace("cuda:", "").split(",")]
            self.device = torch.device(dev_list[0])
            self.num_of_gpu = config.NUM_OF_GPU_TRAIN
            print(f"Using {self.num_of_gpu} GPUs: {dev_list[:self.num_of_gpu]}")
        else:
            self.device = torch.device("cpu")
            self.num_of_gpu = 0

        in_channels = self.config.MODEL.FLNM_Net.CHANNELS

        self.model = FLNM_Net(in_channel=in_channels, device=self.device,
                gamma=self.gamma, use_att=self.use_att, use_FAG=self.use_FAG)

        self.model = self.model.to(self.device)
        if torch.cuda.device_count() > 0 and self.num_of_gpu > 0:
            self.model = torch.nn.DataParallel(self.model, device_ids=dev_list[:self.num_of_gpu])
        else:
            self.model = torch.nn.DataParallel(self.model)

        if self.config.TOOLBOX_MODE == "train_and_test" or self.config.TOOLBOX_MODE == "only_train":
            self.num_train_batches = len(data_loader["train"])
            self.criterion = Neg_Pearson()
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.config.TRAIN.LR)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.config.TRAIN.LR, epochs=self.config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif self.config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("FactorizePhys trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()

            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                # DATA
                data, BVP_label = batch[0].to(self.device), batch[1].to(self.device)

                # network
                rPPG = self.model(data)
                # normalize
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / torch.std(BVP_label)          # normalize

                self.optimizer.zero_grad()

                loss = self.criterion(rPPG, BVP_label)
                loss.backward()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())

                self.optimizer.step()
                self.scheduler.step()

                tbar.set_postfix(loss=loss.item())

            # Append the mean training loss for the epoch
            mean_training_losses.append(np.mean(train_loss))
            print("Mean train loss: {}".format(np.mean(train_loss)))

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH:
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validing===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                # DATA
                data, BVP_label = valid_batch[0].to(self.device), valid_batch[1].to(self.device)
                # network
                rPPG = self.model(data)
                # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / torch.std(BVP_label)          # normalize
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize

                loss = self.criterion(rPPG, BVP_label)
                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH, map_location=self.device), strict=False)
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path, map_location=self.device), strict=False)
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path, map_location=self.device), strict=False)

        self.model = self.model.to(self.device)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                # DATA
                batch_size = test_batch[0].shape[0]
                data, BVP_label = test_batch[0].to(self.device), test_batch[1].to(self.device)

                # network && FPS + TIME
                t0 = time.time()
                rPPG = self.model(data)
                # b is batch number, c channels, t frame, fh frame height, and fw frame width
                t1 = time.time()
                batch_time = t1 - t0
                self.time_vec.append(batch_time)         # 推理一波的时间
                self.time_per_vec.append(batch_time / (data.shape[0] * self.chunk_len))     # 推理一张的时间
                self.frame_count += data.shape[0] * self.chunk_len  # 推理一波的总帧数
                # FPS + TIME

                # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / torch.std(BVP_label)          # normalize
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    BVP_label = BVP_label.cpu()
                    rPPG = rPPG.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = rPPG[idx]
                    labels[subj_index][sort_index] = BVP_label[idx]

        # 计算 FLOPs 和参数量
        from thop import profile
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            flops, params = profile(self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model,
                                    inputs=(data, ))
        model_size_MB = (params * 4) / (1024 ** 2)

        # FPS + TIME
        Total_GPU_time = np.median(self.time_vec) * 1000
        Perframe_GPU_time = np.median(self.time_per_vec) * 1000
        fps = self.frame_count / np.sum(self.time_vec)
        print(f"Inference Time on GPU: {Total_GPU_time:.4f} ms")
        print(f"Inference Time of a Picture on GPU: {Perframe_GPU_time:.4f} ms")
        print(f"FPS: {fps:.2f}")
        # 储存
        data_metrics = {
            "FLOPs(G)": flops / 1e9,
            "Model Parameters(M)": params / 1e6,
            "Model Size(MB)": model_size_MB,
            "Inference Time on GPU": Total_GPU_time,
            "Inference Time of a Picture on GPU": Perframe_GPU_time,
            "FPS": fps
        }
        # FPS + TIME
        calculate_metrics(predictions, labels, self.config, data_metrics)
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        print('Saved Model Path: ', model_path)
        torch.save(self.model.state_dict(), model_path)
