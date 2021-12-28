import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import time
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from utils.deep_learning import *
from PIL import Image
import cv2

class Solver(object):
    def __init__(self, model, check_point, **kwargs):
        # self.model = nn.DataParallel(model.cuda(), device_ids=[0])
        self.model = model.cuda()
        # self.mode_name = model_name
        # self.my_train_data = my_train_data
        self.check_point = check_point
        self.batch_size = kwargs.pop('batch_size', 64)
        self.lr = kwargs.pop('lr', 1e-4)
        self.epoch_nums = kwargs.pop('epoch_nums', 10)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-8)  # betas=(0.9, 0.999),
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        # self.scheduler = CosineAnnealingRestartLR(self.optimizer, periods=[5e+4, 5e+4, 5e+4, 5e+4, 5e+4, 5e+4],restart_weights=[1, 0.5, 0.5, 0.5, 0.5, 0.5], eta_min=1e-7)
        self.loss_func = nn.MSELoss(reduce=True, size_average=False)  # kwargs.pop('loss_func', nn.MSELoss())
        self.use_gpu = 1
        self.device0 = torch.device("cuda:0")
        self.device1 = torch.device("cuda:0")

    def p_epoch_step(self, dataset, epoch):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, drop_last=False,
                                pin_memory=True)
        num_batchs = len(dataset) // self.batch_size
        running_loss = 0
        batch = 0
        avr_enhance = 0
        for i, sample in enumerate(dataloader):
            # train_I1 = self.my_train_data["compress/I1"]
            # train_P1 = self.my_train_data["compress/P1"]
            # train_P2 = self.my_train_data["compress/P2"]
            # train_P3 = self.my_train_data["compress/P3"]
            # train_I2 = self.my_train_data["compress/I2"]
            # train_P2_label = self.my_train_data["label/P2"]
            # num_frame = train_P2_label.shape[0]
            # print(num_frame)
            # for i in range(0, int(num_frame/10), self.batch_size):
            # print(i)
            # s_time = time.time()

            train_I_batch = sample['train_I']
            train_P1_batch = sample['train_P1']
            train_P2_batch = sample['train_P2']
            train_P3_batch = sample['train_P3']
            train_I2_batch = sample['train_I2']

            # label_I_batch = sample['label_I']
            # label_P1_batch = sample['label_P1']
            label_P2_batch = sample['label_P2']
            # label_P3_batch = sample['label_P3']
            # train_I_batch = torch.from_numpy(train_I1[i:i+self.batch_size,...])/255.0
            # train_P1_batch = torch.from_numpy(train_P1[i:i+self.batch_size,...])/255.0
            # train_P2_batch = torch.from_numpy(train_P2[i:i+self.batch_size,...])/255.0
            # train_P3_batch = torch.from_numpy(train_P3[i:i+self.batch_size,...])/255.0
            # train_I2_batch = torch.from_numpy(train_I2[i:i+self.batch_size,...])/255.0

            # label_P2_batch = torch.from_numpy(train_P2_label[i:i+self.batch_size,...])/255.0

            x = torch.cat([train_I_batch, train_P1_batch, train_P2_batch, train_P3_batch, train_I2_batch], dim=1)
            train_frame = train_P2_batch
            label_frame = label_P2_batch
            # num = random.randint(0,3)
            # if num==0:
            #     train_frame = train_I_batch
            #     label_frame = label_I_batch
            # elif num==1:
            #     train_frame = train_P1_batch
            #     label_frame = label_P1_batch
            # elif num==2:
            #     train_frame = train_P2_batch
            #     label_frame = label_P2_batch
            # else:
            #     train_frame = train_P3_batch
            #     label_frame = label_P3_batch
            if self.use_gpu:
                x = Variable(x.cuda())
                train_frame = Variable(train_frame.cuda())
                label_frame = Variable(label_frame.cuda())
            else:
                pass
            self.optimizer.zero_grad()
            # self.model.share_memory()
            output = self.model(x, train_frame)
            # output = self.model(x)
            # torch.cuda.empty_cache()
            # a = aligned[:,1,...].unsqueeze(1).repeat(1, 3, 1, 1, 1)
            loss = self.loss_func(output, label_frame) \
                # + 0.5 * self.loss_func(sife, label_frame)
            # + 0.3 * self.loss_func(img_out1, label_frame) \
            # + 0.3 * self.loss_func(img_out2.to(self.device0), label_frame) \
            # + 0.3 * self.loss_func(img_out3.to(self.device0), label_frame)

            running_loss += loss.data
            loss.backward()
            self.optimizer.step()
            batch += self.batch_size

            psnr = compare_psnr(label_frame.cpu().data.numpy(), output.cpu().data.numpy(), 1.0)
            psnr_past = compare_psnr(label_frame.cpu().data.numpy(), train_frame.cpu().data.numpy(), 1.0)
            avr_enhance += psnr - psnr_past
            print('Epoch(%2d / %3d), nums: (%5d / %5d), loss: %.6f, psnr: %.3f psnr_past: %.3f, Enhanced: %.6f'
                  % (epoch, self.epoch_nums, batch, dataset.__len__(), loss, psnr, psnr_past, psnr - psnr_past))

            # e_time = time.time()
            # print('Batch_time:%.2fs' % (e_time - s_time))

        avr_enhance = avr_enhance / num_batchs
        average_loss = (running_loss / num_batchs)
        print('Epoch %5d, loss=%.6f, avr_Enhanced=%.6f' % (epoch, average_loss, avr_enhance))
        with open(os.path.join(self.check_point, 'PSNR' + '.txt'), 'a') as f:
            f.write('Epoch %5d, loss=%.6f, avr_Enhanced=%.6f\n' % (epoch, average_loss, avr_enhance))
        writer = SummaryWriter('./log/train')
        writer.add_scalar('PSNR', avr_enhance)
        writer.add_scalar('Loss', average_loss)
        writer.close()

    def train_P(self, train_dataset, val_dataset):
        best_psnr = -1
        with open(os.path.join(self.check_point, 'PSNR' + '.txt'), 'w') as f:
            self.model.train()
            setup_seed(7)
            for epoch in range(self.epoch_nums):
                self.p_epoch_step(train_dataset, epoch)
                self.scheduler.step()
                torch.save(self.model, os.path.join(self.check_point, 'epoch-' + str(epoch) + '-model.pt'))

                # self.model.eval()
                # if epoch % 1 == 0:
                #      train_psnr, train_psnr_past = self.check_PSNR_P(val_dataset)
                #
                #      f.write('epoch%d:\t avr_psnr:%.3f\t avr_psnr_past:%.3f\t Enhanced:%.3f\n' % (epoch, train_psnr, train_psnr_past,train_psnr-train_psnr_past))
                #      print('epoch%d:\t avr_psnr:%.3f\t avr_psnr_past:%.3f\t Enhanced:%.3f\n' % (epoch, train_psnr, train_psnr_past,train_psnr-train_psnr_past))
                #      if best_psnr < train_psnr:
                #          best_psnr = train_psnr
                #          if not os.path.exists(self.check_point):
                #              os.mkdir(self.check_point)
                #          model_path = os.path.join(self.check_point, 'val' + '-model.pt')
                #          torch.save(self.model, model_path)
                #          print('Best average psnr: %.3f' % best_psnr)
                #          print('')
                # self.model.train()

    def test_P(self, dataset):
        self.model.eval()
        psnr = 0
        psnr_past = 0
        ssim = 0
        ssim_past = 0
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, drop_last=False)
        num = dataloader.__len__() * 4
        for i, sample in tqdm(enumerate(dataloader)):
            train_I_batch = sample['train_I']
            train_P1_batch = sample['train_P1']
            train_P2_batch = sample['train_P2']
            train_P3_batch = sample['train_P3']
            train_I2_batch = sample['train_I2']

            label_I_batch = sample['label_I']
            label_P1_batch = sample['label_P1']
            label_P2_batch = sample['label_P2']
            label_P3_batch = sample['label_P3']

            if self.use_gpu:
                train_I = Variable(train_I_batch.cuda())
                train_P1 = Variable(train_P1_batch.cuda())
                train_P2 = Variable(train_P2_batch.cuda())
                train_P3 = Variable(train_P3_batch.cuda())
                train_I2 = Variable(train_I2_batch.cuda())
            else:
                train_I = Variable(train_I_batch)
                train_P1 = Variable(train_P1_batch)
                train_P2 = Variable(train_P2_batch)
                train_P3 = Variable(train_P3_batch)
                train_I2 = Variable(train_I2_batch)
            label_I = label_I_batch
            label_P1 = label_P1_batch
            label_P2 = label_P2_batch
            label_P3 = label_P3_batch
            x = torch.cat([train_I, train_P1, train_P2, train_P3, train_I2], dim=1)
            for j in range(4):
                if j == 0:
                    # x = torch.cat([train_P1, train_P3, train_I, train_P2, train_I2], dim=1)
                    PFrm = train_I
                    train = np.squeeze(train_I.cpu().permute(0, 2, 3, 1).numpy())
                    label = np.squeeze(label_I.cpu().permute(0, 2, 3, 1).numpy())
                elif j == 1:
                    # x = torch.cat([train_I, train_P3,train_P1, train_P2, train_I2], dim=1)
                    PFrm = train_P1
                    train = np.squeeze(train_P1.cpu().permute(0, 2, 3, 1).numpy())
                    label = np.squeeze(label_P1.cpu().permute(0, 2, 3, 1).numpy())
                elif j == 2:
                    # x = torch.cat([train_I, train_P1, train_P2, train_P3, train_I2], dim=1)
                    PFrm = train_P2
                    train = np.squeeze(train_P2.cpu().permute(0, 2, 3, 1).numpy())
                    label = np.squeeze(label_P2.cpu().permute(0, 2, 3, 1).numpy())
                else:
                    # x = torch.cat([train_I, train_P1, train_P3, train_P2, train_I2], dim=1)
                    PFrm = train_P3
                    train = np.squeeze(train_P3.cpu().permute(0, 2, 3, 1).numpy())
                    label = np.squeeze(label_P3.cpu().permute(0, 2, 3, 1).numpy())
                with torch.no_grad():
                    output = self.model(x)
                output = np.squeeze(output.cpu().permute(0, 2, 3, 1).detach().numpy())
                # psnr
                psnr += compare_psnr(label * 255, output * 255, 255.0)
                psnr_past += compare_psnr(label * 255, train * 255, 255.0)
                a = compare_psnr(label * 255, output * 255, 255.0)
                b = compare_psnr(label * 255, train * 255, 255.0)
                print('psnr:%.3f, psnr_compressed:%.3f, improve:%.3f' % (a, b, a - b))
                # ssim
                # print(output.shape)
                # for k in range(output.shape[0]):
                ssim += compare_ssim(output, label, data_range=1.0) * 10000
                ssim_past += compare_ssim(train, label, data_range=1.0) * 10000

        psnr = psnr / num
        psnr_past = psnr_past / num
        ssim /= num
        ssim_past /= num
        print('nums of Frame: %d' % num)
        print('pnsr: %.3f, compressed_psnr: %.3f, improve: %.3f' % (psnr, psnr_past, psnr - psnr_past))
        print('ssim: %.3f, compressed_ssim: %.3f, improve: %.3f' % (ssim, ssim_past, ssim - ssim_past))

    def check_PSNR_P(self, dataset):
        avr_psnr = 0
        avr_psnr_past = 0
        avr_ssim = 0
        count = 0

        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4, drop_last=False)
        for i, sample in enumerate(dataloader):
            train_I_batch = sample['train_I']
            train_P1_batch = sample['train_P1']
            train_P2_batch = sample['train_P2']
            train_P3_batch = sample['train_P3']
            train_I2_batch = sample['train_I2']

            label_P2_batch = sample['label_P2']

            train_I_batch = Variable(train_I_batch.cuda())
            train_P1_batch = Variable(train_P1_batch.cuda())
            train_P2_batch = Variable(train_P2_batch.cuda())
            train_P3_batch = Variable(train_P3_batch.cuda())
            train_I2_batch = Variable(train_I2_batch.cuda())
            label_P2_batch = Variable(label_P2_batch.cuda())
            x = torch.cat([train_I_batch, train_P1_batch, train_P2_batch, train_P3_batch, train_I2_batch], dim=1)
            with torch.no_grad():
                output = self.model(x, train_P2_batch)
            psnr = compare_psnr(output.cpu().data.numpy(), label_P2_batch.cpu().data.numpy(), 1.0)
            psnr_past = compare_psnr(train_P2_batch.cpu().data.numpy(), label_P2_batch.cpu().data.numpy(), 1.0)

            avr_psnr_past += psnr_past
            count += 1
            avr_psnr += psnr
        # print('count:', count)
        avr_psnr /= count
        avr_psnr_past /= count
        return avr_psnr, avr_psnr_past

    def test_r3(self, train_dir, label_dir, h, w, tot_frm):
        video_train_Y = import_yuv(train_dir, h=h, w=w, tot_frm=tot_frm).astype(np.float32) / 255.
        video_label_Y = import_yuv(label_dir, h=h, w=w, tot_frm=tot_frm).astype(np.float32) / 255.
        self.model.eval()
        psnr = 0
        psnr_past = 0
        for idx in range(tot_frm):
            # load lq
            idx_list = list(range(idx - 3, idx + 4))
            idx_list = np.clip(idx_list, 0, tot_frm - 1)
            input_data = []
            for idx_ in idx_list:
                input_data.append(video_train_Y[idx_])
            input_data = torch.from_numpy(np.array(input_data))
            input_data = torch.unsqueeze(input_data, 0).cuda()

            train = torch.unsqueeze(input_data[:, 3, ...], 0).cuda()
            label = video_label_Y[idx]
            with torch.no_grad():
                output = self.model(input_data)
            output = np.squeeze(output.cpu().detach().numpy())
            train = np.squeeze(train.cpu().detach().numpy())
            # psnr
            psnr += compare_psnr(label * 255, output * 255, 255.0)
            psnr_past += compare_psnr(label * 255, train * 255, 255.0)
            a = compare_psnr(label * 255, output * 255, 255.0)
            b = compare_psnr(label * 255, train * 255, 255.0)
            print('(%3d / %3d):psnr:%.3f, psnr_compressed:%.3f, improve:%.3f' % (idx, tot_frm, a, b, a - b))
        psnr = psnr / tot_frm
        psnr_past = psnr_past / tot_frm
        # ssim /= num
        # ssim_past /= num
        print('Summary:-----------------')
        print('pnsr: %.3f, compressed_psnr: %.3f, improve: %.3f' % (psnr, psnr_past, psnr - psnr_past))
    def test_r1(self, train_dir, label_dir, h, w, tot_frm):
        video_train_Y, video_train_U, video_train_V = import_yuv(train_dir, h=h, w=w, tot_frm=tot_frm)#.astype(np.float32) / 255.
        video_label_Y, video_label_U, video_label_V = import_yuv(label_dir, h=h, w=w, tot_frm=tot_frm)#.astype(np.float32) / 255.    
        video_train_Y = video_train_Y.astype(np.float32) / 255.
        video_label_Y = video_label_Y.astype(np.float32) / 255.
        self.model.eval()
        ssim = 0
        ssim_past = 0
        psnr = 0
        psnr_past = 0
        for idx in range(tot_frm):
            # load lq
            idx_list = list(range(idx - 1, idx + 2))
            idx_list = np.clip(idx_list, 0, tot_frm - 1)
            input_data = []
            for idx_ in idx_list:
                input_data.append(video_train_Y[idx_])
            input_data = torch.from_numpy(np.array(input_data))
            input_data = torch.unsqueeze(input_data, 0).cuda()

            train = torch.unsqueeze(input_data[:, 1, ...], 0).cuda()
            label = video_label_Y[idx]
            with torch.no_grad():
                output = self.model(input_data)
            output = np.squeeze(output.cpu().detach().numpy())
            train = np.squeeze(train.cpu().detach().numpy())
            # psnr
            psnr += peak_signal_noise_ratio(label * 255, output * 255, data_range=255)
            psnr_past += peak_signal_noise_ratio(label * 255, train * 255, data_range=255)
            a = peak_signal_noise_ratio(label * 255, output * 255, data_range=255)
            b = peak_signal_noise_ratio(label * 255, train * 255, data_range=255)
            ssim += structural_similarity(label * 255, output * 255, data_range=255)
            ssim_past += structural_similarity(label * 255, train * 255, data_range=255)
            print('(%3d / %3d):psnr:%.3f, psnr_compressed:%.3f, improve:%.3f' % (idx, tot_frm, a, b, a - b))
            # save image
            #yuv2rgb(output * 255, video_train_U[idx, ...], video_train_V[idx, ...], idx, 'my')
            #yuv2rgb(train * 255, video_train_U[idx, ...], video_train_V[idx, ...], idx, 'compress')
            #yuv2rgb(label * 255, video_label_U[idx, ...], video_label_V[idx, ...], idx, 'label')
                        
        psnr = psnr / tot_frm
        psnr_past = psnr_past / tot_frm
        ssim /= tot_frm
        ssim_past /= tot_frm
        print('Summary:-----------------')
        print('pnsr: %.3f, compressed_psnr: %.3f, improve: %.3f' % (psnr, psnr_past, psnr - psnr_past))
        print('ssim: %.3f, past_psnr: %.3f, improve: %.3f' % (ssim, ssim_past, (ssim - ssim_past)*100))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def yuv2rgb(Y, U, V, i, mark):
    h = 1080
    w = 1920
    U = np.array(Image.fromarray(U).resize((w, h)))
    V = np.array(Image.fromarray(V).resize((w, h)))
    rf = Y + 1.4075 * (V - 128.0)
    gf = Y - 0.3455 * (U - 128.0) - 0.7169 * (V - 128.0)
    bf = Y + 1.7790 * (U - 128.0)

    rf[rf > 255] = 255
    rf[rf < 0] = 0
    gf[gf > 255] = 255
    gf[gf < 0] = 0
    bf[bf > 255] = 255
    bf[bf < 0] = 0

    r = rf.astype(np.uint8)
    g = gf.astype(np.uint8)
    b = bf.astype(np.uint8)
    img = cv2.merge([b, g, r])
    cv2.imwrite('./TestImg/' + mark + '/' + str(i) + '.jpg', img)
    return r, g, b
def import_yuv(seq_path, h, w, tot_frm=300, yuv_type='420p', start_frm=0, only_y=0, date_type=np.uint8):
    """Load Y, U, and V channels separately from a 8bit yuv420p video.

    Args:
        seq_path (str): .yuv (imgs) path.
        h (int): Height.
        w (int): Width.
        tot_frm (int): Total frames to be imported.
        yuv_type: 420p or 444p
        start_frm (int): The first frame to be imported. Default 0.
        only_y (bool): Only import Y channels.

    Return:
        y_seq, u_seq, v_seq (3 channels in 3 ndarrays): Y channels, U channels,
        V channels.

    Note:
        YUV传统上是模拟信号格式, 而YCbCr才是数字信号格式.YUV格式通常实指YCbCr文件.
        参见: https://en.wikipedia.org/wiki/YUV
    """

    # setup params
    if yuv_type == '420p':
        hh, ww = h // 2, w // 2
    elif yuv_type == '444p':
        hh, ww = h, w
    else:
        raise Exception('yuv_type not supported.')

    y_size, u_size, v_size = h * w, hh * ww, hh * ww
    # if(date_type == np.uint16):
    #    y_size=y_size*2
    #    u_size=u_size*2
    #    v_size=v_size*2
    blk_size = y_size + u_size + v_size
    if (date_type == np.uint16):
        blk_size = blk_size * 2
    # init
    y_seq = np.zeros((tot_frm, h, w), dtype=date_type)
    if not only_y:
        u_seq = np.zeros((tot_frm, hh, ww), dtype=date_type)
        v_seq = np.zeros((tot_frm, hh, ww), dtype=date_type)

    # read data
    with open(seq_path, 'rb') as fp:
        fp.seek(0, 2)
        fp.seek(0, 2)
        for i in range(tot_frm):
            fp.seek(int(blk_size * (start_frm + i)), 0)  # skip frames
            y_frm = np.fromfile(fp, dtype=date_type, count=y_size).reshape(h, w)
            if only_y:
                y_seq[i, ...] = y_frm
            else:
                u_frm = np.fromfile(fp, dtype=date_type, \
                                    count=u_size).reshape(hh, ww)
                v_frm = np.fromfile(fp, dtype=date_type, \
                                    count=v_size).reshape(hh, ww)
                y_seq[i, ...], u_seq[i, ...], v_seq[i, ...] = y_frm, u_frm, v_frm
    if only_y:
        return y_seq
    else:
        return y_seq, u_seq, v_seq











