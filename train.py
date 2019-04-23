import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from tqdm import tqdm
import numpy as np
import cv2
import json
import argparse
import os
from os.path import exists, join

from model.pggan import Generator, Discriminator
from model.loss import GANLoss, GradientPenaltyLoss
from dataloader.dataset import TrainDataset, TestDataset
from utils.util import save_result


def train(config):
    print(config)

    if config.device_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = config.device_id

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    if not exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)

    if not exists(config.result_path):
        os.makedirs(config.result_path)

    with open(join(config.result_path, 'config.txt'), 'w') as f:
        f.write(str(config))

    writer = SummaryWriter(config.result_path)
    
    if config.output_act == 'linear':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif config.output_act == 'tanh':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    train_dataset = TrainDataset(config.celeba_hq_dir, config.train_file, resolution=config.resolution,
                                 transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    generator = Generator(resolution=config.resolution, output_act=config.output_act, norm=config.norm, device=device).to(device)
    discriminator = Discriminator(resolution=config.resolution, device=device).to(device)

    ganLoss = GANLoss(gan_mode=config.gan_type, target_real_label=0.9, target_fake_label=0.).to(device)

    if config.gan_type == 'wgangp':
        gpLoss = GradientPenaltyLoss(device).to(device)

    if config.load_G:
        if torch.cuda.is_available():
            generator.load_model(config.load_G)
        else:
            generator.load_model(config.load_G, map_location='cpu')
        print('Loading %s' % config.load_G)

    if config.load_D:
        if torch.cuda.is_available():
            discriminator.load_model(config.load_D)
        else:
            discriminator.load_model(config.load_D, map_location='cpu')
        print('Loading %s' % config.load_D)

    g_lr = config.g_lr
    d_lr = config.d_lr

    optimG = torch.optim.Adam(params=generator.parameters(), lr=g_lr)
    optimD = torch.optim.Adam(params=discriminator.parameters(), lr=d_lr)

    # Loading config from last training
    if config.start_idx > 0:
        with open(join(config.result_path, 'config-%d.json' % config.start_idx), 'r') as f:
            temp_data = json.load(f)
            alpha = temp_data['alpha']
            start_train_point = temp_data['start_train_point']
            g_lr = temp_data['g_lr']
            d_lr = temp_data['d_lr']

    if config.phase == 'fadein':
        if config.start_idx > 0:
            alpha = temp_data['alpha']
            start_train_point = temp_data['start_train_point']
            g_lr = temp_data['g_lr']
            d_lr = temp_data['d_lr']
        else:
            # alpha: [0, 0.9]
            alpha = 0.0
            start_train_point = 1
        epoch_length = config.epochs // 10
    elif config.phase == 'stabilize':
        if config.start_idx > 0:
            g_lr = temp_data['g_lr']
            d_lr = temp_data['d_lr']
        alpha = 1.0
        start_train_point = 0

    for epoch in range(1 + config.start_idx, config.epochs + 1):

        for i, data in enumerate(tqdm(train_loader)):
            real_images = data['image'].to(device)
            noises = data['noise'].float().to(device)

            optimG.zero_grad()
            fake_images = generator(noises, alpha)
            fake_labels = discriminator(fake_images, alpha)
            gan_loss = ganLoss(fake_labels, True)
            gan_loss.backward()
            optimG.step()

            optimD.zero_grad()
            dis_loss = ganLoss(discriminator(real_images, alpha), True) + \
                       ganLoss(discriminator(fake_images.detach(), alpha), False)
            if config.gan_type == 'wgangp':
                gp_loss = gpLoss(discriminator, real_images, fake_images.detach())
                dis_loss = dis_loss + config.l_gp * gp_loss
            dis_loss.backward()
            optimD.step()

            if config.phase == 'fadein':
                if epoch > epoch_length * start_train_point:
                    start_train_point += 1
                    alpha += 0.1

            if i % 500 == 0:
                if config.gan_type == 'wgangp':
                    print('Epoch: %d/%d | Step: %d/%d | G loss: %.4f | D loss: %.4f | gp loss: %.4f' % (epoch, config.epochs,
                         i, len(train_loader), gan_loss.item(), dis_loss.item(), gp_loss.item()))
                else:
                    print('Epoch: %d/%d | Step: %d/%d | G loss: %.4f | D loss: %.4f' %
                          (epoch, config.epochs, i, len(train_loader), gan_loss.item(), dis_loss.item()))

                if config.output_act == 'tanh':
                    fake_images = (fake_images.detach().cpu().numpy()[0:6].transpose((0, 2, 3, 1)) + 1.) * 0.5
                    real_images = (real_images.detach().cpu().numpy()[0:6].transpose((0, 2, 3, 1)) + 1.) * 0.5
                else:
                    fake_images = fake_images.detach().cpu().numpy()[0:6].transpose((0, 2, 3, 1))
                    real_images = real_images.detach().cpu().numpy()[0:6].transpose((0, 2, 3, 1))
                save_result(rows=2, cols=3, images=fake_images,
                            result_file=join(config.result_path, "fake-epoch-%d-step-%d-alpha-%.1f.png" %
                                             (epoch, i, alpha)))
                save_result(rows=2, cols=3, images=real_images,
                            result_file=join(config.result_path, "real-epoch-%d-step-%d.png" % (epoch, i)))

                writer.add_scalars('loss', {'G loss': gan_loss.item(),
                                            'D loss': dis_loss.item()}, (epoch-1)*len(train_loader) + i)

                if config.gan_type == 'wgangp':
                    writer.add_scalars('loss', {'gp': gp_loss.item()}, (epoch-1)*len(train_loader) + i)

        with open(join(config.result_path, 'config-%d.json' % epoch), 'w') as f:
            temp_data = {
                'alpha': alpha,
                'start_train_point': start_train_point,
                'g_lr': g_lr,
                'd_lr': d_lr,
            }
            json.dump(temp_data, f)

        if epoch % 4 == 0:
            generator.save_model(join(config.ckpt_path, 'G-epoch-%d.pkl' % epoch))
            discriminator.save_model(join(config.ckpt_path, 'D-epoch-%d.pkl' % epoch))

        if epoch % 20 == 0:
            g_lr *= 0.1
            d_lr *= 0.1
            optimG = torch.optim.Adam(params=generator.parameters(), lr=g_lr)
            optimD = torch.optim.Adam(params=discriminator.parameters(), lr=d_lr)

    writer.export_scalars_to_json(join(config.result_path, 'scalars.json'))
    writer.close()


def test(config):
    print(config)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    if not exists(join(config.result_path, 'test-%d' % config.test_epoch)):
        os.makedirs(join(config.result_path, 'test-%d' % config.test_epoch))

    test_dataset = TestDataset()
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=1)

    generator = Generator(resolution=config.resolution, output_act=config.output_act, norm=config.norm, device=device).to(device)

    if config.load_G:
        if torch.cuda.is_available():
            generator.load_model(config.load_G)
        else:
            generator.load_model(config.load_G, map_location='cpu')

    generator.eval()

    for i, data in enumerate(tqdm(test_loader)):
        noises = data['noise'].float().to(device)
        fake_images = generator(noises, alpha=1.0)

        if config.output_act == 'tanh':
            fake_images = (fake_images.detach().cpu().numpy()[0:6].transpose((0, 2, 3, 1)) + 1.) * 0.5
        else:
            fake_images = fake_images.detach().cpu().numpy()[0:6].transpose((0, 2, 3, 1))

        save_result(rows=2, cols=3, images=fake_images,
                    result_file=join(config.result_path, 'test-%d' % config.test_epoch, "fake-%d.png" % i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device_id', type=str, default=None)
    parser.add_argument('--celeba_hq_dir', type=str, default='/media/liuwq/data/Dataset/Celeba/Celeba-HQ')
    parser.add_argument('--train_file', type=str, default='data/train_list.txt')
    parser.add_argument('--ckpt_path', type=str, default='ckpt/reso-4x4/')
    parser.add_argument('--result_path', type=str, default='result/reso-4x4/')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--g_lr', type=float, default=1e-3)
    parser.add_argument('--d_lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--gan_type', type=str, default='vanilla')
    parser.add_argument('--l_gp', type=float, default=10.)
    parser.add_argument('--resolution', type=int, default=4)
    # parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--norm', type=str, default='pixelnorm')
    parser.add_argument('--output_act', type=str, default='linear')

    parser.add_argument('--start_idx', type=int, default=0)

    parser.add_argument('--test_epoch', type=int, default=40)

    parser.add_argument('--load_G', type=str, default=None)
    parser.add_argument('--load_D', type=str, default=None)

    parser.add_argument('--phase', type=str, default='fadein')

    config = parser.parse_args()

    if config.phase == 'fadein' or config.phase == 'stabilize':
        train(config)
    elif config.phase == 'test':
        test(config)

