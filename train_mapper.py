import os
import torch
from torch import nn
import sys

sys.path.append('..')
sys.path.append('../styleGAN2_ada_model/stylegan2_ada')
from mapper.networks.level_mapper import LevelMapper
from mapper.dataset import LatentsDataset, LatentsTestDataset
from torch.utils.data import DataLoader
from styleGAN2_ada_model.stylegan2_ada_generator import StyleGAN2adaGenerator
import torchvision
from torch.utils.tensorboard import SummaryWriter
import mapper.id_loss as id_loss
import argparse


def aggregate_loss_dict(agg_loss_dict):
    mean_vals = {}
    for output in agg_loss_dict:
        for key in output:
            mean_vals[key] = mean_vals.setdefault(key, []) + [output[key]]
    for key in mean_vals:
        if len(mean_vals[key]) > 0:
            mean_vals[key] = sum(mean_vals[key]) / len(mean_vals[key])
        else:
            print('{} has no value'.format(key))
            mean_vals[key] = 0
    return mean_vals


class Trainer:
    def __init__(self, args):

        self.data_dir = f'./training_runs/{args.mapper_name}/data'

        self.output_dir = f'./training_runs/{args.mapper_name}'
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_dir = os.path.join(self.output_dir, './logs')
        self.checkpoint_dir = os.path.join(self.output_dir, './checkpoints')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        print(f'============= Loading training data from {self.data_dir} =============')
        print(f'============= Save logs to {self.log_dir}, save ckpts to {self.checkpoint_dir} =============')

        self.best_val_loss = None
        self.max_steps = args.max_steps
        self.save_interval = args.save_interval
        self.image_interval = args.image_interval
        self.board_interval = args.board_interval
        self.val_interval = args.val_interval
        self.batch_size = 1
        self.test_batch_size = 1
        self.learning_rate = args.learning_rate

        self.alpha = args.alpha
        self.input_dim = args.input_dim

        self.device = 'cuda:0'

        self.test_index = 0
        self.global_step = 0

        self.mapper = LevelMapper(input_dim=self.input_dim).to(self.device)
        self.truncation_psi = args.truncation_psi
        self.Generator = StyleGAN2adaGenerator('stylegan2_ada', None, truncation_psi=self.truncation_psi)

        if args.resume != '':
            self.load_weights(args.resume)

        self.latent_l2_loss = nn.MSELoss().to(self.device).eval()
        self.latent_l2_lambda = args.latent_l2_lambda
        self.img_l2_lambda_res = args.img_l2_lambda_res
        self.img_l2_lambda_origin = args.img_l2_lambda_origin

        self.id_lambda = args.id_lambda

        self.id_loss = id_loss.IDLoss().to(self.device).eval()

        # Initialize optimizer
        self.optimizer = self.configure_optimizers()

        # Initialize dataset
        self.train_dataset, self.val_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=0,
                                           drop_last=True)
        self.val_dataloader = DataLoader(self.val_dataset,
                                         batch_size=self.test_batch_size,
                                         shuffle=False,
                                         num_workers=0,
                                         drop_last=True)

        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.test_batch_size,
                                          shuffle=False,
                                          num_workers=0,
                                          drop_last=True)

    def train(self):
        self.mapper.train()
        while self.global_step < self.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()

                origin_wp, res_wp, mask = batch

                origin_wp = origin_wp.to(self.device)

                res_wp = res_wp.to(self.device)
                mask = mask.to(self.device)
                with torch.no_grad():
                    res_x, _ = self.Generator.model(
                        z=res_wp,
                        c=self.Generator.model.c_dim,
                        truncation_psi=self.truncation_psi,
                        truncation_cutoff=None,
                        input_latent_space_type='wp')
                    origin_img, _ = self.Generator.model(
                        z=origin_wp,
                        c=self.Generator.model.c_dim,
                        truncation_psi=self.truncation_psi,
                        truncation_cutoff=None,
                        input_latent_space_type='wp')
                mapper_input = torch.clone(origin_wp)

                w_hat = origin_wp
                w_hat[:, :8, :] += self.mapper(mapper_input) * self.alpha

                x_hat, _ = self.Generator.model(
                    z=w_hat,
                    c=self.Generator.model.c_dim,
                    truncation_psi=self.truncation_psi,
                    truncation_cutoff=None,
                    input_latent_space_type='wp')

                loss, loss_dict = self.calc_loss(res_w=res_wp, res_x=res_x, w_hat=w_hat, x_hat=x_hat,
                                                 origin_img=origin_img, mask=mask)

                loss.backward()
                self.optimizer.step()

                # Logging related
                if self.global_step % self.image_interval == 0 or (
                        self.global_step < 1000 and self.global_step % 1000 == 0):
                    self.parse_and_log_images(res_x, x_hat, origin_img, title='images_train')
                if self.global_step % self.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Validation related
                val_loss_dict = None

                if self.global_step % self.val_interval == 0 or self.global_step == self.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        if self.global_step != 0:
                            self.checkpoint_me(val_loss_dict, is_best=True)
                if self.global_step != 0:
                    if self.global_step % self.save_interval == 0 or self.global_step == self.max_steps:
                        if val_loss_dict is not None:
                            self.checkpoint_me(val_loss_dict, is_best=False)
                        else:
                            self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.max_steps:
                    break

                self.global_step += 1

    def validate(self):
        self.mapper.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.val_dataloader):
            if batch_idx > 10:
                break

            origin_wp, res_wp, mask = batch
            origin_wp = origin_wp.to(self.device).float()

            res_wp = res_wp.to(self.device).float()
            mask = mask.to(self.device)
            with torch.no_grad():

                res_x, _ = self.Generator.model(
                    z=res_wp,
                    c=self.Generator.model.c_dim,
                    truncation_psi=self.truncation_psi,
                    truncation_cutoff=None,
                    input_latent_space_type='wp')
                origin_img, _ = self.Generator.model(
                    z=origin_wp,
                    c=self.Generator.model.c_dim,
                    truncation_psi=self.truncation_psi,
                    truncation_cutoff=None,
                    input_latent_space_type='wp')

                mapper_input = torch.clone(origin_wp)
                w_hat = origin_wp
                w_hat[:, :8, :] += self.mapper(mapper_input) * self.alpha * 1.2
                x_hat, _ = self.Generator.model(
                    z=w_hat,
                    c=self.Generator.model.c_dim,
                    truncation_psi=self.truncation_psi,
                    truncation_cutoff=None,
                    input_latent_space_type='wp')
                loss, cur_loss_dict = self.calc_loss(res_w=res_wp, res_x=res_x, w_hat=w_hat, x_hat=x_hat,
                                                     origin_img=origin_img, mask=mask)

            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            self.parse_and_log_images(res_x, x_hat, origin_img, title='images_val', index=batch_idx)

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:
                break

        loss_dict = aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')
        count = 0
        for batch_idx, batch in enumerate(self.test_dataloader):
            if batch_idx < self.test_index:
                continue

            if count > 10:
                break
            origin_wp, mask, origin_img = batch
            origin_wp = origin_wp.to(self.device).float()

            mask = mask.to(self.device).float()
            origin_img = origin_img.to(self.device).float()
            with torch.no_grad():
                mapper_input = torch.clone(origin_wp)
                w_hat = origin_wp
                w_hat[:, :8, :] += self.mapper(mapper_input) * self.alpha * 1.2
                x_hat, _ = self.Generator.model(
                    z=w_hat,
                    c=self.Generator.model.c_dim,
                    truncation_psi=self.truncation_psi,
                    truncation_cutoff=None,
                    input_latent_space_type='wp')
            # Logging related
            res = origin_img * mask + x_hat * (1 - mask)
            self.parse_and_log_images(x_hat, res, origin_img, title='images_test', index=batch_idx)
            count += 1
            self.test_index += 1
            if self.test_index >= len(self.test_dataset):
                self.test_index = 0

        self.mapper.train()
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(
                    '**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def configure_optimizers(self):
        params = list(self.mapper.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        return optimizer

    def configure_datasets(self):
        train_dataset = LatentsDataset(data_dir=self.data_dir, mode='train')
        val_dataset = LatentsDataset(data_dir=self.data_dir, mode='train')
        test_dataset = LatentsTestDataset(data_dir=self.data_dir)
        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of val samples: {}".format(len(val_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        return train_dataset, val_dataset, test_dataset

    def calc_loss(self, res_w, res_x, w_hat, x_hat, origin_img, mask):
        loss_dict = {}
        loss = 0.0

        loss_l2_latent = self.latent_l2_loss(w_hat, res_w)
        loss_dict['loss_l2_latent'] = float(loss_l2_latent)
        loss += loss_l2_latent * self.latent_l2_lambda

        loss_l2_img = torch.mean(((res_x - x_hat)) ** 2, dim=[0, 1, 2, 3])
        loss_dict['loss_l2_res_img'] = float(loss_l2_img)
        loss += loss_l2_img * self.img_l2_lambda_res

        loss_l2_img = torch.mean(((origin_img - x_hat) * mask) ** 2, dim=[0, 1, 2, 3])
        loss_dict['loss_l2_origin_img'] = float(loss_l2_img)
        loss += loss_l2_img * self.img_l2_lambda_origin

        if self.id_lambda > 0:
            loss_id, sim_improvement = self.id_loss(x_hat, res_x)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_id * self.id_lambda

        loss_dict['loss'] = float(loss)
        return loss, loss_dict

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            # pass
            print(f"step: {self.global_step} \t metric: {prefix}/{key} \t value: {value}")

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)

    def parse_and_log_images(self, x, x_hat, origin_img, title, index=None):
        if index is None:
            path = os.path.join(self.log_dir, title, f'{str(self.global_step).zfill(5)}.jpg')
        else:
            path = os.path.join(self.log_dir, title, f'{str(self.global_step).zfill(5)}_{str(index).zfill(5)}.jpg')

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torchvision.utils.save_image(torch.cat([x.detach().cpu(), x_hat.detach().cpu(), origin_img.detach().cpu()]),
                                     path,
                                     normalize=True, scale_each=True, range=(-1, 1))

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.mapper.state_dict(),
            'alpha': self.alpha
        }

        return save_dict

    def load_weights(self, checkpoint_path):
        print('Loading from checkpoint: {}'.format(checkpoint_path))
        ckpt = torch.load(checkpoint_path)

        self.mapper.load_state_dict(ckpt['state_dict'], strict=True)


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Edit image synthesis with given semantic boundary.')

    parser.add_argument('--mapper_name', type=str, required=True,
                        help='model name (required)')

    parser.add_argument('--max_steps', type=int, default=100000,
                        help='max steps.')

    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='learning rate.')
    parser.add_argument('--val_interval', type=int, default=2000,
                        help='interval of validation.')
    parser.add_argument('--board_interval', type=int, default=50,
                        help='interval of printing metrics.')
    parser.add_argument('--image_interval', type=int, default=100,
                        help='interval of saving images.')
    parser.add_argument('--save_interval', type=int, default=2000,
                        help='interval of saving models.')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='alpha.')

    parser.add_argument('--input_dim', type=int, default=512,
                        help='input dim.')

    parser.add_argument('--id_lambda', type=float, default=0.1,
                        help='id loss weight.')

    parser.add_argument('--img_l2_lambda_origin', type=float, default=0.4,
                        help='original image L2 loss weight.')

    parser.add_argument('--img_l2_lambda_res', type=float, default=0.4,
                        help='result image L2 loss weight.')

    parser.add_argument('--latent_l2_lambda', type=float, default=0.1,
                        help='latent code L2 loss weight.')

    parser.add_argument('--truncation_psi', type=float, default=0.8,
                        help='truncation_psi.')

    parser.add_argument('--resume', type=str, default='',
                        help='resume model path.')

    return parser.parse_args()


def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
