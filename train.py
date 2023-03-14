import time
import argparse
import torch
from model import EnhanceGANModel
from dataset import AlignedDataset
from util import Visualizer


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default=r'E:\Database\RUIJIN_data\03_DATASET\mask_dataset', type=str, help='path/to/data')
    parser.add_argument('--EnhanceT', default=True, help='')

    # training related
    parser.add_argument('--isTrain', default=True, help='')
    parser.add_argument('--epoch_count', default=1, help='the starting epoch count')
    parser.add_argument('--lr', default=0.0002, help='learning rate')
    parser.add_argument('--lr_policy', default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--batch_size', default=1, help='')
    parser.add_argument('--niter', default=100, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', default=100, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--epoch', default='latest', help='')
    parser.add_argument('--gpu_ids', default=[0], help='which device')

    parser.add_argument('--checkpoints_dir', default='./checkpoints', help='')
    parser.add_argument('--name', default='experiment_name', help='')
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_parser()
    dataset = AlignedDataset(opt.data_root, 'train')
    dataset_size = len(dataset)
    dataset = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=not False,
        num_workers=0)
    print('The number of training images = %d' % dataset_size)

    model = EnhanceGANModel(opt)
    model.setup(opt)

    visualizer = Visualizer(opt)
    total_iters = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % 100 == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % 400 == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % 1000 == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % 100 == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % 5000 == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % 5 == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, 100 + 100, time.time() - epoch_start_time))
        model.update_learning_rate()


