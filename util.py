import numpy as np
import os
import sys
import ntpath
import time
import cv2
import torch
from PIL import Image
from subprocess import Popen, PIPE
from util import *
# from scipy.misc import imresize

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer:
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = 1
        self.win_size = 256
        self.name = opt.name
        self.port = 8097
        self.saved = False
        self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
        self.img_dir = os.path.join(self.web_dir, 'images')
        mkdirs([self.web_dir, self.img_dir])
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.ncols = 4
            self.vis = visdom.Visdom(server='http://localhost', port=8097, env='main')
            if not self.vis.check_connection():
                self.create_visdom_connections()

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:        # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = tensor2im(image)
                    # print(image_numpy.shape)
                    if not image_numpy.shape[:2] == (h, w):
                        image_numpy = cv2.resize(image_numpy, (w, h))
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:     # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = tensor2im(image)
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        # save images to the disk
        for label, image in visuals.items():
            image_numpy = tensor2im(image)
            img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
            save_image(image_numpy, img_path)

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message


def extract_bboxes(mask, exp_ratio=1.0):
    """Compute bounding boxes from mask.
    mask: [height, width]. Mask pixels are either 1 or 0.
    exp_ratio: expand the side length of the rectangle in some ratio. >=1.0
    Returns: bbox array [y1, x1, y2, x2].
    """
    boxes = np.zeros([1, 4], dtype=np.int32)
    m = mask[:, :]
    # Bounding box.
    horizontal_indicies = np.where(np.any(m, axis=0))[0]
    vertical_indicies = np.where(np.any(m, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
        if exp_ratio > 1.0:
            side_x = (x2 - x1 + 1) * exp_ratio
            side_y = (y2 - y1 + 1) * exp_ratio
            x1 = x1 - side_x / 2 if (x1 - side_x/2) > 0 else 0
            x2 = x2 + side_x / 2 if (x2 + side_x/2) < np.size(mask, 2) else np.size(mask, 2)
            y1 = y1 - side_y / 2 if (y1 - side_y/2) > 0 else 0
            y2 = y2 + side_y / 2 if (y2 + side_y/2) < np.size(mask, 1) else np.size(mask, 1)
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    boxes = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def index_closest(pixel, bar):
    n = len(bar)           #  bar:     =====================
    temp = []              #           n           ^       0
    for i in range(n):     #  index:               |
        dis = sum(list(map(lambda x: abs(x[0]-x[1]), zip(pixel, bar[i]))))
        temp.append(dis)
    value = n - temp.index(min(temp))
    return value


def cal_color_vector(color_img, cluster):
    """ Caculate hard-code strain ratio according cluster"""
    color_bar = [[0, 0, 143], [0, 0, 159],[0, 0, 175] ,[0, 0, 191] ,[0, 0, 207] ,[0, 0, 223] ,[0, 0, 239] ,[0, 0, 255] ,
                [0, 16, 255],[0, 32, 255],[0, 48, 255],[0, 64, 255],[0, 80, 255],[0, 96, 255],[0, 112, 255],[0, 128, 255],
                [0, 143, 255],[0, 159, 255],[0, 175, 255],[0, 191, 255],[0, 207, 255],[0, 223, 255],[0, 239, 255],[0, 255, 255],
                [16, 255, 239],[32, 255, 223],[48, 255, 207],[64, 255, 191],[80, 255, 175],[96, 255, 159],[112, 255, 143],[128, 255, 128],
                [143, 255, 112],[159, 255, 96],[175, 255, 80],[191, 255, 64],[207, 255, 48],[223, 255, 32],[239, 255, 16],[255, 255, 0],
                [255, 239, 0],[255, 223, 0],[255, 207, 0],[255, 191, 0],[255, 175, 0],[255, 159, 0],[255, 143, 0],[255, 128, 0],
                [255, 112, 0],[255, 96, 0],[255, 80, 0],[255, 64, 0],[255, 48, 0],[255, 32, 0],[255, 16, 0],[255, 0, 0],
                [239, 0, 0],[223, 0, 0],[207, 0, 0],[191, 0, 0],[175, 0, 0],[159, 0, 0],[143, 0, 0],[128, 0, 0]]
    color_img = np.squeeze(color_img)
    cluster = np.squeeze(cluster)  # shape (1, 1, 256, 256)
    num = torch.max(cluster) + 1   # number clusters
    vec = []
    print(num)
    for i in range(num):
        values = 0
        XYs = torch.nonzero(cluster == i)  # tuple (Xs, Ys)
        n = XYs.size()[0]  # the number of nonzero value
        if n > 0:
            for j in range(n):
                # print(j)
                pixel = color_img[:, XYs[j, 0], XYs[j, 1]]
                value = index_closest(pixel, color_bar)
                values += value
            vec.append(values / n)
        else:
            vec.append(0)
    return vec


