import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
# from net.models import deeplabv3plus
from sklearn.metrics import accuracy_score
import loss
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
# from apex import amp
from tensorboardX import SummaryWriter
# Loading model
from models.pspnet import pspnet_res50
from Loading_Lesion_Image import *
import random
import argparse

INPUT_SIZE = '224, 224'
w, h = map(int, INPUT_SIZE.split(','))
LEARNING_RATE = 0.001
MOMENTUM = 0.9
POWER = 0.9
WEIGHT_DECAY = 0.0005
NUM_CLASSES = 2
TRAIN_NUM = 2000
BATCH_SIZE = 16
EPOCH = 25
STEPS = (TRAIN_NUM/BATCH_SIZE)*EPOCH
FP16 = False
NAME = 'pspnet/'
best_score = 0

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(LEARNING_RATE, i_iter, STEPS, POWER)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def val_mode_seg(valloader, model, path, epoch):
    dice = []
    sen = []
    spe = []
    acc = []
    jac_score = []
    for index, batch in enumerate(valloader):

        data, mask, name = batch
        data = data.cuda()
        mask = mask[0].data.numpy()
        val_mask = np.int64(mask > 0)
        # print(name)

        model.eval()
        with torch.no_grad():
            pred = model(data)

        pred = torch.softmax(pred, dim=1).cpu().data.numpy()
        pred_arg = np.argmax(pred[0], axis=0)


        # #y_pred
        y_true_f = val_mask.reshape(val_mask.shape[1]*val_mask.shape[2], order='F')
        y_pred_f = pred_arg.reshape(pred_arg.shape[0]*pred_arg.shape[1], order='F')

        intersection = np.float64(np.sum(y_true_f * y_pred_f))
        dice.append((2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f)))
        sen.append(intersection / np.sum(y_true_f))
        intersection0 = np.float(np.sum((1 - y_true_f) * (1 - y_pred_f)))
        spe.append(intersection0 / np.sum(1 - y_true_f))
        acc.append(accuracy_score(y_true_f, y_pred_f))
        jac_score.append(intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection))

    return np.array(acc), np.array(dice), np.array(sen), np.array(spe), np.array(jac_score)


def Jaccard(pred_arg, mask):
    pred_arg = np.argmax(pred_arg.cpu().data.numpy(), axis=1)
    mask = mask.cpu().data.numpy()

    y_true_f = mask.reshape(mask.shape[0] * mask.shape[1] * mask.shape[2], order='F')
    y_pred_f = pred_arg.reshape(pred_arg.shape[0] * pred_arg.shape[1] * pred_arg.shape[2], order='F')

    intersection = np.float(np.sum(y_true_f * y_pred_f))
    jac_score = intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection)

    return jac_score

def main():

    """Create the network and start the training."""
    writer = SummaryWriter('models/' + NAME)

    cudnn.enabled = True

    ############# Create coarse segmentation network
    model = pspnet_res50()
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE,momentum=0.9,weight_decay=WEIGHT_DECAY)
    model.cuda()
    if FP16 is True:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model = torch.nn.DataParallel(model)

    model.train()
    model.float()

    Dice_loss = loss.Dice()

    cudnn.benchmark = True

    ############# Load training and validation data
    trainloader = data_loader(mode='train')
    valloader = data_loader(mode='valid')

    path = 'Record/' + NAME
    if not os.path.isdir(path):
        os.mkdir(path)
    f_path = path + 'output__noise.txt'

    val_jac = []
    best_score=0
    list_dice= []
    mean_acc=0
    mean_dice=0
    mean_sen=0
    mean_spe=0
    mean_jac=0

    ############# Start the training
    for epoch in range(EPOCH):

        train_loss_total = []
        train_jac = []

        for i_iter, batch in tqdm(enumerate(trainloader)):

            step = (TRAIN_NUM/BATCH_SIZE)*epoch+i_iter

            images, labels, name = batch
            images = images.cuda()
            labels = labels.cuda().squeeze(1)

            optimizer.zero_grad()
            lr = adjust_learning_rate(optimizer, step)

            model.train()
            preds = model(images)

            term = Dice_loss(preds, labels)
            # term = loss_D + 0.05 * loss_R

            if FP16 is True:
                with amp.scale_loss(term, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                term.backward()

            optimizer.step()

            writer.add_scalar('learning_rate', lr, step)
            writer.add_scalar('loss', term.cpu().data.numpy(), step)

            # train_loss_D.append(loss_D.cpu().data.numpy())
            # train_loss_R.append(loss_R.cpu().data.numpy())
            train_loss_total.append(term.cpu().data.numpy())
            train_jac.append(Jaccard(preds, labels))


        print("train_epoch%d:  lossDice=%f, Jaccard=%f \n" % (epoch,1 - np.nanmean(train_loss_total), np.nanmean(train_jac)))

        ############# Start the validation
        [vacc, vdice, vsen, vspe, vjac_score] = val_mode_seg(valloader, model, path, epoch)
        line_val = "val%d: vacc=%f, vdice=%f, vsensitivity=%f, vspecifity=%f, vjac=%f \n" % \
                (epoch, np.nanmean(vacc), np.nanmean(vdice), np.nanmean(vsen), np.nanmean(vspe),
                    np.nanmean(vjac_score))

        print(line_val)
        # f = open(f_path, "a")
        # f.write(line_val)

        ############# Plot val curve
        # val_jac.append(np.nanmean(vjac_score))
        # plt.figure()
        # plt.plot(val_jac, label='val jaccard', color='blue', linestyle='--')
        # plt.legend(loc='best')
        #
        # plt.savefig(os.path.join(path, 'jaccard.png'))
        # plt.clf()
        # plt.close()
        # plt.show()
        #
        # plt.close('all')

        writer.add_scalar('val_Jaccard', np.nanmean(vjac_score), epoch)

        score = np.nanmean(vdice)
        # if score > best_score:
        #     best_score = score
        #     print("!!!!!!!!!!!!!!!!!!!!!!!!!Saving...")
        mean_acc += np.nanmean(vacc)
        mean_dice += np.nanmean(vdice)
        mean_sen += np.nanmean(vsen)
        mean_spe += np.nanmean(vspe)
        mean_jac += np.nanmean(vjac_score)
        list_dice.append(score)
    print("!!!!")
    print("acc",mean_acc/EPOCH)
    print("dice",mean_dice/EPOCH)
    print("sen",mean_sen/EPOCH)
    print("spe",mean_spe/EPOCH)
    print("jac",mean_jac/EPOCH)
    print("var_dice",np.var(list_dice))
    line_val = "vacc=%f, vdice=%f, vsensitivity=%f, vspecifity=%f, vjac=%f, vae_dice=%f \n" % \
               (mean_acc/EPOCH, mean_dice/EPOCH, mean_sen/EPOCH, mean_spe/EPOCH, mean_jac/EPOCH, np.var(list_dice)*100)
    f = open(f_path, "a")
    f.write(line_val)
if __name__ == '__main__':
    main()

