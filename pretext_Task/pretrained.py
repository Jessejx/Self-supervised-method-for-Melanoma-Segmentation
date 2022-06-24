import torch.optim as optim
from utils import (AverageMeter, Logger, Memory, ModelCheckpoint,
                   NoiseContrastiveEstimator, Progbar)
from utils_loss import *
from dataloader_224 import JigsawLoader
from torchvision.utils import save_image
from architecture_224 import Network, Decoder


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = 'to/your/unlabeled/dataset'
    negative_nb = 5000  # number of negative examples in NCE
    checkpoint_dir = 'jigsaw_models'
    log_filename = 'pretraining_log_jigsaw'

    ''' Dataloader and train_loader '''
    dataset = JigsawLoader(data_dir)
    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=32, num_workers=14)
    ''' architecture and optimizer '''
    encoder = Network().to(device)
    decoder = Decoder().to(device)
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=1e-3, momentum=0.9)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-4)
    optimizers = [encoder_optimizer]
    optimizers += [decoder_optimizer]
    schedulars = build_lr_schedulers(optimizers)

    '''memory bank'''
    memory = Memory(device=device, size=len(dataset), weight=0.5)
    memory.initialize(encoder, train_loader)

    '''some parameters setting'''
    noise_contrastive_estimator = NoiseContrastiveEstimator(device)
    logger = Logger(log_filename)
    loss_weight = 0.5
    n,m = 0,0

    for epoch in range(1000):
        print('\nEpoch: {}'.format(epoch))
        memory.update_weighted_count()
        '''contrastive and generative self-supervised pre-training'''
        encoder.train()
        decoder.train()
        for step, batch in enumerate(train_loader):

            # prepare batch
            images = batch['original'].to(device)
            batch_size = images.size()[0]
            patches = [element.to(device) for element in batch['patches']]
            index = batch['index']
            representations = memory.return_representations(index).to(device).detach()
            # zero grad
            for optimizer in optimizers:
                optimizer.zero_grad()
            # forward
            original_features, patches_features = encoder(images=images, patches=patches, mode=1)
            images_output = decoder(original_features)

            '''Generative Learning Loss'''
            recon_loss = reconstruction_l1_loss(images,images_output)
            mmd_loss = compute_mmd(original_features)/batch_size
            generative_loss = recon_loss + mmd_loss
            '''Contrastive Learning Loss'''
            contrastive_loss_1 = noise_contrastive_estimator(representations, patches_features, index, memory, negative_nb=negative_nb)
            contrastive_loss_2 = noise_contrastive_estimator(representations, original_features, index, memory, negative_nb=negative_nb)
            contrastive_loss = loss_weight * contrastive_loss_1 + (1 - loss_weight) * contrastive_loss_2
            '''loss and backward'''
            loss = generative_loss + contrastive_loss
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()

            # update representation memory
            memory.update(index, original_features.detach().cpu().numpy())

            '''visualize loss and images'''
            if n % 20 ==0:
                print("Epoch", epoch, "contrastive_loss", contrastive_loss.item(), "generative_loss", generative_loss.item())
                print("generative_loss", generative_loss.item(), "recon_loss", recon_loss.item(), "mmd_loss", mmd_loss.item())
                save_image(images, 'images/in/in_{}.jpg'.format(n))
                save_image(images_output, 'images/out/out_{}.jpg'.format(n))
            n+=1

        for schedular in schedulars:
            schedular.step()
        print("!"*10+" lr "+"!"*10, optimizers[0].param_groups[0]['lr'])
        if epoch % 30 == 0:
            torch.save({
                "encoder": encoder.state_dict(),
            }, 'checkpoint_cg_{}.pth'.format(epoch))
