import torch
import train
import mnist
import mnistm
import model
import logging
import os
import datetime
import json
from utils import get_free_gpu

save_name = 'omg'

source_train_loader = mnist.mnist_train_loader
target_train_loader = mnistm.mnistm_train_loader


def single_step(is_sprt=True, size='small', mode_='forward'):
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    separate_or_joint = 'separate' if is_sprt else 'joint'
    file_name = f'{size}_{separate_or_joint}_{mode_}'
    logger_info = dict(filename=f'./logs/{file_name}_{datetime.datetime.now().__str__()}.log', level=logging.INFO)

    if is_sprt:
        order = True if mode_ == 'forward' else False

    encoder = model.Extractor(size=size).cuda()
    classifier = model.Classifier(size=size).cuda()
    discriminator = model.Discriminator(size=size).cuda()
    if is_sprt:
        train.source_only(encoder, classifier, source_train_loader, target_train_loader, save_name,
                          order=order, logger_info=logger_info)
        train.dann(encoder, classifier, discriminator, source_train_loader, target_train_loader, save_name,
                   order=order, logger_info=logger_info)
    else:
        train.joint_ds_training(encoder, classifier, save_name, logger_info=logger_info)


def grid_report():
    path = './logs/gris.json'
    if os.path.exists(path):
        with open(path, 'a') as stream:
            stats = json.load(stream)
    else:
        stats = {}

    if torch.cuda.is_available():
        for is_separate_ in [True, False]:
            s = 'separate' if is_separate_ else 'joint'
            if s not in stats:
                stats[s] = {}
            for arch_size_ in ['small', 'middle', 'large']:
                if arch_size_ not in stats[s]:
                    stats[s][arch_size_] = {}
                for mode_ in ['forward', 'reversed']:
                    if mode_ not in stats[s][arch_size_]:
                        stats[s][arch_size_][mode_] = ''
                    if stats[s][arch_size_][mode_] == 'Done':
                        continue
                    single_step(is_sprt=is_separate_, size=arch_size_, mode_=mode_)
                    stats[s][arch_size_][mode_] = 'Done'
        with open(path, 'a') as stream:
            json.dump(stats, stream)

    else:
        print("There is no GPU -_-!")


if __name__ == "__main__":
    is_separate = False
    arch_size = 'middle'
    mode = 'forward'
    single_step(is_sprt=is_separate, size=arch_size, mode_=mode)

    # grid_report()
