import datetime
import json
import logging
import os
import torch
from model import model
from model import train
from datasets.combined_mnist import create_loaders
from datasets.mnist import create_mnist
from datasets.mnistm import create_mnist_m
from datasets.svhn import create_svhn
from model import params
from model.model import weights_loader


def single_step(source, target, paths, save_name, is_sprt=True, size='small', mode_='forward'):
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    separate_or_joint = 'separate' if is_sprt else 'joint'
    file_name = f'{size}_{separate_or_joint}_{mode_}'
    logger_info = dict(filename=f'./logs/{file_name}_{datetime.datetime.now().__str__()}.log', level=logging.INFO)

    if is_sprt:
        order = True if mode_ == 'forward' else False

    encoder = model.Extractor(size=size).cuda()
    encoder = weights_loader(encoder, 'encoder', paths)

    classifier = model.Classifier(size=size).cuda()
    classifier = weights_loader(classifier, 'classifier', paths)

    discriminator = model.Discriminator(size=size).cuda()
    discriminator = weights_loader(discriminator, 'discriminator', paths)

    if is_sprt:
        train.source_only(encoder, classifier,
                          source_loader_creator=source,
                          target_loader_creator=target,
                          save_name=save_name,
                          order=order,
                          logger_info=logger_info)
        train.dann(encoder, classifier, discriminator, source, target, save_name,
                   order=order, logger_info=logger_info)
    else:
        train.joint_ds_training(encoder, classifier, train_set=source, test_sets=target, save_name=save_name,
                                logger_info=logger_info)


# @TODO WIP
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
            for arch_size_ in ['small', 'middle', 'large', 'mixed']:
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


# @TODO WIP
def get_grid_json(transform_hyperparameters_version=['1', '1', '1']):
    mnist_creator = create_mnist(transform_hyperparameters_version[0])
    mnistm_creator = create_mnist_m(transform_hyperparameters_version[1])
    svhn_creator = create_svhn(transform_hyperparameters_version[2])

    return {
        'source_dann': {
            'mnist': {
                'mnistm': [mnist_creator, mnistm_creator],
                'svhn': [mnist_creator, svhn_creator]
            },
            'mnistm': {
                'mnist': [mnistm_creator, mnist_creator],
                'svhn': [mnistm_creator, svhn_creator]
            },
            'svhn': {
                'mnist': [svhn_creator, mnist_creator],
                'mnistm': [svhn_creator, mnistm_creator]
            }
        },
        'joint': {
            'mnist+mnistm': ['mnist', 'mnist_m'],
            'mnist+svhn': ['mnist', 'svhn'],
            'mnistn+svhn': ['mnist_m', 'svhn'],
            'mnist+mnistm+svhn': ['mnist', 'mnist_m', 'svhn']
        }
    }


if __name__ == "__main__":
    # 1) флаг для указания типа обучения:
    # True - прямой прогон на сорсе + доменная адаптация на таргете
    # False - обучение без доменной адаптации на смешанном датасете
    # NOTE! : если обучение на смешанном датасете, настоятельно рекомендуется закомменировать строки с инициализацией
    # датасетов в п.2 чтобы не упасть по размеру оперативнйо памяти(все 3 датасета "съедают" примерно 15,6 Гб ОЗУ)
    is_separate = True

    # 2) выбор архитектуры:
    # "small" - маленькая(1-я в статье) сетка
    # "middle" - большая(2-я в статье), но с большой головой
    # "large" - средняя(3-я в статье), с 3 свертками, но маленькой головой
    # "mixed" - кастомная сетка: экстрактор от 3-ей сетки, голова от 2ой сетки
    arch_size = 'small'

    # 3) опциональный флаг('forward'): введен для удобства, чтобы каждый раз не править сорс и таргет датасеты
    # если сорс это MNIST, а таргет это MNIST-M, при указании:
    # `mode = 'forward'` обучение будет проходить на сорсе(MNIST) + доменная адаптация на таргете(MNIST-M).
    # `mode = '{any_string}'` обучение будет проходить на таргете(MNIST-M) + доменная адаптация на сорсе(MNIST).
    mode = 'forward'

    set_norm_hyperparameters = []
    data_warehouse = {
        'mnist': create_mnist(transform_hyperparameters_version='2'),
        'mnist_m': create_mnist_m(transform_hyperparameters_version='2'),
        'svhn': create_svhn(transform_hyperparameters_version='2')
    }
    ds_names = ''

    if is_separate:
        # для прогона сначала на сорсе , а потом на доменной адаптации нужно раскомментировать необходимые датасеты
        source_dataset = 'mnist'
        # source_dataset = 'mnist_m'
        # source_dataset = 'svhn'

        # target_dataset = 'mnist'
        target_dataset = 'mnist_m'
        # target_dataset = 'svhn'

        source = data_warehouse[source_dataset]
        target = data_warehouse[target_dataset]

        ds_names = f'{source_dataset}2{target_dataset}'
        print(f'Training on {ds_names}')
    else:
        # в словаре необходимо указать, какие датасеты будем смешивать
        used_datasets = {
            # 'mnist': '1',
            'mnist_m': '1',
            'svhn': '1'
        }
        source_name = '-'.join(list(used_datasets.keys()))
        # инициализация смешанного датасета
        source = create_loaders(datasets_list=used_datasets)
        #  раскомментировать строки, которые необходимы для тестирования
        target = [
            data_warehouse['mnist'],
            # data_warehouse['mnist_m'],
            # data_warehouse['svhn']
        ]
        ds_names = ''

    weights = {
        'encoder': f'./trained_models/encoder_source_{arch_size}_{ds_names}_{params.epochs}.pt',
        'classifier': f'./trained_models/classifier_source_{arch_size}_{ds_names}_{params.epochs}.pt',
        'discriminator': None
    }

    # 4) функция запуска обучения.
    # source - запрашивает сорс датасет
    # target - запрашивает
    #       1) таргет датасет для случая is_separate = True(случай с доменной адаптацией)
    #       2) список с датасетами, на которых будет происходить тестирование при is_separate = False
    #          (случай для обучения на смешанном датасете)
    #
    single_step(
        source=source,
        target=target,
        paths=weights,
        save_name=f'{arch_size}_{ds_names}_{params.epochs}',
        is_sprt=is_separate,
        size=arch_size, mode_=mode
    )

    # @TODO WIP
    # функция предполагает прохождение по всем указанным параметрам(wip)
    # grid_report(**args)
    # @TODO сохранение весов обученных моделей с нормальными названиями
    # @TODO поправить логгирование
