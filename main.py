import datetime
import json
import logging
import os
import torch
import model.model as model
import model.train as train

from datasets.mnist import create_mnist
from datasets.mnistm import create_mnist_m
from datasets.svhn import create_svhn
from datasets.combined_mnist import create_loaders

save_name = 'omg'


def single_step(source, target, is_sprt=True, size='small', mode_='forward'):
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
        train.source_only(encoder, classifier,
                          source_loader_creator=source,
                          target_loader_creator=target,
                          save_name=save_name,
                          order=order,
                          logger_info=logger_info)
        train.dann(encoder, classifier, discriminator, source, target, save_name,
                   order=order, logger_info=logger_info)
    else:
        # train_set, test_sets =
        train.joint_ds_training(encoder, classifier, train_set=source, test_sets=target, save_name=save_name,
                                logger_info=logger_info)


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
    #  1) для прогона сначала на сорсе , а потом на доменной адаптации нужно раскомментировать необходимые датасеты
    mnist_loader_creator = create_mnist(transform_hyperparameters_version='1')
    mnistm_loader_creator = create_mnist_m(transform_hyperparameters_version='1')
    svhn_loader_creator = create_svhn(transform_hyperparameters_version='1')

    # 2) в словаре необходимо указать, какие датасеты будем смешивать
    used_datasets = {
        'mnist': '1',
        'mnist_m': '1',
        'svhn': '1'
    }
    # 3) инициализация смешанного датасета
    combined_loader_creator = create_loaders(datasets_list=used_datasets)

    # 4) флаг для указания типа обучения:
    # True - прямой прогон на сорсе + доменная адаптация на таргете
    # False - обучение без доменной адаптации на смешанном датасете
    # NOTE! : если обучение на смешанном датасете, настоятельно рекомендуется закомменировать строки с инициализацией
    # датасетов в п. 1 чтобы не упасть по размеру оперативнйо памяти(все 3 датасета "съедают" примерно 15,6 Гб ОЗУ)
    is_separate = False

    # 5) выбор архитектуры:
    # "small" - маленькая(1-я в статье) сетка
    # "middle" - большая(2-я в статье), но с большой головой
    # "large" - средняя(3-я в статье), с 3 свертками, но маленькой головой
    # "mixed" - кастомная сетка: экстрактор от 3-ей сетки, голова от 2ой сетки
    arch_size = 'mixed'

    # 6) опциональный флаг('forward'): введен для удобства, чтобы каждый раз не править сорс и таргет датасеты
    # если сорс это MNIST, а таргет это MNIST-M, при указании:
    # `mode = 'forward'` обучение будет проходить на сорсе(MNIST) + доменная адаптация на таргете(MNIST-M).
    # `mode = '{any_string}'` обучение будет проходить на таргете(MNIST-M) + доменная адаптация на сорсе(MNIST).
    mode = 'forward'

    # 7) функция запуска обучения.
    # source - запрашивает сорс датасет
    # target - запрашивает
    #       1) таргет датасет для случая is_separate = True(случай с доменной адаптацией)
    #       2) список с датасетами, на которых будет происходить тестирование при is_separate = False
    #          (случай для обучения на смешанном датасете)
    #
    if is_separate:
        source = mnist_loader_creator
        # source = mnistm_loader_creator
        # source = svhn_loader_creator
        # target = mnist_loader_creator
        target = mnistm_loader_creator
        # target = svhn_loader_creator
    else:
        source = combined_loader_creator
        target = [
            mnist_loader_creator,
            mnistm_loader_creator,
            svhn_loader_creator
        ]

    single_step(
        source=source,
        target=target,
        is_sprt=is_separate,
        size=arch_size, mode_=mode
    )

    # @TODO WIP
    # функция предполагает прохождение по всем указанным параметрам(wip)
    # grid_report(**args)
    # @TODO сохранение весов обученных моделей с нормальными названиями
    # @TODO логгирование
