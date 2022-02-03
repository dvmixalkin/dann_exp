import torch
import numpy as np
import model.utils as utils
import torch.optim as optim
import torch.nn as nn
import model.test as test
from model.utils import save_model
from model.utils import visualize
from model.utils import set_model_mode
import model.params as params
import logging


def source_only(encoder, classifier, source_loader_creator, target_loader_creator, save_name, order=True, logger_info=None):
    source_train_loader, source_test_loader = source_loader_creator['train'], source_loader_creator['test']
    target_train_loader, target_test_loader = target_loader_creator['train'], target_loader_creator['test']

    file_name = logger_info['filename']
    level = logger_info['level']
    logging.basicConfig(filename=file_name, level=level)
    msg = "Source-only training"
    logging.info(msg)
    print(msg)

    classifier_criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(
        list(encoder.parameters()) +
        list(classifier.parameters()),
        lr=0.01, momentum=0.9)

    for epoch in range(params.epochs):
        msg = 'Epoch : {}'.format(epoch)
        logging.info(msg)
        print(msg)
        set_model_mode('train', [encoder, classifier])

        start_steps = epoch * len(source_train_loader)
        total_steps = params.epochs * len(target_train_loader)

        for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):
            source_image, source_label = source_data
            if source_image.shape[1] != 3:
                source_image = torch.cat((source_image, source_image, source_image), 1)  # MNIST convert to 3 channel
            target_image, target_label = target_data
            if target_image.shape[1] != 3:
                target_image = torch.cat((target_image, target_image, target_image), 1)  # MNIST convert to 3 channel

            p = float(batch_idx + start_steps) / total_steps

            if order:
                # source_image, source_label
                image_, label_ = source_image.cuda(), source_label.cuda()  # 32
            else:
                image_, label_ = target_image.cuda(), target_label.cuda()

            optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()
            source_feature = encoder(image_)

            # Classification loss
            class_pred = classifier(source_feature)
            class_loss = classifier_criterion(class_pred, label_)

            class_loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 150 == 0:
                msg = '[{}/{} ({:.0f}%)]\tClass Loss: {:.6f}'.format(batch_idx * len(image_),
                                                                     len(source_train_loader.dataset),
                                                                     100. * batch_idx / len(source_train_loader),
                                                                     class_loss.item())
                logging.info(msg)
                print(msg)
        # if (epoch + 1) % 3 == 0:
        if order:
            source_t_loader, target_t_loader = source_test_loader, target_test_loader
        else:
            source_t_loader, target_t_loader = target_test_loader, source_test_loader

        test.tester(encoder, classifier, None, source_t_loader, target_t_loader, training_mode='source_only',
                    logger_info=logger_info)
    save_model(encoder, classifier, None, 'source', save_name)
    visualize(encoder, source_test_loader, target_test_loader, 'source', save_name, order=order)


def dann(encoder, classifier, discriminator, source_loader_creator, target_loader_creator, save_name, order=True, logger_info=None):
    source_train_loader, source_test_loader = source_loader_creator['train'], source_loader_creator['test']
    target_train_loader, target_test_loader = target_loader_creator['train'], target_loader_creator['test']

    file_name = logger_info['filename']
    level = logger_info['level']
    logging.basicConfig(filename=file_name, level=level)
    msg = "DANN training"
    logging.info(msg)
    print(msg)

    classifier_criterion = nn.CrossEntropyLoss().cuda()
    discriminator_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(
        list(encoder.parameters()) +
        list(classifier.parameters()) +
        list(discriminator.parameters()),
        lr=0.01,
        momentum=0.9)

    for epoch in range(params.epochs):
        msg = 'Epoch : {}'.format(epoch)
        logging.info(msg)
        print(msg)
        set_model_mode('train', [encoder, classifier, discriminator])

        start_steps = epoch * len(source_train_loader)
        total_steps = params.epochs * len(target_train_loader)

        for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):

            source_image, source_label = source_data
            target_image, target_label = target_data

            p = float(batch_idx + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            source_image = torch.cat((source_image, source_image, source_image), 1)

            source_image, source_label = source_image.cuda(), source_label.cuda()
            target_image, target_label = target_image.cuda(), target_label.cuda()
            combined_image = torch.cat((source_image, target_image), 0)

            optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()

            combined_feature = encoder(combined_image)
            source_feature = encoder(source_image)

            # 1.Classification loss
            class_pred = classifier(source_feature)
            class_loss = classifier_criterion(class_pred, source_label)

            # 2. Domain loss
            domain_pred = discriminator(combined_feature, alpha)

            domain_source_labels = torch.zeros(source_label.shape[0]).type(torch.LongTensor)
            domain_target_labels = torch.ones(target_label.shape[0]).type(torch.LongTensor)
            domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).cuda()
            domain_loss = discriminator_criterion(domain_pred, domain_combined_label)

            total_loss = class_loss + domain_loss
            total_loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 150 == 0:
                msg = '[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'. \
                    format(batch_idx * len(target_image), len(target_train_loader.dataset),
                           100. * batch_idx / len(target_train_loader), total_loss.item(), class_loss.item(),
                           domain_loss.item())
                logging.info(msg)
                print(msg)
        # if (epoch + 1) % 10 == 0:
        test.tester(encoder, classifier, discriminator, source_test_loader, target_test_loader, training_mode='dann', logger_info=logger_info)

    save_model(encoder, classifier, discriminator, 'source', save_name)
    visualize(encoder, source_test_loader, target_test_loader, 'source', save_name)


def joint_ds_training(encoder, classifier, train_set, test_sets=None, save_name=None, logger_info=None):
    combined_train_loader, combined_test_loader = train_set['train'], train_set['test']

    file_name = logger_info['filename']
    level = logger_info['level']
    logging.basicConfig(filename=file_name, level=level)
    msg = "joint ds training"
    logging.info(msg)
    print(msg)

    classifier_criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(
        list(encoder.parameters()) +
        list(classifier.parameters()),
        lr=0.01, momentum=0.9)

    for epoch in range(params.epochs):
        msg = 'Epoch : {}'.format(epoch)
        logging.info(msg)
        print(msg)
        set_model_mode('train', [encoder, classifier])

        start_steps = epoch * len(combined_train_loader)
        total_steps = params.epochs * len(combined_train_loader)

        for batch_idx, source_data in enumerate(combined_train_loader):
            source_image, source_label = source_data
            p = float(batch_idx + start_steps) / total_steps

            source_image, source_label = source_image.cuda(), source_label.cuda()  # 32

            optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()

            source_feature = encoder(source_image)

            # Classification loss
            class_pred = classifier(source_feature)
            class_loss = classifier_criterion(class_pred, source_label)

            class_loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 150 == 0:
                msg = '[{}/{} ({:.0f}%)]\tClass Loss: {:.6f}'.format(batch_idx * len(source_image),
                                                                     len(combined_train_loader.dataset),
                                                                     100. * batch_idx / len(combined_train_loader),
                                                                     class_loss.item())
                logging.info(msg)
                print(msg)

        # if (epoch + 1) % 3 == 0:
        test.tester(encoder, classifier, None, combined_test_loader, combined_test_loader, training_mode='source_only', logger_info=logger_info)
        if test_sets:
            for test_set in test_sets:
                test.tester(encoder, classifier, None, combined_test_loader, test_set['test'], training_mode='source_only', logger_info=logger_info)
    save_model(encoder, classifier, None, 'joint', save_name)
    # visualize(encoder, 'joint', save_name)
