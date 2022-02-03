import torch
import numpy as np
from model.utils import set_model_mode
import logging


def tester(encoder, classifier, discriminator, source_test_loader, target_test_loader, training_mode, logger_info):
    file_name = logger_info['filename']
    level = logger_info['level']
    logging.basicConfig(filename=file_name, level=level)

    msg = "Model test ..."
    logging.info(msg)
    print(msg)

    encoder.cuda()
    classifier.cuda()
    set_model_mode('eval', [encoder, classifier])

    if training_mode == 'dann':
        discriminator.cuda()
        set_model_mode('eval', [discriminator])
        domain_correct = 0

    source_correct = 0
    target_correct = 0

    for batch_idx, (source_data, target_data) in enumerate(zip(source_test_loader, target_test_loader)):
        p = float(batch_idx) / len(source_test_loader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # 1. Source input -> Source Classification
        source_image, source_label = source_data
        source_image, source_label = source_image.cuda(), source_label.cuda()
        if source_image.shape[1] != 3:
            source_image = torch.cat((source_image, source_image, source_image), 1)  # MNIST convert to 3 channel
        source_feature = encoder(source_image)
        source_output = classifier(source_feature)
        source_pred = source_output.data.max(1, keepdim=True)[1]
        source_correct += source_pred.eq(source_label.data.view_as(source_pred)).cpu().sum()

        # 2. Target input -> Target Classification
        target_image, target_label = target_data
        target_image, target_label = target_image.cuda(), target_label.cuda()
        if target_image.shape[1] != 3:
            target_image = torch.cat((target_image, target_image, target_image), 1)  # MNIST convert to 3 channel
        target_feature = encoder(target_image)
        target_output = classifier(target_feature)
        target_pred = target_output.data.max(1, keepdim=True)[1]
        target_correct += target_pred.eq(target_label.data.view_as(target_pred)).cpu().sum()

        if training_mode == 'dann':
            # 3. Combined input -> Domain Classificaion
            combined_image = torch.cat((source_image, target_image), 0)  # 64 = (S:32 + T:32)
            domain_source_labels = torch.zeros(source_label.shape[0]).type(torch.LongTensor)
            domain_target_labels = torch.ones(target_label.shape[0]).type(torch.LongTensor)
            domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).cuda()
            domain_feature = encoder(combined_image)
            domain_output = discriminator(domain_feature, alpha)
            domain_pred = domain_output.data.max(1, keepdim=True)[1]
            domain_correct += domain_pred.eq(domain_combined_label.data.view_as(domain_pred)).cpu().sum()

    min_val = min(len(source_test_loader.dataset), len(target_test_loader.dataset))
    if training_mode == 'dann':
        msg = "Test Results on DANN :"
        logging.info(msg)
        print(msg)
        msg = '\nSource Accuracy: {}/{} ({:.2f}%)\nTarget Accuracy: {}/{} ({:.2f}%)\nDomain Accuracy: {}/{} ({:.2f}%)\n'. \
            format(source_correct, min_val, 100. * source_correct.item() / len(source_test_loader.dataset),
                   target_correct, min_val, 100. * target_correct.item() / len(target_test_loader.dataset),
                   domain_correct, 2 * min_val, 100. * domain_correct.item() / (2 * min_val))
        logging.info(msg)
        print(msg)
    else:
        msg = "Test results on source_only :"
        logging.info(msg)
        print(msg)
        min_val = min(len(source_test_loader.dataset), len(target_test_loader.dataset))
        msg = '\nSource Accuracy: {}/{} ({:.2f}%)\nTarget Accuracy: {}/{} ({:.2f}%)\n'.format(source_correct, min_val,
                                                                                              100. * source_correct.item() / min_val,
                                                                                              target_correct, min_val,
                                                                                              100. * target_correct.item() / min_val)
        logging.info(msg)
        print(msg)
