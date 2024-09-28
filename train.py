import numpy as np
from tqdm import tqdm

from sklearn.metrics import roc_auc_score

from torch import save

from utils.draw_metric_curve import draw

import config


def set_all_seeds(seed):
    """
    init random seeds
    :param seed: seed number
    :return: No
    """
    config.torch.manual_seed(seed)
    np.random.seed(seed)
    config.torch.cuda.manual_seed_all(seed)
    config.torch.backends.cudnn.deterministic = True
    config.torch.backends.cudnn.benchmark = False


def train():
    set_all_seeds(123)

    # print info
    config.logger.info(
        '******************************************** train info ********************************************')
    config.logger.info(f'training dataset have {len(config.train_data)} images.')
    config.logger.info(
        f'Percentage of positive samples in {len(config.class_labels)} categories in the training dataset: ')
    config.logger.info(f"{config.class_labels}")
    config.logger.info(f"{np.round(np.sum(config.train_data.labels, axis=0) / len(config.train_data), 4).tolist()}")
    config.logger.info(f'validation dataset have {len(config.valid_data)} images.')
    config.logger.info(f'testing dataset have {len(config.test_data)} images.')
    config.logger.info(
        f'dataset:{config.datasets}, model:{config.model_name}, batch_size:{config.batch_size}, lr:{config.lr}, device:{config.device}')
    config.logger.info(f'optimizer:{config.optimizer}, loss_function:{config.loss_func}')
    config.logger.info(
        '============================================ start train ============================================')

    # save loss and auc
    average_train_loss = []
    batch_train_loss = []
    valid_auc = []

    # split epoch to small batches
    iter_batch_num = 100
    epoch_iter_batch_num = len(config.train_data_loader) // iter_batch_num

    with tqdm(total=config.epochs * epoch_iter_batch_num, desc='Train', leave=True, unit='img',
              unit_scale=True) as pbar:

        # count the number of iterations
        n = 1
        for epoch in range(config.epochs):
            config.logger.info(
                f'=================================== epoch: {epoch} ===================================')

            # save batch loss
            batch_loss = 0.
            for idx, batch_data in enumerate(config.train_data_loader):
                # train
                config.model.train()

                img, label = batch_data[0].to(config.device), batch_data[1].to(config.device)
                pred = config.model(img).sigmoid()
                loss = config.loss_func(pred, label)
                config.optimizer.zero_grad()
                loss.backward()
                config.optimizer.step()
                batch_loss += np.asarray(loss.detach().cpu())

                if idx % iter_batch_num == iter_batch_num - 1:
                    config.logger.info(f'BatchId: {idx + 1}')
                    b_loss = batch_loss / iter_batch_num
                    batch_train_loss.append(b_loss)
                    avg_loss = np.mean(batch_train_loss)
                    average_train_loss.append(avg_loss)
                    config.logger.info(f'\t--> batch loss: {b_loss}, average loss: {avg_loss}')

                    # validation
                    config.model.eval()
                    with config.torch.no_grad():
                        test_pred = []
                        test_true = []
                        for _, batch_data in enumerate(config.valid_data_loader):
                            img, label = batch_data[0].to(config.device), batch_data[1]
                            y_pred = config.model(img).sigmoid()
                            test_pred.append(y_pred.detach().cpu().numpy())
                            test_true.append(label.numpy())
                        test_true = np.concatenate(test_true)
                        test_pred = np.concatenate(test_pred)
                        val_auc = [roc_auc_score(test_true[:, i], test_pred[:, i]) for i in range(test_true.shape[1])]
                        val_auc_mean = np.mean(val_auc)
                        valid_auc.append(val_auc_mean)
                        config.logger.info(f'\t--> valid auc: {val_auc}')
                        config.logger.info(
                            f'\t--> average valid auc: {val_auc_mean}, best average valid auc: {max(valid_auc)}')

                        # save better model
                        if val_auc_mean >= max(valid_auc):
                            # save model parameters
                            save(config.model.state_dict(), config.model_file_path)
                            config.logger.info('save best model successfully!')

                    # draw and save: loss and auc curve image
                    draw(n, [batch_train_loss, average_train_loss], ['batch loss', 'average loss'],
                         'train', 'epoch', 'value', ['red', 'green'], config.curve_save_path)
                    draw(n, [valid_auc], ['validation auc'], 'validation', 'epoch', 'auc',
                         ['green'], config.curve_save_path)

                    # save loss and auc data
                    np.savetxt(f"{config.curve_save_path}/train_loss_data.txt",
                               np.transpose([average_train_loss, batch_train_loss], (1, 0)), '%.4f')
                    np.savetxt(f"{config.curve_save_path}/validation_auc_data.txt", valid_auc, '%.4f')

                    n += 1
                    batch_loss = 0.
                    pbar.update()

                # discard incomplete batches
                if idx >= iter_batch_num * epoch_iter_batch_num:
                    break

            config.scheduler.step()

    config.logger.info("train finished!\n")


if __name__ == '__main__':
    train()
