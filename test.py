import numpy as np
from sklearn.metrics import roc_auc_score

from torch import load

from utils.draw_metric_curve import drawROC, drawPR

import config


def predict():
    config.logger.info(
        '============================================ start test ============================================')

    config.logger.info(f'loading model: {config.model_file_path}')
    config.model.load_state_dict(load(config.model_file_path))
    config.logger.info('load model success !')

    # test
    config.model.eval()
    with config.torch.no_grad():
        test_pred = []
        test_true = []
        for idx, batch_data in enumerate(config.test_data_loader):
            img, label = batch_data[0].to(config.device), batch_data[1].numpy()
            pred = config.model(img).sigmoid().detach().cpu().numpy()
            test_pred.append(pred)
            test_true.append(label)
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_auc = [roc_auc_score(test_true[:, i], test_pred[:, i]) for i in range(test_true.shape[1])]
        test_auc_mean = np.mean(test_auc)
        config.logger.info(f'test auc: {test_auc}')
        config.logger.info(f'test average auc: {test_auc_mean}')

        drawROC(y_true=test_true, y_score=test_pred, n_classes=len(config.class_labels),
                class_labels=config.class_labels,
                save_path=config.curve_save_path)
        drawPR(y_true=test_true, y_score=test_pred, n_classes=len(config.class_labels),
               class_labels=config.class_labels,
               save_path=config.curve_save_path)

    config.logger.info("predict finished!\n")

    return test_auc


if __name__ == '__main__':
    predict()
