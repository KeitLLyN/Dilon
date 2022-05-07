from utils.reader import Reader, Data
from utils.config import Config
from models.predictor import Predictor
import os
import numpy as np
from sklearn.metrics import auc, roc_curve, precision_score, recall_score
import matplotlib.pyplot as plt


test_path = 'datasets/test.txt'
test_anomaly_path = 'datasets/test_anomaly.txt'
path_normal_data = "datasets/vulnbank_train.txt"
path_anomaly_data = "datasets/vulnbank_anomaly.txt"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

config = Config()
#
data = Data(path_normal_data)
predictor = Predictor(config.checkpoints, config.std_factor, config.vocab)

generated_value = data.val_generator()
threshold = predictor.set_threshold(generated_value)
print(f"{threshold=}")

test_gen = data.test_generator()
valid_predictions, valid_loss = predictor.predict(test_gen)


print("-"*20)
print('Number of FP: ', np.sum(valid_predictions))
print('Number of samples: ', len(valid_predictions))
print('FP rate: {:.4f}'.format(np.sum(valid_predictions) / len(valid_predictions)))
print("-"*20)

pred_data = Data(path_anomaly_data, predict=True)
pred_gen = pred_data.predict_generator()
anomaly_pred, anomaly_loss = predictor.predict(pred_gen)

print('\nNumber of TP: ', np.sum(anomaly_pred))
print('Number of samples: ', len(anomaly_pred))
print('TP rate: {:.4f}'.format(np.sum(anomaly_pred) / len(anomaly_pred)))
print("-"*20)

y_true = np.concatenate(([0] * len(valid_predictions), [1] * len(anomaly_pred)), axis=0)
preds = np.concatenate((valid_predictions, anomaly_pred), axis=0)
loss_pred = np.concatenate((valid_loss, anomaly_loss), axis=0)

precision = precision_score(y_true, preds)
recall = recall_score(y_true, preds)
print('Precision: {:.4f}'.format(precision))
print('Recall: {:.4f}'.format(recall))

fpr, tpr, _ = roc_curve(y_true, loss_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()
