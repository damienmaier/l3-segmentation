from data_loading import get_test_set
from model_training import train_best_model

X_train, Y_train = get_test_set()

train_best_model(X_train, Y_train)