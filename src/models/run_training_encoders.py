import argparse
parser = argparse.ArgumentParser(description='Train encoders.')

parser.add_argument('expfile', type=str,
					help='Experiment file.')

# Set seed value
seed_value = 23
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)
import os
import sys
import numpy as np
import pandas as pd
# Bring your packages onto the path
sys.path.insert(0, os.getcwd())
from src.helper.helpers import load_features, load_labels, __model_file__, split_train_test_valid, __encoder_performances_file__, __models_folder__
from model import get_training_model, CheckpointCallback
import warnings
warnings.filterwarnings('ignore')




def run_training(features, kfold, fold, l2_rate, dropout):
	features = features.split(',')
	"""run a single training
	"""
	print("==========================")
	if len(features) > 1:
		print("Training feature {} - l2_rate {} - dropout {} - fold {}".format('_'.join(features), l2_rate, dropout, fold))
		model_path = __model_file__.format('_'.join(features), l2_rate, dropout, fold)
	else:
		print("Training feature {} - l2_rate {} - dropout {} - fold {}".format(features[0], l2_rate, dropout, fold))
		model_path = __model_file__.format(features[0], l2_rate, dropout, fold)
	
	X = load_features(features)
	y = load_labels()

	# outdir = "models/Dataset_10FoldCV_indexed_models"
	if not os.path.exists(__models_folder__):
		os.mkdir(__models_folder__)

	kfold = pd.read_csv(kfold)

	for feature in features:
		if feature in ['vein', 'image']:
			max_epochs = 200
		else:
			max_epochs = 1000000


	(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = split_train_test_valid(features, kfold, fold, X, y)
	
	model_path = os.path.join(__models_folder__, model_path)

	checkpoint = CheckpointCallback(verbose=False)
	# with strategy.scope(): # creating the model in the TPUStrategy scope means we will train the model on the TPU
	model = get_training_model(features, l2_rate=l2_rate, dropout=dropout)

	model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
				verbose=0,
				epochs=max_epochs,
				batch_size=128,
				callbacks=[checkpoint])
	model.save_weights(model_path)

	_, train_acc = model.evaluate(X_train, y_train, verbose=0)
	_, val_acc = model.evaluate(X_valid, y_valid, verbose=0)
	_, test_acc = model.evaluate(X_test, y_test, verbose=0)

	print("Train_time {:.4f}, train_acc {:.4f}, val_acc {:.4f}, test_acc {:.4f}".format(checkpoint.training_time, train_acc, val_acc, test_acc))

	return val_acc, test_acc

def run_training_encoders(experiment_file):
	"""run training from a csv file,
    save trained models to 'Dataset_10FoldCV_indexed_models' directory
	"""
	experiments = pd.read_csv(experiment_file)
	for exp_i in range(len(experiments)):
		kfold_file, folds, features, l2_rate, dropout = experiments.iloc[exp_i][['kfold_file','fold','feature','l2_rate','dropout']]
		val_acc, test_acc = run_training(features, kfold_file, folds, l2_rate, dropout)

		experiments.at[exp_i, 'val_acc'] = val_acc
		experiments.at[exp_i, 'test_acc'] = test_acc

		experiments.to_csv(__encoder_performances_file__, index=False)
	print("Complete training ", experiment_file)

if __name__ == "__main__":
	#Using data/external/train_encoders_setup_parameters.csv for expfile
	args = parser.parse_args()
	run_training_encoders(args.expfile)
	print("Please check the results in data/interim")