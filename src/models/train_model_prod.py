# Set seed value
seed_value = 23
import os
from tkinter import Y
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
from src.helper.helpers import __prod_decoder_file__, __prod_decoders_folder__, normalize_feature, prod_parameters, load_features, load_labels, __prod_encoder_file__, split_train_test_valid_prod, __prod_encoders_folder__, __features__, __prod_encoder_performance_file__
from src.models.model import get_training_model, CheckpointCallback, EncoderExtractor
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
import pickle
import warnings
warnings.filterwarnings('ignore')




def run_prod_encoder(feature, l2_rate, dropout):
	
	features = feature.split(',')
	"""run a single training
	"""
	print("==========================")
	if len(features) > 1:
		print("Training feature {} - l2_rate {} - dropout {}".format('_'.join(features), l2_rate, dropout))
		model_path = __prod_encoder_file__.format('_'.join(features), l2_rate, dropout)
	else:
		print("Training feature {} - l2_rate {} - dropout {}".format(features[0], l2_rate, dropout))
		model_path = __prod_encoder_file__.format(features[0], l2_rate, dropout)
	
	X = load_features(features)
	y = load_labels()


	# outdir = "models/Dataset_10FoldCV_indexed_models"
	if not os.path.exists(__prod_encoders_folder__):
		os.mkdir(__prod_encoders_folder__)


	for feature in features:
		if feature in ['vein', 'image']:
			max_epochs = 200
		else:
			max_epochs = 1000000

	(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = split_train_test_valid_prod(features, X, y)
	
	model_path = os.path.join(__prod_encoders_folder__, model_path)

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
	if len(features) > 1:
		db = pd.DataFrame([["all_features", model_path, l2_rate, dropout, val_acc, test_acc]],columns=["feature", "model_file", "l2_rate", "dropout", "val_acc","test_acc"])
	else:
		db = pd.DataFrame([[features[0], model_path, l2_rate, dropout, val_acc, test_acc]],columns=["feature", "model_file", "l2_rate", "dropout", "val_acc","test_acc"])

	return db, model_path

def run_prod_decoder(encoder_paths, decoder_path):
	#Call encoders
	extractor = EncoderExtractor()
	extractor.load_encoders(encoder_paths)

	#Data preparation
	X = load_features(__features__)
	y = load_labels()
	X = normalize_feature(X, __features__)
	X = extractor.extract(X)

	#Model
	__C_values__ = [1e3, 1e4, 1e5]
	__gamma_values__ = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
	svc_grid = {'C': __C_values__, 'gamma': __gamma_values__}
	decoder = SVC()
	decoder_search = RandomizedSearchCV(decoder, svc_grid, scoring = 'accuracy', cv=5, return_train_score=True, n_jobs=-1, n_iter=30, verbose=1)

	#Prediction
	decoder_search.fit(X,y)

	#Save decoders
	with open(decoder_path, "wb") as outfile:
		pickle.dump(decoder_search.best_estimator_, outfile)



if __name__ == "__main__":
	df = pd.DataFrame()
	encoder_paths = []
	for feature in __features__:
		l2_rate = prod_parameters[feature][0]
		dropout = prod_parameters[feature][1]
		db, model_path = run_prod_encoder(feature, l2_rate, dropout)
		df = pd.concat([df, db])
		encoder_paths.append(model_path)

	db.to_csv(__prod_encoder_performance_file__, index=False)

	decoder_path = os.path.join(__prod_decoders_folder__, __prod_decoder_file__)
	if not os.path.exists(decoder_path):
		os.mkdir(decoder_path)

	run_prod_decoder(encoder_paths, decoder_path)
        
	print("All encoders were saved in models/prod_models")