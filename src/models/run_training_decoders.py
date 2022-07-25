import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.getcwd())
from model import EncoderExtractor
from src.helper.helpers import num_folds, __decoder_performances_file__, load_features, load_labels, __index_kfold__, __features__, split_train_test_valid, normalize_feature_data, __models_folder__, __best_encoder_file__, __decoder_file__, __result_files__
from sklearn.svm import SVC
import pickle
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

__C_values__ = [1e3, 1e4, 1e5]
__gamma_values__ = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

#Read data
image_labels = pd.read_csv(__index_kfold__)
flowers = image_labels['Flower'].values
counter = Counter(flowers)
class_word_names = np.array(sorted(list(counter.keys())))
images = load_features('image')


def run_training_decoders():
    predicted_targets = np.array([])
    actual_targets = np.array([])
    false_data    = []
    kfold = pd.read_csv(__index_kfold__)

    best_encoders = pd.read_csv(__best_encoder_file__)
    best_decoders = pd.DataFrame(columns=['kfold_file', 'fold', 'model_path', 'C', 'gamma', 'val_acc', 'test_acc'])
    extractor = EncoderExtractor()
    X = load_features(__features__)
    y = load_labels()
    
    i = 0
    val_accs, test_accs = [], [] 
    for fold in range(1, num_folds + 1):
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = split_train_test_valid(__features__, kfold, fold, X, y)

        test_label_fold = image_labels[image_labels[f'Fold_{fold}'] == 'Test']
        image_arr = images[test_label_fold.index]

        model_paths = []
        for feature in __features__:
            encoder_file = best_encoders[np.logical_and(best_encoders['fold'] == fold, best_encoders['feature'] == feature)].iloc[0]['model_file']
            model_paths.append(encoder_file)
        extractor.load_encoders(model_paths)

        X_train = extractor.extract(X_train)
        X_valid = extractor.extract(X_valid)
        X_test = extractor.extract(X_test)
        X_train, X_valid, X_test = normalize_feature_data("combine",  X_train, X_valid, X_test)

        best_val_acc, best_test_acc = 0.0, 0.0
        output_path = os.path.join(__models_folder__, __decoder_file__.format(fold))
        bad_C_gamma_values = [] 
        for C in __C_values__:
            for gamma in __gamma_values__:
                if [C, gamma] in bad_C_gamma_values:
                    continue
                decoder = SVC(gamma=gamma, C=C)
                decoder.fit(X_train, y_train)
                val_acc = np.mean(decoder.predict(X_valid) == y_valid)
                test_acc = np.mean(decoder.predict(X_test) == y_test)
                y_pred = decoder.predict(X_test)
                if val_acc < 0.98:
                    bad_C_gamma_values.append([C, gamma])
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_y_pred = y_pred
                    best_C, best_gamma = C, gamma
                    with open(output_path, "wb") as outfile:
                        pickle.dump(decoder, outfile)

        indices = np.where(best_y_pred != y_test)[0]
        for id in indices:
          image_data = (image_arr[id], class_word_names[y_test[id]], class_word_names[best_y_pred[id]]) #image arr, true, false
          false_data.append(image_data)

        predicted_targets = np.append(predicted_targets, best_y_pred)
        actual_targets = np.append(actual_targets, y_test)

        print("Fold {} - val_acc {} - test_acc {}".format(fold, best_val_acc, best_test_acc))
        val_accs.append(best_val_acc)
        test_accs.append(best_test_acc)
        best_decoders.loc[i] = [__index_kfold__, fold, output_path, best_C, best_gamma, best_val_acc, best_test_acc]
        i += 1

    print("End of 10-fold CV")
    print("Valid accuracy: {:.4f} +- {:.4f}".format(np.mean(val_accs), np.std(val_accs)))
    print("Test accuracy: {:.4f} +- {:.4f}".format(np.mean(test_accs), np.std(test_accs)))
    best_decoders.to_csv(__decoder_performances_file__, index=False)

    #Save result
    np.save(__result_files__['prediction'], predicted_targets)
    np.save(__result_files__['true'], actual_targets)
    np.save(__result_files__['false'], false_data)
    np.save(__result_files__['kfold_val_acc'],val_accs)
    np.save(__result_files__['kfold_test_acc'],test_accs)

if __name__ == "__main__":
    run_training_decoders()
    print("Please check the results in data/interim.")
        