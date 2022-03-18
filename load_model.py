import pickle

def load_bayes_model():
    with open("./model/bayes_model.pkl", "rb") as f:
        cv_model, tf_model = pickle.load(f)
    return cv_model, tf_model

def load_svm_model():
    with open("./model/svm_model.pkl", "rb") as f:
        cv_model = pickle.load(f)
    return cv_model

def load_linear_model():
    with open("./model/linear_model.pkl", "rb") as f:
        cv_model, tf_model = pickle.load(f)
    return cv_model, tf_model

def load_vectorizes():
    with open("./model/vectorizes.pkl", "rb") as f:
        cv_model, tf_model = pickle.load(f)
    return cv_model, tf_model