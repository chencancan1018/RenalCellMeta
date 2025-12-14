import sklearn
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

class ModelSelect():
    def __init__(self, modelname=None):
        self._modelname = modelname
    def dismatch(self, *args, **kwgs):
#     def dismatch(self):
        try:
            getattr(self, 'select_{}'.format(self._modelname))
        except AttributeError as error:
            self._modelname = ""
        finally:
            pass
        return getattr(self, 'select_{}'.format(self._modelname))(*args, **kwgs)
    def select_svm(self):
        return SVC(gamma='auto', probability=True)
    def select_bayes(self):
        return GaussianNB()
    def select_decision_tree(self):
        return DecisionTreeClassifier(max_depth=80, min_samples_split=10, min_samples_leaf=10)
    def select_random_forest(self):
        return RandomForestClassifier(n_estimators=1000, max_depth=80, min_samples_split=10, random_state=0)
    def select_neutral_network(self):
        return MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(40, 10), random_state=1)
    def select_xgboost(self):
        return XGBClassifier(max_depth=5, learning_rate=1, n_estimators=10)

if __name__ == '__main__':
    ml = MachineLearning('svm')
    model = ml.dismatch()
    print(model)