import pickle
# model = emodel
# filename = 'finalized_model.sav'
# pickle.dump(model, open(filename, 'wb'))
# load the model from disk
loaded_model = pickle.load(open('C:\Users\DELL\COV_Project\models\finalized_model.sav', 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)