import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
test_idx = [0, 50, 100]

train_taget = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_taget)

print test_target
print clf.predict(test_data)


# import pydotplus 
# dot_data = tree.export_graphviz(clf, out_file=None) 
# graph = pydotplus.graph_from_dot_data(dot_data) 
# graph.write_pdf("iris.pdf") 





# from sklearn.externals.six import StringIO
# import pydotplus

# dot_data = StringIO();

# tree.export_graphviz(clf,
# 	out_file = dot_data,
# 	feature_names=iris.feature_names,  
#     class_names=iris.target_names,  
#     filled=True, rounded=True,  
#     impurity=False ) 

# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# graph.write_pdf("iris.pdf")




from IPython.display import Image  
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())  





# print train_taget
# print iris.target


# print iris.feature_names
# print iris.target_names