import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
#قراءة البيانات
data = pd.read_csv('/Users/nawaf/Desktop/datasets/creditcard.csv')
print(data.head())
#تقسيم البيانات
x = data.drop('Class', axis = 1 ) #حذف كل الاعمدة الميزات ماعدا عمود (class)
y = data['Class'] #الهدف =  الاجابة : احتيال او لا
#تقسيم الى تدريب واختبار
X_train, X_test , Y_train , Y_test = train_test_split(
x, y , test_size= 0.3, random_state = 42 )
#تدريب نموذج Decision tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, Y_train)
#توقع على بيانات جديدة
dt_predictions = dt_model.predict(X_test)
#تقييم الاجابات والتوقع ودقتها
print(confusion_matrix(Y_test, dt_predictions))
print(classification_report(Y_test, dt_predictions))
#بناء نموذج SVM
# بناء نموذج SVM
from sklearn.svm import SVC

svm_model = SVC(kernel='linear')      # نستخدم kernel خطي
svm_model.fit(X_train, Y_train)       # تدريب النموذج على بيانات التدريب

# التنبؤ على بيانات الاختبار
svm_predictions = svm_model.predict(X_test)

# تقييم أداء النموذج
from sklearn.metrics import classification_report, confusion_matrix

print("\nنتائج SVM:")
print(confusion_matrix(Y_test, svm_predictions))        # كم مرة صح وكم مرة غلط
print(classification_report(Y_test, svm_predictions))   # دقة وتذكّر وتقييم عام
