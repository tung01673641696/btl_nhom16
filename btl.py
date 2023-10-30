import tkinter as tk
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

df = pd.read_csv('C:/Users/asus/Desktop/BTL/attachment_default.csv')
df['student'] = df['student'].map({'No': 0, 'Yes': 1})
df['default'] = df['default'].map({'No': 0, 'Yes': 1})
X = df.drop('default', axis=1)
y = df['default']

feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()  

scaler.fit(X)

X_train = pd.DataFrame(scaler.transform(X_train), columns=feature_names)
X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

log_model = LogisticRegression(solver='lbfgs')
log_model.fit(X_train, y_train)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_default(features):
    z = np.dot(log_model.coef_[0], features) + log_model.intercept_[0]
    probability = sigmoid(z)
    return probability

def clear_input_fields():
    student_entry.delete(0, tk.END)
    balance_entry.delete(0, tk.END)
    income_entry.delete(0, tk.END)
    result_text.delete("1.0", tk.END)

root = tk.Tk()
root.title("Dự đoán khả năng hoàn trả tín dụng")

student_label = tk.Label(root, text="Sinh viên(0 - Không phải, 1 - Phải):", bg='#F0F0F0', font=("Arial", 12))
balance_label = tk.Label(root, text="Số dư tài khoản:", bg='#F0F0F0', font=("Arial", 12))
income_label = tk.Label(root, text="Thu nhập của cá nhân:", bg='#F0F0F0', font=("Arial", 12))

student_var = tk.StringVar()
student_entry = tk.Entry(root, textvariable=student_var, font=("Arial", 12))
balance_var = tk.StringVar()
balance_entry = tk.Entry(root, textvariable=balance_var, font=("Arial", 12))
income_var = tk.StringVar()
income_entry = tk.Entry(root, textvariable=income_var, font=("Arial", 12))

def predict_default_gui():
    student = int(student_var.get())
    balance = float(balance_var.get())
    income = float(income_var.get())
    new_user_features = [student, balance, income]
    new_user_features = scaler.transform([new_user_features])
    default_probability = predict_default(new_user_features[0])
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, "Prediction: " + ("Khách hàng có thể hoàn trả tín dụng" if default_probability >= 0.5 else "Khách hàng không thể hoàn trả tín dụng"))

    student = int(student_var.get())
    balance = float(balance_var.get())
    income = float(income_var.get())
    new_user_features = [student, balance, income]
    new_user_features = scaler.transform([new_user_features])
    default_probability = predict_default(new_user_features[0])
    result_text.delete("1.0", tk.END)
    prediction_result = "Khách hàng có thể hoàn trả tín dụng" if default_probability >= 0.5 else "Khách hàng không thể hoàn trả tín dụng"
    result_text.insert(tk.END, "Prediction: " + prediction_result)

    y_pred = [1 if default_probability >= 0.5 else 0]
    accuracy = accuracy_score([y_test[0]], y_pred)
    precision = precision_score([y_test[0]], y_pred)
    recall = recall_score([y_test[0]], y_pred)
    f1 = f1_score([y_test[0]], y_pred)

    result_text.insert(tk.END, f"\nAccuracy: {accuracy:.2f}")
    result_text.insert(tk.END, f"\nPrecision: {precision:.2f}")
    result_text.insert(tk.END, f"\nRecall: {recall:.2f}")
    result_text.insert(tk.END, f"\nF1-score: {f1:.2f}")

predict_button = tk.Button(root, text="Dự đoán khả năng hoàn trả tín dụng", command=predict_default_gui, font=("Arial", 12), bg='#4CAF50', fg='white')

clear_button = tk.Button(root, text="Xóa dữ liệu đầu vào", command=clear_input_fields, font=("Arial", 12), bg='#F44336', fg='white')

result_text = tk.Text(root, width=40, height=7, font=("Arial", 12))

student_label.grid(row=0, column=0, sticky='w', padx=10, pady=10)
balance_label.grid(row=1, column=0, sticky='w', padx=10, pady=10)
income_label.grid(row=2, column=0, sticky='w', padx=10, pady=10)
student_entry.grid(row=0, column=1, padx=10, pady=10, sticky='w')
balance_entry.grid(row=1, column=1, padx=10, pady=10, sticky='w')
income_entry.grid(row=2, column=1, padx=10, pady=10, sticky='w')
predict_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)
clear_button.grid(row=4, column=0, columnspan=2, padx=10, pady=10)
result_text.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

knn = KNeighborsClassifier()
svm_model = SVC()

knn.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

def compare_models():
    result_text.delete("1.0", tk.END)
    accuracy_log = accuracy_score(y_test, log_model.predict(X_test))
    accuracy_knn = accuracy_score(y_test, knn.predict(X_test))
    accuracy_svm = accuracy_score(y_test, svm_model.predict(X_test))
    result_text.insert(tk.END, "Hồi Quy Logistic {:.6f}\n".format(accuracy_log))
    result_text.insert(tk.END, "K-Nearest Neighbors: {:.6f}\n".format(accuracy_knn))
    result_text.insert(tk.END, "SVM (Support Vector Machine): {:.6f}\n".format(accuracy_svm))

compare_button = tk.Button(root, text="So sánh mô hình", command=compare_models)
compare_button.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()