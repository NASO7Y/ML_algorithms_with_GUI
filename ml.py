from tkinter import *
from tkinter import messagebox, filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier

global file_path
file_path = None

def openFile():
    global file_path
    if file_path is not None:
        data = pd.read_csv(file_path)
        return data
    else:
        messagebox.showerror("Error", "No file selected.")
        return None

def preProcessing(data):
    
    data.drop_duplicates(inplace=True)
    
    le=LabelEncoder()
    for column in data.columns :
        data.loc[:,column]=le.fit_transform(data.loc[:,column])
        
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(data.iloc[:, :])
    data.iloc[:, :] = imputer.transform(data.iloc[:, :])  
    
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return data

def split(data):
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=24)
    return x_train, x_test, y_train, y_test

def RF(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=7)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, conf_matrix

def SVM(x_train, x_test, y_train, y_test):
    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(x_train, y_train)
    svm_pred = svm.predict(x_test)
    accuracy = accuracy_score(y_test, svm_pred)
    precision = precision_score(y_test, svm_pred, average='macro')
    recall = recall_score(y_test, svm_pred, average='macro')
    f1 = f1_score(y_test, svm_pred, average='macro')
    conf_matrix = confusion_matrix(y_test, svm_pred)
    return accuracy, precision, recall, f1, conf_matrix

def Neural_Network_Algorithm(x_train, x_test, y_train, y_test):
    NN = MLPClassifier(hidden_layer_sizes=1, activation="relu", learning_rate="constant")#0.1
    NN.fit(x_train, y_train)
    NN_1_pred = NN.predict(x_test)
    accuracy = accuracy_score(y_test, NN_1_pred)
    precision = precision_score(y_test, NN_1_pred, average='macro')
    recall = recall_score(y_test, NN_1_pred, average='macro')
    f1 = f1_score(y_test, NN_1_pred, average='macro')
    conf_matrix = confusion_matrix(y_test, NN_1_pred)
    return accuracy, precision, recall, f1, conf_matrix

def LinearRegressionAlgorithm(x_train, x_test, y_train, y_test):
    lin_reg = LinearRegression(fit_intercept=True)
    lin_reg.fit(x_train, y_train)
    predictions = lin_reg.predict(x_test)
    return lin_reg.intercept_, lin_reg.coef_

def Decision_Tree_Algorithm(x_train, x_test, y_train, y_test):
    model_1 = DecisionTreeClassifier(criterion="entropy")#gini 
    model_1.fit(x_train, y_train)
    model_1_pred = model_1.predict(x_test)
    accuracy = accuracy_score(y_test, model_1_pred)
    precision = precision_score(y_test, model_1_pred, average='macro')
    recall = recall_score(y_test, model_1_pred, average='macro')
    f1 = f1_score(y_test, model_1_pred, average='macro')
    conf_matrix = confusion_matrix(y_test, model_1_pred)
    return accuracy, precision, recall, f1, conf_matrix

def KNN(x_train,y_train,x_test,y_test):
    model_2 = KNeighborsClassifier(n_neighbors=3)
    model_2.fit(x_train, y_train)
    model_2_pred = model_2.predict(x_test)
    accuracy = accuracy_score(y_test, model_2_pred)
    precision = precision_score(y_test, model_2_pred, average='macro')
    recall = recall_score(y_test, model_2_pred, average='macro')
    f1 = f1_score(y_test, model_2_pred, average='macro')
    conf_matrix = confusion_matrix(y_test, model_2_pred)
    return accuracy, precision, recall, f1, conf_matrix

def KMeansAlgorithm(data):
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)
    labels = kmeans.labels_
    return labels

def display_results(algorithm, process_type, *results):
    result_window = Toplevel(window)
    result_window.title(f"{algorithm} Results")
    result_window.geometry("1000x1600")
    result_window.configure(bg="lightblue")

    if algorithm == "Linear Regression":
        intercept, coef = results
        result_text = f"Selected Algorithm: {algorithm}\n\n"
        result_text += f"Intercept: {intercept}\n"
        result_text += f"Coefficients: {coef}\n"
    else:
        if process_type == 'Classification':
            accuracy, precision, recall, f1, conf_matrix = results
            result_text = f"Selected Algorithm: {algorithm}\n\n"
            result_text += f"Accuracy: {accuracy:.2f}\n"
            result_text += f"Precision: {precision:.2f}\n"
            result_text += f"Recall: {recall:.2f}\n"
            result_text += f"F1 Score: {f1:.2f}\n\n"
            result_text += "Confusion Matrix:\n"
            result_text += f"{conf_matrix}\n"
        elif process_type == 'Clustering':
            labels, x_train = results
            plot_clusters(labels, x_train)
            return

    result_label = Label(result_window, text=result_text, font=("Arial", 20), justify=LEFT, bg="lightblue")
    result_label.pack(pady=20, padx=20)

def plot_clusters(labels, x_train):
    plt.figure(figsize=(8, 6))
    plt.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], c=labels, cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-Means Clustering')
    plt.show()

def choose_process_type(selection):
    global process_type
    process_type = selection

def show_algorithm_page(algorithm):
    algorithm_window = Toplevel(window)
    algorithm_window.title(f"{algorithm} Algorithms")
    algorithm_window.geometry("1000x600")
    algorithm_window.configure(bg="lightblue")

    process_label = Label(algorithm_window, text=f"Selected Process Type: {process_type}", font=("Arial", 30), bg="lightblue")
    process_label.pack(pady=15)

    if process_type == "Classification":
        algorithms = ['Random forest', 'SVM', 'Decision Tree', 'Neural Network','KNN']
    elif process_type == "Regression":
        algorithms = ['Linear Regression']
    elif process_type == "Clustering":
        algorithms = ['K-Means']

    for algo in algorithms:
        button = Button(algorithm_window, text=algo, command=lambda algo=algo: run_algorithm(algo, process_type), font=("Arial", 20), bg="purple", fg="white")
        button.pack(pady=15)

def run_algorithm(algorithm, process_type):
    df = openFile()
    if df is None:
        return
    df = preProcessing(data=df)
    x_train, x_test, y_train, y_test = split(data=df)

    if process_type == 'Classification':
        if algorithm == 'Random forest':
            results = RF(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        elif algorithm == 'SVM':
            results = SVM(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        elif algorithm == 'Decision Tree':
            results = Decision_Tree_Algorithm(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        elif algorithm == 'Neural Network':
            results = Neural_Network_Algorithm(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        
        elif algorithm =="KNN":
            display_results(algorithm)  
    
    elif process_type == 'Regression':
        if algorithm == "Linear Regression":
            results = LinearRegressionAlgorithm(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    elif process_type == 'Clustering':
        if algorithm == 'K-Means':
            labels = KMeansAlgorithm(data=x_train)
            results = (labels, x_train)
            
    display_results(algorithm, process_type, *results)

def start_process(selection):
    global process_type
    process_type = selection
    if process_type:
        algorithm_window = Toplevel(window)
        algorithm_window.title(f"{process_type} Algorithms")
        algorithm_window.geometry("1000x600")
        algorithm_window.configure(bg="lightblue")

        process_label = Label(algorithm_window, text=f"Selected Process Type: {process_type}", font=("Arial", 30), fg="black", bg="lightblue")
        process_label.pack(pady=15)

        if process_type == "Classification":
            algorithms = ['Random forest', 'SVM', 'Decision Tree', 'Neural Network',"KNN"]
        elif process_type == "Regression":
            algorithms = ['Linear Regression']
        elif process_type == "Clustering":
            algorithms = ['K-Means']
       
        for algo in algorithms:
            button = Button(algorithm_window, text=algo, command=lambda algo=algo: run_algorithm(algo, process_type), font=("Arial", 20), bg="purple", fg="white")
            button.pack(pady=15)

def choose_file():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        show_main_page()

def show_main_page():
    main_page = Toplevel(window)
    main_page.title("Process Type Selection")
    main_page.geometry("1000x600")
    main_page.configure(bg="lightblue")

    process_label = Label(main_page, text="Choose Process Type:", font=("Arial", 30), bg="lightblue")
    process_label.pack(pady=30)

    processes = ["Classification", "Regression", "Clustering"]
    for process in processes:
        button = Button(main_page, text=process, command=lambda process=process: start_process(process), font=("Arial", 20), bg="purple", fg="white")
        button.pack(pady=15)

def show_file_selection_page():
    file_selection_page = Toplevel(window)
    file_selection_page.title("File Selection")
    file_selection_page.geometry("1000x600")
    file_selection_page.configure(bg="lightblue")

    instruction_label = Label(file_selection_page, text="Please choose a CSV file to proceed.", font=("Arial", 20), bg="lightblue")
    instruction_label.pack(pady=30)

    choose_file_button = Button(file_selection_page, text="Choose File", command=choose_file, font=("Arial", 20), bg="purple", fg="white")
    choose_file_button.pack(pady=15)

# Main Tkinter Window
window = Tk()
window.title("Machine Learning Algorithms")
window.geometry("1000x600")
window.configure(bg="lightblue")

welcome_label = Label(window, text="Welcome to the Machine Learning Algorithm Application", font=("Arial", 30), bg="lightblue")
welcome_label.pack(pady=30)

proceed_button = Button(window, text="Proceed", command=show_file_selection_page, font=("Arial", 20), bg="purple", fg="white")
proceed_button.pack(pady=30)

window.mainloop()