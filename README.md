# ML_algorithms_with_GUI
This Tkinter-based app allows users to upload CSV files, preprocess data, and apply machine learning algorithms like Random Forest, SVM, Decision Tree, Neural Network, KNN for classification, Linear Regression for regression, and K-Means for clustering. It displays metrics and visualizes  results, offering an interface for machine learning tasks

Main Features:
File Selection:

The user can select a CSV file containing the data using a file dialog.
The selected file is read into a pandas DataFrame, which is then processed by the app.
Data Preprocessing:

The data is cleaned (removing duplicates).
All categorical features are encoded using LabelEncoder, and missing values are imputed with the mean using SimpleImputer.
The data is scaled using MinMaxScaler.
Model Training and Evaluation:

The application provides options for Classification, Regression, and Clustering.
Classification: Algorithms like Random Forest, Support Vector Machine (SVM), Decision Trees, Neural Networks, and K-Nearest Neighbors (KNN) are implemented. Performance metrics like Accuracy, Precision, Recall, F1-Score, and Confusion Matrix are calculated and displayed.
Regression: A Linear Regression model is available, which outputs the intercept and coefficients.
Clustering: K-Means clustering is available, and it visualizes the results using scatter plots.
Result Display:

Results are displayed in a new window, showing the selected algorithm, performance metrics, and confusion matrix.
For clustering algorithms, a scatter plot is generated to visualize the clusters.
User Interface (Tkinter):

A welcome page with a button to proceed to the file selection page.
After selecting a file, the user is presented with options to choose the type of machine learning process they want to run (Classification, Regression, Clustering).
Each process type has a list of corresponding algorithms. When the user selects an algorithm, the application runs the model and displays the results.
Key Functions:
openFile(): Opens the selected file and returns a pandas DataFrame.
preProcessing(data): Cleans, encodes, imputes missing values, and scales the data.
split(data): Splits the data into features (X) and target variable (y), then into training and test sets.
Classification Algorithms (RF(), SVM(), Neural_Network_Algorithm(), etc.): These functions train the respective models and return evaluation metrics.
display_results(): Displays the results of the selected algorithm in a new Tkinter window.
plot_clusters(): Visualizes clustering results using a scatter plot.
start_process() and choose_process_type(): Handle the process type selection (Classification, Regression, or Clustering) and display available algorithms.
Tkinter UI: Contains pages for file selection, process type selection, and algorithm execution. The results are shown in separate windows after the user runs an algorithm.
How It Works:
The user starts by clicking the "Proceed" button in the main window, which opens the File Selection window. Here, they can choose a CSV file.
After selecting the file, the user selects the Process Type (Classification, Regression, or Clustering).
Depending on the process type, they are presented with a list of available algorithms.
The user clicks on an algorithm, and the model is trained and evaluated. The results (metrics or plots) are displayed in a new window.
Additional Notes:
Error Handling: The code includes error handling using messagebox.showerror() to ensure that the user selects a file before proceeding.
Data Imputation: The SimpleImputer with the "mean" strategy is used for missing data, which fills in missing values with the mean of the respective columns.
Data Encoding: All categorical features are transformed into numeric values using LabelEncoder, which makes them suitable for use with machine learning models.
Usage Instructions:
Run the application.
Choose a CSV file when prompted. The file should contain numerical and/or categorical data for machine learning.
Select the type of process (Classification, Regression, or Clustering).
Choose an algorithm and the app will process the data and display the results.
For classification, metrics like accuracy, precision, recall, and confusion matrix will be shown. For regression, coefficients and intercept will be displayed. For clustering, a plot of the clusters will be shown.
