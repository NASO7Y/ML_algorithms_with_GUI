
# ML Algorithms with GUI

A repository showcasing various Machine Learning algorithms integrated with user-friendly Graphical User Interfaces (GUIs). This project is designed to simplify the understanding and application of ML concepts by providing visual and interactive tools.

## Features

- **Implemented Algorithms:**
  - Supervised Learning: Linear Regression, Logistic Regression, Decision Trees, Random Forest, etc.
  - Unsupervised Learning: K-Means Clustering, Hierarchical Clustering, PCA, etc.
  - Deep Learning (Optional): Integration with frameworks like TensorFlow or PyTorch.
- **Interactive GUI:**
  - Built using libraries such as `Tkinter`, `PyQt`, or others.
  - Easy input of dataset, parameter tuning, and visualization of results.
- **Visualization:**
  - Plots for training/validation performance.
  - Decision boundaries for classifiers.
  - Data distribution and clustering visuals.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/NASO7Y/ML_algorithms_with_GUI.git
   cd ML_algorithms_with_GUI
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the GUI application:
   ```bash
   python main.py
   ```

2. Load a dataset (CSV format recommended).
3. Select the desired ML algorithm from the GUI menu.
4. Adjust hyperparameters and visualize results.

## File Structure

- **`/algorithms`**: Contains implementations of various ML algorithms.
- **`/gui`**: Includes GUI code and layout files.
- **`/data`**: Sample datasets for testing.
- **`requirements.txt`**: Python dependencies for the project.
- **`main.py`**: Entry point for the application.

## Dependencies

- Python 3.x
- Required libraries (see `requirements.txt`):
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - GUI-specific library (`tkinter`/`pyqt5`)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature/bugfix.
3. Commit your changes and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or suggestions, feel free to open an issue or contact the repository owner.
