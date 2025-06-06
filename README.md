# AI-Powered Credit Card Fraud Detection System

A machine learning-based system that detects fraudulent credit card transactions using Logistic Regression. Built with Python and Streamlit, this system provides a user-friendly web interface for real-time fraud detection.

## 🌟 Features

- Machine Learning-based fraud detection using Logistic Regression
- Interactive web interface built with Streamlit
- Real-time transaction prediction
- Data preprocessing and class balancing using undersampling
- High accuracy in detecting fraudulent transactions
- Simple CSV-based input system

## 🛠️ Technologies Used

- Python 3.x
- Scikit-learn (for machine learning)
- Pandas (for data manipulation)
- NumPy (for numerical operations)
- Streamlit (for web interface)

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ShreyaMahadev/AI-Powered-Credit-Card-Fraud-Detection-System.git
   cd AI-Powered-Credit-Card-Fraud-Detection-System
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the credit card dataset (`creditcard.csv`) and place it in the project directory.
   Note: Due to size limitations, the dataset is not included in the repository.

## 🚀 Usage

1. Make sure you have the `creditcard.csv` file in the project directory.

2. Run the Streamlit application:
   ```bash
   streamlit run creditcard.py
   ```

3. Once the application starts, it will open in your default web browser (typically at http://localhost:8501)

4. Enter the transaction features as comma-separated values when prompted

5. Click "Submit" to get the prediction result (Legitimate or Fraudulent)

## 📊 Model Details

The system utilizes a Logistic Regression model with the following characteristics:
- Handles imbalanced dataset using undersampling technique
- Features: Standard credit card transaction attributes
- Training-Testing split: 80-20 ratio
- Stratified sampling to maintain class distribution

## 📝 Input Format

The model expects input features in the following format:
- Comma-separated numerical values
- All features should be preprocessed similar to the training data
- The number of features should match the training dataset

## 🔐 Privacy Note

This project is designed for educational purposes. When using with real credit card data:
- Ensure proper data anonymization
- Follow relevant data protection regulations
- Do not store sensitive credit card information

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
