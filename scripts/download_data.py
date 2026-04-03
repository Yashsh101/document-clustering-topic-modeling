"""Download sample dataset for demonstration."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def create_sample_documents():
    """Create sample documents for testing."""
    sample_texts = [
        "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
        "Deep learning uses neural networks with multiple layers to process data.",
        "Python is a popular programming language for data science and machine learning.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information from images.",
        "Data preprocessing is crucial for building effective machine learning models.",
        "Supervised learning requires labeled data for training models.",
        "Unsupervised learning discovers patterns in unlabeled data.",
        "Clustering groups similar data points together without labeled training data.",
        "Classification assigns data points to predefined categories or classes.",
        "Regression models predict continuous numerical values based on input features.",
        "Neural networks are inspired by biological neurons in the brain.",
        "Convolutional neural networks are widely used in computer vision tasks.",
        "Recurrent neural networks process sequential data like time series.",
        "Transformers have revolutionized natural language processing with attention mechanisms.",
        "TensorFlow and PyTorch are popular frameworks for deep learning.",
        "Scikit-learn provides simple tools for machine learning in Python.",
        "Pandas is essential for data manipulation and analysis in Python.",
        "NumPy enables numerical computing with multi-dimensional arrays.",
        "Matplotlib is used for creating visualizations and plots.",
        "Feature engineering is the art of creating meaningful features from raw data.",
        "Cross-validation helps assess model generalization performance.",
        "Hyperparameter tuning optimizes model performance through systematic search.",
        "Gradient descent is a fundamental optimization algorithm for training models.",
        "Backpropagation efficiently computes gradients in neural networks.",
        "Overfitting occurs when a model learns noise instead of true patterns.",
        "Regularization techniques prevent overfitting by penalizing model complexity.",
        "Batch normalization stabilizes and accelerates neural network training.",
        "Dropout is a regularization technique that randomly disables neurons.",
        "Activation functions introduce non-linearity into neural networks.",
        "ReLU is a simple and effective activation function for hidden layers.",
        "Sigmoid squashes values to the range between 0 and 1.",
        "Softmax is used for multi-class classification problems.",
        "Loss functions measure the difference between predicted and actual values.",
        "Mean squared error is common for regression problems.",
        "Cross-entropy loss is standard for classification tasks.",
        "Metrics like accuracy and F1-score evaluate model performance.",
        "Precision and recall are important for imbalanced datasets.",
        "Confusion matrices show classification results across different classes.",
        "ROC curves visualize classification performance at different thresholds.",
    ]

    # Create sample data directory
    sample_dir = Path("data/sample")
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Write sample files
    for i, text in enumerate(sample_texts):
        with open(sample_dir / f"sample_{i:02d}.txt", "w") as f:
            f.write(text)

    logger.info(f"Created {len(sample_texts)} sample documents in {sample_dir}")


if __name__ == "__main__":
    create_sample_documents()
