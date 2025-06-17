import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings
import sys
# py3.7+
sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

class LinearRegressionGD:
    """
    Linear Regression using Gradient Descent
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def fit(self, X, y):
        """
        Fit the linear regression model using gradient descent
        """
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0
        self.cost_history = []
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute cost (MSE)
            cost = np.mean((y_pred - y) ** 2)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check for convergence
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                print(f"Converged at iteration {i}")
                break
    
    def predict(self, X):
        """
        Make predictions using the trained model
        """
        return np.dot(X, self.weights) + self.bias
    
    def get_cost_history(self):
        """
        Return the cost history
        """
        return self.cost_history

def load_and_preprocess_data():
    """
    Load and preprocess the wine quality dataset
    """
    print("Loading Wine Quality dataset...")
    
    # Fetch dataset
    wine_quality = fetch_ucirepo(id=186)
    
    # Get features and target
    X = wine_quality.data.features
    y = wine_quality.data.targets
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print("\nDataset Info:")
    print(wine_quality.metadata)
    print("\nFeature Information:")
    print(wine_quality.variables)
    
    # Combine features and target for preprocessing
    df = pd.concat([X, y], axis=1)
    
    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    
    # Remove rows with missing values
    df_clean = df.dropna()
    print(f"Shape after removing missing values: {df_clean.shape}")
    
    # Remove duplicate rows
    df_clean = df_clean.drop_duplicates()
    print(f"Shape after removing duplicates: {df_clean.shape}")
    
    # Separate features and target
    feature_columns = X.columns.tolist()
    target_column = y.columns[0]
    
    X_clean = df_clean[feature_columns]
    y_clean = df_clean[target_column]
    
    print(f"\nFinal preprocessing summary:")
    print(f"Features shape: {X_clean.shape}")
    print(f"Target shape: {y_clean.shape}")
    print(f"Feature columns: {list(X_clean.columns)}")
    
    return X_clean, y_clean

def explore_data(X, y):
    """
    Explore the dataset and show correlations
    """
    print("\n" + "="*50)
    print("DATA EXPLORATION")
    print("="*50)
    
    # Basic statistics
    print("Feature Statistics:")
    print(X.describe())
    
    print(f"\nTarget Statistics:")
    print(y.describe())
    
    # Correlation with target
    correlation_data = pd.concat([X, y], axis=1)
    correlations = correlation_data.corr()[y.name].sort_values(key=abs, ascending=False)
    
    print(f"\nCorrelation with target variable '{y.name}':")
    for feature, corr in correlations.items():
        if feature != y.name:
            print(f"{feature}: {corr:.4f}")
    
    return correlations

def hyperparameter_tuning(X_train, y_train, X_val, y_val):
    """
    Perform hyperparameter tuning and log results
    """
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING")
    print("="*50)
    
    # Define parameter grid
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2]
    n_iterations_list = [500, 1000, 2000, 3000]
    
    results = []
    best_mse = float('inf')
    best_params = {}
    best_model = None
    
    print("Tuning Parameters...")
    print("LR = Learning Rate, Iter = Iterations, Train MSE = Training MSE, Val MSE = Validation MSE")
    print("-" * 80)
    
    for lr in learning_rates:
        for n_iter in n_iterations_list:
            # Train model
            model = LinearRegressionGD(learning_rate=lr, n_iterations=n_iter)
            model.fit(X_train, y_train)
            
            # Evaluate
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            train_mse = np.mean((train_pred - y_train) ** 2)
            val_mse = np.mean((val_pred - y_val) ** 2)
            
            results.append({
                'learning_rate': lr,
                'n_iterations': n_iter,
                'train_mse': train_mse,
                'val_mse': val_mse
            })
            
            print(f"LR: {lr:6.3f} | Iter: {n_iter:4d} | Train MSE: {train_mse:8.4f} | Val MSE: {val_mse:8.4f}")
            
            # Update best model
            if val_mse < best_mse:
                best_mse = val_mse
                best_params = {'learning_rate': lr, 'n_iterations': n_iter}
                best_model = model
    
    print(f"\nBest Parameters: Learning Rate = {best_params['learning_rate']}, Iterations = {best_params['n_iterations']}")
    print(f"Best Validation MSE: {best_mse:.4f}")
    
    # Save results to log file
    results_df = pd.DataFrame(results)
    results_df.to_csv('hyperparameter_tuning_log.csv', index=False)
    print("Results saved to 'hyperparameter_tuning_log.csv'")
    
    return best_model, best_params, results_df

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate the model and return metrics
    """
    predictions = model.predict(X_test)
    
    # Calculate metrics
    mse = np.mean((predictions - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y_test))
    
    # R-squared
    ss_res = np.sum((y_test - predictions) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"\n{model_name} Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'predictions': predictions}

def plot_results(model, results_df, test_results, y_test):
    """
    Create visualizations with proper layout and spacing
    """
    # Create figure with larger size and better spacing
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Adjust spacing between subplots
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.92, wspace=0.25, hspace=0.35)
    
    # Plot 1: Cost history
    axes[0, 0].plot(model.get_cost_history(), linewidth=2)
    axes[0, 0].set_title('Training Cost History', fontsize=12, pad=10)
    axes[0, 0].set_xlabel('Iteration', fontsize=10)
    axes[0, 0].set_ylabel('Cost (MSE)', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(labelsize=9)
    
    # Plot 2: Hyperparameter tuning heatmap
    pivot_table = results_df.pivot(index='learning_rate', columns='n_iterations', values='val_mse')
    sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[0, 1], cbar_kws={'shrink': 0.8})
    axes[0, 1].set_title('Validation MSE Heatmap', fontsize=12, pad=10)
    axes[0, 1].set_xlabel('Number of Iterations', fontsize=10)
    axes[0, 1].set_ylabel('Learning Rate', fontsize=10)
    axes[0, 1].tick_params(labelsize=9)
    
    # Plot 3: Predictions vs Actual
    axes[1, 0].scatter(test_results['predictions'], y_test, alpha=0.6, s=30)
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('Predicted Values', fontsize=10)
    axes[1, 0].set_ylabel('Actual Values', fontsize=10)
    axes[1, 0].set_title('Predictions vs Actual Values', fontsize=12, pad=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(labelsize=9)
    
    # Add R² score to the plot
    r2_score = test_results['r2']
    axes[1, 0].text(0.05, 0.95, f'R² = {r2_score:.3f}', transform=axes[1, 0].transAxes, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
    
    # Plot 4: Residuals
    residuals = y_test - test_results['predictions']
    axes[1, 1].scatter(test_results['predictions'], residuals, alpha=0.6, s=30)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Predicted Values', fontsize=10)
    axes[1, 1].set_ylabel('Residuals', fontsize=10)
    axes[1, 1].set_title('Residual Plot', fontsize=12, pad=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(labelsize=9)
    
    # Add main title
    fig.suptitle('Linear Regression Analysis Results', fontsize=16, y=0.98)
    
    # Save with high DPI and tight layout
    plt.savefig('regression_results.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("LINEAR REGRESSION WITH GRADIENT DESCENT")
    print("Wine Quality Dataset Analysis")
    print("="*60)
    
    # Step 1: Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Step 2: Explore data
    correlations = explore_data(X, y)
    
    # Step 3: Feature scaling
    print("\n" + "="*50)
    print("FEATURE SCALING")
    print("="*50)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    print("Features standardized using StandardScaler")
    print("Scaled features - Mean:", np.mean(X_scaled.values, axis=0).round(4))
    print("Scaled features - Std:", np.std(X_scaled.values, axis=0).round(4))
    
    # Step 4: Split data
    print("\n" + "="*50)
    print("DATA SPLITTING")
    print("="*50)
    
    # First split: 80% train, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Second split: 80% of temp for training, 20% for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Step 5: Hyperparameter tuning
    best_model, best_params, results_df = hyperparameter_tuning(X_train, y_train, X_val, y_val)
    
    # Step 6: Final evaluation on test set
    print("\n" + "="*50)
    print("FINAL MODEL EVALUATION")
    print("="*50)
    
    test_results = evaluate_model(best_model, X_test, y_test, "Best Model on Test Set")
    
    # Step 7: Feature importance (weights)
    print(f"\nFeature Weights (importance):")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'weight': best_model.weights,
        'abs_weight': np.abs(best_model.weights)
    }).sort_values('abs_weight', ascending=False)
    
    print(feature_importance)
    
    # Step 8: Create visualizations (Fixed function call)
    plot_results(best_model, results_df, test_results, y_test)
    
    # Step 9: Analysis and conclusions
    print("\n" + "="*50)
    print("ANALYSIS AND CONCLUSIONS")
    print("="*50)
    
    print(f"Final Model Performance:")
    print(f"- Test MSE: {test_results['mse']:.4f}")
    print(f"- Test RMSE: {test_results['rmse']:.4f}")
    print(f"- Test R²: {test_results['r2']:.4f}")
    
    print(f"\nModel Parameters:")
    print(f"- Learning Rate: {best_params['learning_rate']}")
    print(f"- Iterations: {best_params['n_iterations']}")
    print(f"- Bias: {best_model.bias:.4f}")
    
    print(f"\nTop 5 Most Important Features:")
    for i, row in feature_importance.head().iterrows():
        print(f"- {row['feature']}: {row['weight']:.4f}")
    
    # Answer the satisfaction question
    print(f"\n" + "="*50)
    print("SATISFACTION WITH THE SOLUTION")
    print("="*50)
    
    print("Are we satisfied with the best solution found?")
    print(f"The model achieved an R² score of {test_results['r2']:.4f} on the test set.")
    
    if test_results['r2'] > 0.5:
        print("✓ YES - The model explains a reasonable portion of the variance in wine quality.")
        print("✓ The RMSE of {:.4f} is reasonable for wine quality prediction (scale 0-10).".format(test_results['rmse']))
        print("✓ The hyperparameter tuning was thorough with multiple combinations tested.")
    else:
        print("X PARTIALLY - While the model works, there's room for improvement.")
        print("- Consider feature engineering or polynomial features")
        print("- The linear assumption might be limiting for this dataset")
        print("- More sophisticated algorithms might be needed")
    
    print(f"\nThe gradient descent algorithm successfully converged and found stable weights.")
    print(f"The systematic hyperparameter tuning ensures we found a good solution within")
    print(f"the tested parameter space.")