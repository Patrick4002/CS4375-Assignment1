import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import warnings
import sys

# py3.7+
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

class MLLibraryRegression:
    """
    Linear Regression using ML Libraries (SGDRegressor and LinearRegression)
    """
    def __init__(self):
        self.sgd_model = None
        self.lr_model = None
        self.scaler = None
        self.best_sgd_params = None
        self.sgd_results = []
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess the wine quality dataset (same as Part 1)
        """
        print("Loading Wine Quality dataset...")
        
        # Fetch dataset
        wine_quality = fetch_ucirepo(id=186)
        
        # Get features and target
        X = wine_quality.data.features
        y = wine_quality.data.targets
        
        print(f"Dataset shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
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
    
    def hyperparameter_tuning_sgd(self, X_train, y_train, X_val, y_val):
        """
        Perform hyperparameter tuning for SGDRegressor
        """
        print("\n" + "="*50)
        print("SGD REGRESSOR HYPERPARAMETER TUNING")
        print("="*50)
        
        # Define parameter grid for SGD
        learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2]
        max_iterations = [500, 1000, 2000, 3000, 5000]
        alpha_values = [0.0001, 0.001, 0.01, 0.1]  # Regularization strength
        
        best_val_mse = float('inf')
        best_params = {}
        best_model = None
        
        print("Tuning SGD Parameters...")
        print("LR = Learning Rate, Iter = Max Iterations, Alpha = Regularization, Train MSE, Val MSE")
        print("-" * 90)
        
        for lr in learning_rates:
            for max_iter in max_iterations:
                for alpha in alpha_values:
                    # Create and train SGD model
                    sgd = SGDRegressor(
                        learning_rate='constant',
                        eta0=lr,
                        max_iter=max_iter,
                        alpha=alpha,
                        random_state=42,
                        tol=1e-6
                    )
                    
                    sgd.fit(X_train, y_train)
                    
                    # Evaluate
                    train_pred = sgd.predict(X_train)
                    val_pred = sgd.predict(X_val)
                    
                    train_mse = mean_squared_error(y_train, train_pred)
                    val_mse = mean_squared_error(y_val, val_pred)
                    
                    self.sgd_results.append({
                        'learning_rate': lr,
                        'max_iter': max_iter,
                        'alpha': alpha,
                        'train_mse': train_mse,
                        'val_mse': val_mse,
                        'n_iter': sgd.n_iter_
                    })
                    
                    print(f"LR: {lr:6.3f} | Iter: {max_iter:4d} | Alpha: {alpha:6.4f} | Train MSE: {train_mse:8.4f} | Val MSE: {val_mse:8.4f}")
                    
                    # Update best model
                    if val_mse < best_val_mse:
                        best_val_mse = val_mse
                        best_params = {
                            'learning_rate': lr,
                            'max_iter': max_iter,
                            'alpha': alpha
                        }
                        best_model = sgd
        
        print(f"\nBest SGD Parameters: LR = {best_params['learning_rate']}, Max Iter = {best_params['max_iter']}, Alpha = {best_params['alpha']}")
        print(f"Best Validation MSE: {best_val_mse:.4f}")
        
        # Save results to log file
        results_df = pd.DataFrame(self.sgd_results)
        results_df.to_csv('sgd_hyperparameter_tuning_log.csv', index=False)
        print("SGD tuning results saved to 'sgd_hyperparameter_tuning_log.csv'")
        
        self.best_sgd_params = best_params
        return best_model, best_params, results_df
    
    def train_linear_regression(self, X_train, y_train):
        """
        Train standard Linear Regression model for comparison
        """
        print("\n" + "="*50)
        print("STANDARD LINEAR REGRESSION TRAINING")
        print("="*50)
        
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        print("Standard Linear Regression model trained successfully")
        return lr_model
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Evaluate the model and return comprehensive metrics
        """
        predictions = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Additional metrics
        explained_variance = r2_score(y_test, predictions, multioutput='variance_weighted')
        
        print(f"\n{model_name} Performance:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"Explained Variance: {explained_variance:.4f}")
        
        # Model-specific information
        if hasattr(model, 'coef_'):
            print(f"Number of features: {len(model.coef_)}")
            # Fix for intercept formatting
            if hasattr(model, 'intercept_'):
                if isinstance(model.intercept_, np.ndarray):
                    if model.intercept_.size == 1:
                        print(f"Intercept: {model.intercept_[0]:.4f}")
                    else:
                        print(f"Intercept: {model.intercept_}")
                else:
                    print(f"Intercept: {model.intercept_:.4f}")
        
        if hasattr(model, 'n_iter_'):
            print(f"Iterations to converge: {model.n_iter_}")
        
        return {
            'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
            'explained_variance': explained_variance, 'predictions': predictions
        }
    
    def compare_models(self, sgd_model, lr_model, X_test, y_test):
        """
        Compare SGD and standard Linear Regression models
        """
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        sgd_results = self.evaluate_model(sgd_model, X_test, y_test, "SGD Regressor")
        lr_results = self.evaluate_model(lr_model, X_test, y_test, "Standard Linear Regression")
        
        # Comparison summary
        print(f"\nComparison Summary:")
        print(f"{'Metric':<20} {'SGD':<12} {'LinearReg':<12} {'Difference':<12}")
        print("-" * 60)
        
        metrics = ['mse', 'rmse', 'mae', 'r2']
        for metric in metrics:
            sgd_val = sgd_results[metric]
            lr_val = lr_results[metric]
            diff = sgd_val - lr_val
            print(f"{metric.upper():<20} {sgd_val:<12.4f} {lr_val:<12.4f} {diff:<12.4f}")
        
        return sgd_results, lr_results
    
    def plot_results(self, sgd_model, lr_model, sgd_results, lr_results, y_test, results_df, feature_names):
        """
        Create comprehensive visualizations
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        plt.subplots_adjust(left=0.06, bottom=0.08, right=0.95, top=0.92, wspace=0.25, hspace=0.35)
        
        # Plot 1: SGD Hyperparameter tuning (Learning Rate vs Validation MSE)
        lr_grouped = results_df.groupby('learning_rate')['val_mse'].min()
        axes[0, 0].plot(lr_grouped.index, lr_grouped.values, 'o-', linewidth=2, markersize=6)
        axes[0, 0].set_title('SGD: Learning Rate vs Best Validation MSE', fontsize=12, pad=10)
        axes[0, 0].set_xlabel('Learning Rate', fontsize=10)
        axes[0, 0].set_ylabel('Best Validation MSE', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xscale('log')
        
        # Plot 2: Predictions vs Actual (SGD)
        axes[0, 1].scatter(sgd_results['predictions'], y_test, alpha=0.6, s=30, label='SGD', color='blue')
        axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Predicted Values', fontsize=10)
        axes[0, 1].set_ylabel('Actual Values', fontsize=10)
        axes[0, 1].set_title('SGD: Predictions vs Actual', fontsize=12, pad=10)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].text(0.05, 0.95, f'R² = {sgd_results["r2"]:.3f}', transform=axes[0, 1].transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontsize=10)
        
        # Plot 3: Predictions vs Actual (Linear Regression)
        axes[0, 2].scatter(lr_results['predictions'], y_test, alpha=0.6, s=30, label='LinearReg', color='green')
        axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 2].set_xlabel('Predicted Values', fontsize=10)
        axes[0, 2].set_ylabel('Actual Values', fontsize=10)
        axes[0, 2].set_title('Linear Regression: Predictions vs Actual', fontsize=12, pad=10)
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].text(0.05, 0.95, f'R² = {lr_results["r2"]:.3f}', transform=axes[0, 2].transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontsize=10)
        
        # Plot 4: Residuals comparison
        sgd_residuals = y_test - sgd_results['predictions']
        lr_residuals = y_test - lr_results['predictions']
        
        axes[1, 0].scatter(sgd_results['predictions'], sgd_residuals, alpha=0.6, s=30, color='blue', label='SGD')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Predicted Values', fontsize=10)
        axes[1, 0].set_ylabel('Residuals', fontsize=10)
        axes[1, 0].set_title('SGD: Residual Plot', fontsize=12, pad=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Feature importance comparison
        if hasattr(sgd_model, 'coef_') and hasattr(lr_model, 'coef_'):
            feature_comparison = pd.DataFrame({
                'feature': feature_names,
                'sgd_coef': sgd_model.coef_,
                'lr_coef': lr_model.coef_
            })
            
            x_pos = np.arange(len(feature_names))
            width = 0.35
            
            axes[1, 1].bar(x_pos - width/2, feature_comparison['sgd_coef'], width, label='SGD', alpha=0.8)
            axes[1, 1].bar(x_pos + width/2, feature_comparison['lr_coef'], width, label='Linear Reg', alpha=0.8)
            axes[1, 1].set_xlabel('Features', fontsize=10)
            axes[1, 1].set_ylabel('Coefficient Values', fontsize=10)
            axes[1, 1].set_title('Feature Coefficients Comparison', fontsize=12, pad=10)
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Model performance comparison
        metrics = ['MSE', 'RMSE', 'MAE', 'R²']
        sgd_values = [sgd_results['mse'], sgd_results['rmse'], sgd_results['mae'], sgd_results['r2']]
        lr_values = [lr_results['mse'], lr_results['rmse'], lr_results['mae'], lr_results['r2']]
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 2].bar(x_pos - width/2, sgd_values, width, label='SGD', alpha=0.8)
        axes[1, 2].bar(x_pos + width/2, lr_values, width, label='Linear Reg', alpha=0.8)
        axes[1, 2].set_xlabel('Metrics', fontsize=10)
        axes[1, 2].set_ylabel('Values', fontsize=10)
        axes[1, 2].set_title('Performance Metrics Comparison', fontsize=12, pad=10)
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].set_xticklabels(metrics)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add main title
        fig.suptitle('ML Library Linear Regression Analysis Results', fontsize=16, y=0.98)
        
        # Save plot
        plt.savefig('ml_library_regression_results.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

def main():
    """
    Main execution function for Part 2
    """
    print("="*60)
    print("LINEAR REGRESSION USING ML LIBRARIES")
    print("Wine Quality Dataset Analysis - Part 2")
    print("="*60)
    
    # Initialize the ML regression class
    ml_regression = MLLibraryRegression()
    
    # Step 1: Load and preprocess data
    X, y = ml_regression.load_and_preprocess_data()
    
    # Step 2: Feature scaling
    print("\n" + "="*50)
    print("FEATURE SCALING")
    print("="*50)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    print("Features standardized using StandardScaler")
    print("Scaled features - Mean:", np.mean(X_scaled.values, axis=0).round(4))
    print("Scaled features - Std:", np.std(X_scaled.values, axis=0).round(4))
    
    # Step 3: Split data
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
    
    # Step 4: Hyperparameter tuning for SGD
    best_sgd_model, best_sgd_params, sgd_results_df = ml_regression.hyperparameter_tuning_sgd(
        X_train, y_train, X_val, y_val
    )
    
    # Step 5: Train standard Linear Regression
    lr_model = ml_regression.train_linear_regression(X_train, y_train)
    
    # Step 6: Compare models on test set
    sgd_test_results, lr_test_results = ml_regression.compare_models(
        best_sgd_model, lr_model, X_test, y_test
    )
    
    # Step 7: Feature importance analysis
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    if hasattr(best_sgd_model, 'coef_') and hasattr(lr_model, 'coef_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'sgd_coef': best_sgd_model.coef_,
            'lr_coef': lr_model.coef_,
            'sgd_abs_coef': np.abs(best_sgd_model.coef_),
            'lr_abs_coef': np.abs(lr_model.coef_)
        }).sort_values('sgd_abs_coef', ascending=False)
        
        print("Feature Coefficients Comparison:")
        print(feature_importance)
    
    # Step 8: Create visualizations
    ml_regression.plot_results(
        best_sgd_model, lr_model, sgd_test_results, lr_test_results, 
        y_test, sgd_results_df, X.columns
    )
    
    # Step 9: Analysis and conclusions
    print("\n" + "="*50)
    print("ANALYSIS AND CONCLUSIONS")
    print("="*50)
    
    print(f"SGD Regressor Performance:")
    print(f"- Test MSE: {sgd_test_results['mse']:.4f}")
    print(f"- Test RMSE: {sgd_test_results['rmse']:.4f}")
    print(f"- Test R²: {sgd_test_results['r2']:.4f}")
    
    print(f"\nStandard Linear Regression Performance:")
    print(f"- Test MSE: {lr_test_results['mse']:.4f}")
    print(f"- Test RMSE: {lr_test_results['rmse']:.4f}")
    print(f"- Test R²: {lr_test_results['r2']:.4f}")
    
    print(f"\nBest SGD Parameters:")
    print(f"- Learning Rate: {best_sgd_params['learning_rate']}")
    print(f"- Max Iterations: {best_sgd_params['max_iter']}")
    print(f"- Regularization (Alpha): {best_sgd_params['alpha']}")
    
    # Answer the satisfaction question
    print(f"\n" + "="*50)
    print("SATISFACTION WITH THE ML LIBRARY SOLUTION")
    print("="*50)
    
    print("Are we satisfied that the ML library has found the best solution?")
    
    # Compare with theoretical optimum (Standard Linear Regression)
    mse_diff = abs(sgd_test_results['mse'] - lr_test_results['mse'])
    r2_diff = abs(sgd_test_results['r2'] - lr_test_results['r2'])
    
    print(f"\nComparison with Standard Linear Regression (theoretical optimum):")
    print(f"- MSE difference: {mse_diff:.6f}")
    print(f"- R² difference: {r2_diff:.6f}")
    
    if mse_diff < 0.01 and r2_diff < 0.01:
        print("✓ YES - SGD Regressor achieved results very close to the analytical solution.")
        print("✓ The hyperparameter tuning was comprehensive and found optimal parameters.")
        print("✓ The SGD algorithm successfully converged to the global optimum.")
    else:
        print("? PARTIALLY - There's a noticeable difference from the analytical solution.")
        print("- This could be due to the stochastic nature of SGD")
        print("- More extensive hyperparameter tuning might be needed")
        print("- Different regularization strategies could be explored")
    
    print(f"\nHow can we check if we found the best solution?")
    print(f"1. Compare with analytical solution (Standard Linear Regression)")
    print(f"2. Examine coefficient convergence and stability")
    print(f"3. Cross-validation with different random seeds")
    print(f"4. Learning curves to check for convergence")
    print(f"5. Grid search over wider parameter ranges")
    
    print(f"\nThe ML library provides:")
    print(f"- Robust optimization algorithms")
    print(f"- Automatic convergence checking")
    print(f"- Built-in regularization")
    print(f"- Efficient implementation")

if __name__ == "__main__":
    main()