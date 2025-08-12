#!/usr/bin/env python3
"""
Create sample fraud detection dataset for testing bias analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_data(n_samples=10000, fraud_rate=0.05):
    """Create sample fraud detection dataset."""
    
    np.random.seed(42)
    
    # Date range
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    data = []
    
    for i in range(n_samples):
        # Generate transaction date
        random_days = np.random.randint(0, (end_date - start_date).days)
        transaction_date = start_date + timedelta(days=random_days)
        
        # Generate transaction time
        transaction_time = f"{np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}:{np.random.randint(0, 60):02d}"
        
        # Customer ID
        customer_id = np.random.randint(1000, 50000)
        
        # Customer Type (sensitive attribute 1)
        customer_type = np.random.choice(['Premium', 'Standard', 'Basic'], p=[0.3, 0.5, 0.2])
        
        # Transaction Type (sensitive attribute 2)
        transaction_type = np.random.choice(['Online', 'ATM', 'POS'], p=[0.6, 0.2, 0.2])
        
        # Transaction Amount (biased by customer type)
        if customer_type == 'Premium':
            amount_base = np.random.lognormal(6, 1.5)  # Higher amounts for premium
        elif customer_type == 'Standard':
            amount_base = np.random.lognormal(4, 1.2)
        else:  # Basic
            amount_base = np.random.lognormal(3, 1.0)  # Lower amounts for basic
        
        # Add bias to transaction type
        if transaction_type == 'ATM':
            amount_base *= 0.7  # ATM transactions tend to be smaller
        elif transaction_type == 'POS':
            amount_base *= 1.2  # POS transactions tend to be larger
        
        transaction_amount = round(max(10, amount_base), 2)
        
        # Generate fraud label with bias
        fraud_prob = fraud_rate
        
        # Introduce bias: Premium customers and Online transactions have lower fraud rates
        if customer_type == 'Premium':
            fraud_prob *= 0.3  # Premium customers have 70% lower fraud rate
        elif customer_type == 'Basic':
            fraud_prob *= 2.0  # Basic customers have 2x higher fraud rate
        
        if transaction_type == 'Online':
            fraud_prob *= 0.8  # Online transactions slightly lower fraud
        elif transaction_type == 'ATM':
            fraud_prob *= 1.5  # ATM transactions higher fraud
        
        # Higher amounts more likely to be fraud
        if transaction_amount > 1000:
            fraud_prob *= 2.0
        elif transaction_amount > 5000:
            fraud_prob *= 3.0
        
        is_fraudulent = np.random.random() < min(fraud_prob, 0.5)  # Cap at 50%
        
        data.append({
            'Transaction_Date': transaction_date.strftime('%Y-%m-%d'),
            'Transaction_Time': transaction_time,
            'Customer_ID': customer_id,
            'Customer_Type': customer_type,
            'Transaction_Type': transaction_type,
            'Transaction_Amount': transaction_amount,
            'Is_Fraudulent': int(is_fraudulent)
        })
    
    return pd.DataFrame(data)

def main():
    """Generate training and testing datasets."""
    print("Generating sample fraud detection datasets...")
    
    # Create training data (80%)
    train_data = create_sample_data(n_samples=8000, fraud_rate=0.05)
    train_data.to_csv('/workspace/sample_train.csv', index=False)
    print(f"Training data saved: sample_train.csv ({len(train_data)} samples)")
    print(f"Fraud rate in training: {train_data['Is_Fraudulent'].mean():.3f}")
    
    # Create testing data (20%)
    test_data = create_sample_data(n_samples=2000, fraud_rate=0.05)
    test_data.to_csv('/workspace/sample_test.csv', index=False)
    print(f"Testing data saved: sample_test.csv ({len(test_data)} samples)")
    print(f"Fraud rate in testing: {test_data['Is_Fraudulent'].mean():.3f}")
    
    # Print data distribution
    print("\nTraining data distribution:")
    print("Customer Type:")
    print(train_data['Customer_Type'].value_counts())
    print("\nTransaction Type:")
    print(train_data['Transaction_Type'].value_counts())
    
    print("\nFraud rates by Customer Type:")
    fraud_by_customer = train_data.groupby('Customer_Type')['Is_Fraudulent'].mean()
    print(fraud_by_customer)
    
    print("\nFraud rates by Transaction Type:")
    fraud_by_transaction = train_data.groupby('Transaction_Type')['Is_Fraudulent'].mean()
    print(fraud_by_transaction)

if __name__ == "__main__":
    main()