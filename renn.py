import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle



# Step 1: Data Preprocessing

# Load the data in data frame
df = pd.read_csv('fraudTest.csv')
df = df.drop_duplicates()
dub = df.duplicated().sum()
print(f"Duplicates: {dub}")

# Separate the features and the target class
X = df.drop(columns=['Class'])
y = df['Class']

# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handling class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
print(f"Resampled class distribution:\n{pd.Series(y_resampled).value_counts()}")

# Splitting the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
print(f"Train size: {X_train.shape[0]}")
print(f"Test size: {X_test.shape[0]}")




# Step 2: Defining the External Rules for training
def apply_rules(X):
    rule_applied_data = []
    
    for row in X:
        try:
            amount_value = float(row[-2]) 
        except ValueError:
            print(f"Invalid value in row: {row}. Skipping this row.")
            rule_applied_data.append(0)  
            continue
        
        # Apply a rule based on the 'Amount' value
        if amount_value > 1000: 
            rule_applied_data.append(1)
        else:
            rule_applied_data.append(0)
    
    return rule_applied_data

# Apply the rules to the training and test data
rule_train = apply_rules(X_train)
rule_train_tensor = torch.tensor(rule_train, dtype=torch.float32).unsqueeze(1)
rule_test = apply_rules(X_test)
rule_test_tensor = torch.tensor(rule_test, dtype=torch.float32).unsqueeze(1)




# Step 3: ReNN Model Definition
class Renn(nn.Module):
    def __init__(self, input_size, rule_size, hidden_size, output_size):
        super(Renn, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.rule_layer = nn.Linear(rule_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size * 2, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, rule):
        x_out = self.input_layer(x)
        x_out = self.relu(x_out)

        rule_out = self.rule_layer(rule)
        rule_out = self.relu(rule_out)
        rule_out = rule_out.repeat(x_out.size(0) // rule_out.size(0), 1)

        combined_out = torch.cat((x_out, rule_out), dim=1)

        combined_out = self.hidden_layer(combined_out)
        combined_out = self.relu(combined_out)

        output = self.output_layer(combined_out)
        output = self.sigmoid(output)
        return output




# Step 5: Training the Model
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

input_size = X_train_tensor.shape[1]  
rule_size = 1
hidden_size = 120
output_size = 1 

model = Renn(input_size, rule_size, hidden_size, output_size)

# Defining the loss function and optimizer
# Adding weights to the loss function to handle class imbalance
class_weights = torch.tensor([1.0, 10.0])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



# Training Loop
epochs = 300
for epoch in range(epochs):
    model.train()
    outputs = model(X_train_tensor, rule_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")



# Model Evaluation
model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
with torch.no_grad():  
    val_outputs = model(X_test_tensor, rule_test_tensor)
    val_loss = criterion(val_outputs, y_test_tensor)
    print(f'Validation Loss: {val_loss.item():.4f}')
    
    # Convert outputs to binary classification using a threshold of 0.3
    predictions = (val_outputs >= 0.3).float()
    
    # Convert tensors to numpy for sklearn functions
    y_test_np = y_test_tensor.numpy()
    predictions_np = predictions.numpy()

    # Calculate accuracy
    accuracy = accuracy_score(y_test_np, predictions_np)
    print(f'Accuracy: {accuracy:.4f}')
    
    # Print classification report (precision, recall, F1-score)
    report = classification_report(y_test_np, predictions_np, target_names=['Class 0', 'Class 1'])
    print(report)




model_state = model.state_dict()

with open('trained_renn_model.pkl', 'wb') as f:
    pickle.dump(model_state, f)

print("Model has been saved to 'trained_renn_model.pkl'")

# Loading the Model (Example)
with open('trained_renn_model.pkl', 'rb') as f:
    saved_model_state = pickle.load(f)

# Rebuild the model architecture before loading
model = Renn(input_size, rule_size, hidden_size, output_size)
model.load_state_dict(saved_model_state)
print("Model has been loaded from 'trained_renn_model.pkl'")
