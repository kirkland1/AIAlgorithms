import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.preprocessing import KBinsDiscretizer
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Load and preprocess data
def preprocess_data(df, n_bins=3):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    
    columns_to_discretize = ['study_hours', 'sleep_hours', 'previous_score', 
                           'attendance_rate', 'performance_score']
    
    for col in columns_to_discretize:
        df[f'{col}_disc'] = discretizer.fit_transform(df[[col]]).astype(int)
    
    # Normalize stress levels to 0-3 range
    df['stress_level'] = df['stress_level'] - 2  # Shift 2-5 to 0-3
    
    # Ensure health_status is integer
    df['health_status'] = df['health_status'].astype(int)
    
    return df

# Load data
df = pd.read_csv('student_performance.csv')
df_processed = preprocess_data(df)

# Get number of unique values for each variable
n_study = df_processed['study_hours_disc'].nunique()
n_sleep = df_processed['sleep_hours_disc'].nunique()
n_stress = df_processed['stress_level'].nunique()
n_prev = df_processed['previous_score_disc'].nunique()
n_attend = df_processed['attendance_rate_disc'].nunique()
n_health = df_processed['health_status'].nunique()
n_performance = df_processed['performance_score_disc'].nunique()

print("Number of unique values in each variable:")
print(f"Study hours: {n_study}")
print(f"Sleep hours: {n_sleep}")
print(f"Stress levels: {n_stress}")
print(f"Previous score: {n_prev}")
print(f"Attendance rate: {n_attend}")
print(f"Health status: {n_health}")
print(f"Performance score: {n_performance}")

# Define Bayesian Network structure
model = DiscreteBayesianNetwork([
    ('study_hours_disc', 'performance_score_disc'),
    ('sleep_hours_disc', 'performance_score_disc'),
    ('stress_level', 'performance_score_disc'),
    ('previous_score_disc', 'performance_score_disc'),
    ('attendance_rate_disc', 'performance_score_disc'),
    ('health_status', 'performance_score_disc')
])

# Calculate value counts for each variable
study_counts = df_processed['study_hours_disc'].value_counts(normalize=True).sort_index()
sleep_counts = df_processed['sleep_hours_disc'].value_counts(normalize=True).sort_index()
stress_counts = df_processed['stress_level'].value_counts(normalize=True).sort_index()
prev_score_counts = df_processed['previous_score_disc'].value_counts(normalize=True).sort_index()
attendance_counts = df_processed['attendance_rate_disc'].value_counts(normalize=True).sort_index()
health_counts = df_processed['health_status'].value_counts(normalize=True).sort_index()

# Create CPDs for each variable with correct shapes
cpd_study = TabularCPD(
    variable='study_hours_disc', 
    variable_card=n_study,
    values=study_counts.values.reshape(n_study, 1)
)

cpd_sleep = TabularCPD(
    variable='sleep_hours_disc', 
    variable_card=n_sleep,
    values=sleep_counts.values.reshape(n_sleep, 1)
)

cpd_stress = TabularCPD(
    variable='stress_level', 
    variable_card=n_stress,
    values=stress_counts.values.reshape(n_stress, 1)
)

cpd_prev_score = TabularCPD(
    variable='previous_score_disc', 
    variable_card=n_prev,
    values=prev_score_counts.values.reshape(n_prev, 1)
)

cpd_attendance = TabularCPD(
    variable='attendance_rate_disc', 
    variable_card=n_attend,
    values=attendance_counts.values.reshape(n_attend, 1)
)

cpd_health = TabularCPD(
    variable='health_status', 
    variable_card=n_health,
    values=health_counts.values.reshape(n_health, 1)
)

# Calculate conditional probabilities for performance_score
def calculate_performance_cpd(data):
    # Print unique values in each column for debugging
    print("\nUnique values in each column:")
    for col in ['study_hours_disc', 'sleep_hours_disc', 'stress_level', 
                'previous_score_disc', 'attendance_rate_disc', 'health_status', 
                'performance_score_disc']:
        print(f"{col}: {sorted(data[col].unique())}")
    
    # Create a dictionary to store the counts for each combination
    counts = {}
    
    # Initialize counts for all possible combinations
    for study in range(n_study):
        for sleep in range(n_sleep):
            for stress in range(n_stress):
                for prev in range(n_prev):
                    for attend in range(n_attend):
                        for health in range(n_health):
                            key = (study, sleep, stress, prev, attend, health)
                            counts[key] = np.zeros(n_performance)
    
    # Count occurrences of each performance level for each combination
    for idx, row in data.iterrows():
        # Ensure all values are integers and within valid ranges
        try:
            study = int(row['study_hours_disc'])
            sleep = int(row['sleep_hours_disc'])
            stress = int(row['stress_level'])
            prev = int(row['previous_score_disc'])
            attend = int(row['attendance_rate_disc'])
            health = int(row['health_status'])
            perf = int(row['performance_score_disc'])
            
            # Validate ranges
            if not (0 <= study < n_study and 
                    0 <= sleep < n_sleep and 
                    0 <= stress < n_stress and 
                    0 <= prev < n_prev and 
                    0 <= attend < n_attend and 
                    0 <= health < n_health and 
                    0 <= perf < n_performance):
                print(f"\nWarning: Invalid value found in row {idx}:")
                print(f"study: {study}, sleep: {sleep}, stress: {stress}, "
                      f"prev: {prev}, attend: {attend}, health: {health}, "
                      f"perf: {perf}")
                continue
                
            key = (study, sleep, stress, prev, attend, health)
            counts[key][perf] += 1
        except Exception as e:
            print(f"\nError processing row {idx}: {str(e)}")
            print(f"Row data: {row}")
            continue
    
    # Convert counts to probabilities
    cpd_values = np.zeros((n_performance, n_study * n_sleep * n_stress * n_prev * n_attend * n_health))
    
    for idx, key in enumerate(counts):
        total = counts[key].sum()
        if total > 0:
            cpd_values[:, idx] = counts[key] / total
        else:
            # If no data for this combination, use a default distribution
            # that favors higher performance for better conditions
            study, sleep, stress, prev, attend, health = key
            # Calculate a score based on conditions
            score = (study + sleep + (n_stress - 1 - stress) + prev + attend + health) / 6.0
            # Assign higher probabilities to higher performance levels based on the score
            cpd_values[0, idx] = max(0.1, 1 - score)
            cpd_values[1, idx] = 0.3
            cpd_values[2, idx] = max(0.1, score - 0.3)
            # Normalize
            cpd_values[:, idx] = cpd_values[:, idx] / cpd_values[:, idx].sum()
    
    return TabularCPD(
        variable='performance_score_disc',
        variable_card=n_performance,
        values=cpd_values,
        evidence=['study_hours_disc', 'sleep_hours_disc', 'stress_level', 
                 'previous_score_disc', 'attendance_rate_disc', 'health_status'],
        evidence_card=[n_study, n_sleep, n_stress, n_prev, n_attend, n_health]
    )

# Add CPDs to model
model.add_cpds(cpd_study, cpd_sleep, cpd_stress, cpd_prev_score, 
               cpd_attendance, cpd_health, calculate_performance_cpd(df_processed))

# Verify model
print("\nChecking model...")
print("Model is valid:", model.check_model())

# Make predictions
def predict_performance(model, student_data):
    evidence_df = pd.DataFrame([student_data])
    
    # Print the evidence being used
    print("\nEvidence being used for prediction:")
    print(evidence_df)
    
    # Get the inference object
    from pgmpy.inference import VariableElimination
    infer = VariableElimination(model)
    
    # Get the probability distribution for performance_score
    query = infer.query(variables=['performance_score_disc'], 
                       evidence=student_data)
    
    # Print the probability distribution
    print("\nProbability distribution for performance_score:")
    print(query)
    
    # Get the most likely value
    prediction = query.values.argmax()
    print(f"\nMost likely performance level: {prediction}")
    
    return pd.DataFrame({'performance_score_disc': [prediction]})

# Example prediction
print("\nExample Prediction:")
test_student = {
    'study_hours_disc': 1,
    'sleep_hours_disc': 1,
    'stress_level': 2,
    'previous_score_disc': 1,
    'attendance_rate_disc': 1,
    'health_status': 1
}

try:
    prediction = predict_performance(model, test_student)
    print(f"Predicted performance level: {prediction['performance_score_disc'].iloc[0]}")
except Exception as e:
    print(f"Error making prediction: {str(e)}")

# Print model structure
print("\nModel Structure:")
print("Nodes:", model.nodes())
print("Edges:", model.edges())

# Test different scenarios
print("\n=== Testing Different Scenarios ===")

# 1. Poor Conditions
print("\nTest Case 1: Poor Conditions Student:")
poor_conditions = {
    'study_hours_disc': 0,      # Low study hours
    'sleep_hours_disc': 0,      # Low sleep hours
    'stress_level': 3,          # High stress
    'previous_score_disc': 0,   # Low previous score
    'attendance_rate_disc': 0,  # Low attendance
    'health_status': 0          # Poor health
}

try:    
    prediction = predict_performance(model, poor_conditions)
    print("Input conditions:")
    for key, value in poor_conditions.items():
        print(f"{key}: {value}")
    print(f"Predicted performance level: {prediction['performance_score_disc'].iloc[0]}")   
except Exception as e:
    print(f"Error making prediction: {str(e)}")

# 2. Mixed Conditions
print("\nTest Case 2: Mixed Conditions Student:")
mixed_conditions = {
    'study_hours_disc': 2,      # High study hours
    'sleep_hours_disc': 0,      # Low sleep hours
    'stress_level': 2,          # Moderate-high stress
    'previous_score_disc': 1,   # Average previous score
    'attendance_rate_disc': 2,  # High attendance
    'health_status': 0          # Poor health
}

try:    
    prediction = predict_performance(model, mixed_conditions)
    print("Input conditions:")
    for key, value in mixed_conditions.items():
        print(f"{key}: {value}")
    print(f"Predicted performance level: {prediction['performance_score_disc'].iloc[0]}")   
except Exception as e:
    print(f"Error making prediction: {str(e)}")

# 3. Stress Impact Test
print("\nTest Case 3: High Achiever with Stress:")
stress_test = {
    'study_hours_disc': 2,      # High study hours
    'sleep_hours_disc': 2,      # High sleep hours
    'stress_level': 3,          # High stress
    'previous_score_disc': 2,   # High previous score
    'attendance_rate_disc': 2,  # High attendance
    'health_status': 1          # Good health
}

try:    
    prediction = predict_performance(model, stress_test)
    print("Input conditions:")
    for key, value in stress_test.items():
        print(f"{key}: {value}")
    print(f"Predicted performance level: {prediction['performance_score_disc'].iloc[0]}")   
except Exception as e:
    print(f"Error making prediction: {str(e)}")