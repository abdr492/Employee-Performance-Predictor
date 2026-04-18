import pandas as pd
import numpy as np

np.random.seed(42)

def generate_employee_data(n=1000):
    data = {}

    # Basic Info
    data['age'] = np.random.randint(22, 60, n)
    data['experience'] = np.random.randint(0, 35, n)
    data['education'] = np.random.choice(['Bachelors', 'Masters', 'PhD'], n)
    data['department'] = np.random.choice(['IT', 'HR', 'Finance', 'Sales'], n)

    # Work Metrics
    data['projects_completed'] = np.random.randint(1, 20, n)
    data['training_hours'] = np.random.randint(0, 100, n)
    data['avg_work_hours'] = np.random.randint(6, 12, n)
    data['on_time_delivery_rate'] = np.round(np.random.uniform(0.5, 1.0, n), 2)

    # Feedback
    data['manager_rating'] = np.round(np.random.uniform(1, 5, n), 1)
    data['peer_feedback'] = np.round(np.random.uniform(1, 5, n), 1)

    df = pd.DataFrame(data)

    # 🎯 Target Logic (IMPORTANT)
    performance_score = (
        df['projects_completed'] * 0.3 +
        df['training_hours'] * 0.1 +
        df['on_time_delivery_rate'] * 20 +
        df['manager_rating'] * 2 +
        df['peer_feedback'] * 2
    )

    # Categorize performance
    conditions = [
        performance_score >= 40,
        (performance_score >= 25) & (performance_score < 40),
        performance_score < 25
    ]

    choices = ['High', 'Medium', 'Low']

    df['performance'] = np.select(conditions, choices, default='Medium')

    return df


if __name__ == "__main__":
    df = generate_employee_data(1000)
    df.to_csv("data/raw/employee_data.csv", index=False)
    print("Dataset generated successfully!")