
import pandas as pd
import numpy as np

df = pd.read_csv('dataset-2.csv')

df.head()

df.info()



# Question 9

import pandas as pd

def calculate_distance_matrix(df):
    unique_ids = pd.concat([df['id_start'], df['id_end']]).unique()
    distance_matrix = pd.DataFrame(0, index=unique_ids, columns=unique_ids)

    for _, row in df.iterrows():
        start_id = row['id_start']
        end_id = row['id_end']
        distance = row['distance']
        
        distance_matrix.at[start_id, end_id] = distance
        distance_matrix.at[end_id, start_id] = distance

    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if distance_matrix.at[i, k] + distance_matrix.at[k, j] > 0:
                    distance_matrix.at[i, j] = min(distance_matrix.at[i, j] or float('inf'), 
                                                    distance_matrix.at[i, k] + distance_matrix.at[k, j])

    for id_ in unique_ids:
        distance_matrix.at[id_, id_] = 0

    return distance_matrix

# Load the dataset
df = pd.read_csv('dataset-2.csv')

# Calculate the distance matrix
distance_df = calculate_distance_matrix(df)
print(distance_df)



# Question 10

import pandas as pd

def unroll_distance_matrix(distance_matrix):
    unrolled_data = []

    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:
                distance = distance_matrix.at[id_start, id_end]
                if distance > 0:  # Filter out unwanted distances
                    unrolled_data.append({
                        'id_start': id_start,
                        'id_end': id_end,
                        'distance': distance
                    })

    unrolled_df = pd.DataFrame(unrolled_data)
    return unrolled_df

# Unroll the distance matrix
unrolled_df = unroll_distance_matrix(distance_df)
print(unrolled_df)


# Question 11

import pandas as pd

def find_ids_within_ten_percent_threshold(df, reference_id):
    reference_data = df[df['id_start'] == reference_id]
    
    if reference_data.empty:
        return []  # No data for the reference_id

    average_distance = reference_data['distance'].mean()
    lower_threshold = average_distance * 0.9
    upper_threshold = average_distance * 1.1

    filtered_ids = df[(df['distance'] >= lower_threshold) & (df['distance'] <= upper_threshold)]
    unique_ids = filtered_ids['id_start'].unique()
    
    return sorted(unique_ids)

# Example usage
reference_id = 1001402  # Replace with your desired reference ID
result_ids = find_ids_within_ten_percent_threshold(unrolled_df, reference_id)
print(result_ids)




# Question 12


import pandas as pd

def calculate_toll_rate(df):
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    for vehicle, coefficient in rate_coefficients.items():
        df[vehicle] = df['distance'] * coefficient

    return df



# Question 13

import pandas as pd
from datetime import time

def calculate_time_based_toll_rates(df):
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    weekday_factors = [
        ('00:00:00', '10:00:00', 0.8),
        ('10:00:00', '18:00:00', 1.2),
        ('18:00:00', '23:59:59', 0.8)
    ]
    weekend_factor = 0.7
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    results = []

    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']

        for day in days_of_week:
            if day in days_of_week[:5]:  # Weekdays
                for start_time_str, end_time_str, factor in weekday_factors:
                    start_time = time.fromisoformat(start_time_str)
                    end_time = time.fromisoformat(end_time_str)
                    toll_rates = {vehicle: distance * rate * factor for vehicle, rate in rate_coefficients.items()}
                    
                    results.append({
                        'id_start': id_start,
                        'id_end': id_end,
                        'distance': distance,
                        'start_day': day,
                        'start_time': start_time,
                        'end_day': day,
                        'end_time': end_time,
                        **toll_rates
                    })
            else:  # Weekends
                start_time = time(0, 0)
                end_time = time(23, 59)
                toll_rates = {vehicle: distance * rate * weekend_factor for vehicle, rate in rate_coefficients.items()}
                results.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance': distance,
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                    **toll_rates
                })

    return pd.DataFrame(results)

# Assuming unrolled_df is the output from unroll_distance_matrix
time_based_toll_df = calculate_time_based_toll_rates(toll_rate_df)
print(time_based_toll_df)



