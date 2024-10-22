
import pandas as pd
import numpy as np


# Question 1

def reverse_chunks(lst, n):
    reversed_list = []
    
    for idx in range(0, len(lst), n):
        sublist = lst[idx:idx + n]
        
        left, right = 0, len(sublist) - 1
        while left < right:
            sublist[left], sublist[right] = sublist[right], sublist[left]
            left += 1
            right -= 1
        
        reversed_list.extend(sublist)
    
    return reversed_list

# Test cases
print(reverse_chunks([1, 2, 3, 4, 5, 6, 7, 8], 3))  # Output: [3, 2, 1, 6, 5, 4, 8, 7]
print(reverse_chunks([1, 2, 3, 4, 5], 2))           # Output: [2, 1, 4, 3, 5]
print(reverse_chunks([10, 20, 30, 40, 50, 60, 70], 4))  # Output: [40, 30, 20, 10, 70, 60, 50]




# Question 2


def categorize_by_length(strings):
    length_groups = {}
    
    for word in strings:
        word_len = len(word)
        
        if word_len in length_groups:
            length_groups[word_len].append(word)
        else:
            length_groups[word_len] = [word]
    
    return dict(sorted(length_groups.items()))

# Test cases
print(categorize_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))
# Output: {3: ['bat', 'car', 'dog'], 4: ['bear'], 5: ['apple'], 8: ['elephant']}





# Question 3

def flatten_dictionary(d, parent_key=''):
    flat_items = []
    
    for key, value in d.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        
        if isinstance(value, dict):
            flat_items.extend(flatten_dictionary(value, full_key).items())
        elif isinstance(value, list):
            for i, element in enumerate(value):
                flat_items.extend(flatten_dictionary({f"{key}[{i}]": element}, parent_key).items())
        else:
            flat_items.append((full_key, value))
    
    return dict(flat_items)

# Test case
nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

print(flatten_dictionary(nested_dict))

# Expected Output:
# {
#     "road.name": "Highway 1",
#     "road.length": 350,
#     "road.sections[0].id": 1,
#     "road.sections[0].condition.pavement": "good",
#     "road.sections[0].condition.traffic": "moderate"
# }





# Question 4

def get_unique_permutations(nums):
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue
            if used[i]:
                continue
            
            used[i] = True
            path.append(nums[i])
            
            backtrack(path, used)
            
            path.pop()
            used[i] = False

    nums.sort()
    result = []
    used = [False] * len(nums)
    
    backtrack([], used)
    
    return result

# Test case
print(get_unique_permutations([1, 1, 2]))

# Expected Output:
# [
#     [1, 1, 2],
#     [1, 2, 1],
#     [2, 1, 1]
# ]




# Question 5

import re

def extract_dates(text):
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',  # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b' # yyyy.mm.dd
    ]
    
    found_dates = []
    
    for pattern in date_patterns:
        found_dates.extend(re.findall(pattern, text))
    
    return found_dates

# Test case
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
print(extract_dates(text))

# Expected Output:
# ["23-08-1994", "08/23/1994", "1994.08.23"]



# Question 6

import polyline
import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Radius of Earth in meters
    return c * r

def decode_polyline_to_dataframe(polyline_str):
    coordinates = polyline.decode(polyline_str)
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    distances = [0]  # Initial distance
    for i in range(1, len(df)):
        dist = haversine(df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'],
                         df.iloc[i]['latitude'], df.iloc[i]['longitude'])
        distances.append(dist)
    
    df['distance'] = distances
    return df

# Example usage
polyline_str = "gfo}EtohhUxF?dAvHnEv@"
result_df = decode_polyline_to_dataframe(polyline_str)
print(result_df)




# Question 7

def rotate_and_transform(matrix):
    n = len(matrix)
    
    # Rotate the matrix 90 degrees clockwise
    rotated = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated[j][n - 1 - i] = matrix[i][j]
    
    # Transform the rotated matrix
    final_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated[i]) - rotated[i][j]
            col_sum = sum(rotated[k][j] for k in range(n)) - rotated[i][j]
            final_matrix[i][j] = row_sum + col_sum
    
    return final_matrix

# Example usage
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_transform(matrix)
print(result)




# Question 8

import pandas as pd

# Load dataset
df = pd.read_csv('dataset-1.csv')

def check_time_completeness(df):
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], errors='coerce')
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], errors='coerce')
    
    df.set_index(['id', 'id_2'], inplace=True)
    results = {}
    
    for (id_val, id_2_val), group in df.groupby(level=['id', 'id_2']):
        start_times = group['start']
        end_times = group['end']

        time_coverage = start_times.min() < start_times.max() and end_times.min() < end_times.max()
        all_days_covered = len(pd.Series(start_times.dt.dayofweek).unique()) == 6  # Check for Mon-Sun

        results[(id_val, id_2_val)] = not (time_coverage and all_days_covered)
    
    boolean_series = pd.Series(results, dtype=bool)
    boolean_series.index = pd.MultiIndex.from_tuples(boolean_series.index, names=['id', 'id_2'])
    
    return boolean_series

# Check the completeness of time data
result = check_time_completeness(df)
print(result)





