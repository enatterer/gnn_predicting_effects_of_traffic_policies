import itertools
import pandas as pd

'''Concept:
 the code treats your city‐based split as a small “subset-sum” (or partitioning) problem, 
 where you want three disjoint subsets (train, val, test) whose total graph‐counts come as close as possible to your 
 80%/10%/10% targets

1. Define Targets:
    - Compute the total number of graphs (∑ over all cities).
    - Multiply by 0.8, 0.1, and 0.1 to get your ideal train, val, and test sizes.

2. Enumerate Possible Splits:
    - Use itertools.combinations to pick 10 cities out of 16 for the train set.
    - For each train choice, take the remaining 6 cities and enumerate 4 for val; the last 2 become test.

3. Measure “Error” of Each Split:
    - For a given split, compute the sum of graphs in train, val, and test.
    - Calculate the absolute difference between each sum and its target.
    - Add those three differences together—this total is the split’s “error.”

4. Find the Best (or Top-K) Splits:
    - Keep track of the split(s) with the smallest total error.
    - Pull out just the single best, or sort and show the top 5 (or any number) by error.
'''
# City graph counts
city_counts = {
    'muenchen': 2945,
    'augsburg': 2945,
    'nuernberg': 2945,
    'ingolstadt': 2945,
    'regensburg': 2946,
    'wuerzburg': 2946,
    'aschaffenburg': 2946,
    'bamberg': 1000,
    'bayreuth': 2946,
    'erlangen': 2946,
    'fuerth': 2967,
    'kempten': 2967,
    'landshut': 2969,
    'rosenheim': 20,
    'schweinfurt': 600,
    'neuulm': 2967
}

total_graphs = sum(city_counts.values())
target_train = 0.8 * total_graphs
target_val = 0.1 * total_graphs
target_test = total_graphs - target_train - target_val

results = []

cities = list(city_counts.keys())

# Iterate over choices for the 10 train cities
for train_cities in itertools.combinations(cities, 10):
    train_sum = sum(city_counts[c] for c in train_cities)
    error_train = abs(train_sum - target_train)
    remaining1 = set(cities) - set(train_cities)
    
    # Iterate over choices for the 4 val cities from the remainder
    for val_cities in itertools.combinations(remaining1, 4):
        val_sum = sum(city_counts[c] for c in val_cities)
        error_val = abs(val_sum - target_val)
        test_cities = list(remaining1 - set(val_cities))
        test_sum = sum(city_counts[c] for c in test_cities)
        error_test = abs(test_sum - target_test)
        
        total_error = error_train + error_val + error_test
        results.append({
            'error': total_error,
            'train_cities': ', '.join(train_cities),
            'train_sum': train_sum,
            'val_cities': ', '.join(val_cities),
            'val_sum': val_sum,
            'test_cities': ', '.join(test_cities),
            'test_sum': test_sum
        })

# Convert to DataFrame and get top 5 splits
df_results = pd.DataFrame(results)
df_top5 = df_results.nsmallest(5, 'error').reset_index(drop=True)

if __name__ == "__main__":
    print(df_top5)
