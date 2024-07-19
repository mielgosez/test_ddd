from lbc_extra_indexes.presentation.orchestrator import Orchestrator

"""
# Define categories
NODES_ROOMS = {'1': [1, 2], '2': [2, 3], '3': [3, 4], '4': [4, 5], '5_plus': [5, np.Inf]}
NODES_SIZE = {'< 75': [0, 75], '75_100': [75, 100], '100_150': [100, 150], '150_plus': [150, np.Inf]}


# Step 1: Create categorical variables for rooms_slot and size_slot
def categorize_rooms(rooms):
    for key, value in NODES_ROOMS.items():
        if value[0] <= rooms < value[1]:
            return key
    return '5_plus'  # Handle rooms > 4 (5_plus category)


def categorize_size(size):
    for key, value in NODES_SIZE.items():
        if value[0] <= size < value[1]:
            return key
    return '150_plus'  # Handle size > 150 (150_plus category)


df['price_m2'] = df['price'] / df['size']
df['rooms_slot'] = df['rooms'].apply(categorize_rooms)
df['size_slot'] = df['size'].apply(categorize_size)

# Step 2: Filter by valid zip codes with at least 30 observations
valid_zip_codes = df['zip_code'].value_counts()[df['zip_code'].value_counts() >= 30].index
df_filtered = df[df['zip_code'].isin(valid_zip_codes)]

# Step 3: Compute weights per size_slot, rooms_slot, and zip_code
df_weights = df_filtered.groupby(['size_slot', 'rooms_slot', 'zip_code']).size().reset_index(name='weight')

# Step 4: Compute median price_m2 (price_1) per size_slot, rooms_slot, and zip_code
price_1 = df_filtered.groupby(['size_slot', 'rooms_slot', 'zip_code'])['price_m2'].median().reset_index()

# Step 5: Compute median price_m2 (price_2) per size_slot and zip_code
price_2 = df_filtered.groupby(['size_slot', 'zip_code'])['price_m2'].median().reset_index()

# Step 6: Merge price_1 and price_2 to calculate correction factor per size_slot and rooms_slot
merged_prices = price_1.merge(price_2, on=['size_slot', 'zip_code'], suffixes=('_1', '_2'))
merged_prices['correction_factor'] = merged_prices['price_m2_1'] / merged_prices['price_m2_2']

# Step 7: Calculate weighted correction factor per size_slot and rooms_slot
# Merge with df_weights to get weights
merged_prices = merged_prices.merge(df_weights, on=['size_slot', 'rooms_slot', 'zip_code'])

# Calculate weighted_correction_factor
merged_prices['weighted_correction_factor'] = merged_prices['correction_factor'] * merged_prices['weight']


# Step 8: Adjust correction factors based on constraints
def adjust_correction_factor(row):
    size_slot = row['size_slot']
    rooms_slot = row['rooms_slot']
    correction_factor = row['weighted_correction_factor']

    if size_slot == '< 75':
        if rooms_slot == '2':
            return min(correction_factor, 1.035)  # Ensure correction factor is at least 1.0
        elif rooms_slot == '3':
            return min(correction_factor, 1.005)  # Penalize if correction factor is higher than 1.0
        elif rooms_slot == '4':
            return min(correction_factor, 0.95)  # Penalize more for rooms_slot = 4
        elif rooms_slot == '5_plus':
            return min(correction_factor, 0.95)  # Penalize most for rooms_slot = 5_plus
    else:
        if rooms_slot in ['2', '3']:
            return max(min(correction_factor, 1.108), 0.9)  # Ensure correction factor is between 0.9 and 1.1
        else:
            return min(correction_factor, 1.0005)  # Penalize for other rooms_slots


# Apply adjustment function
merged_prices['adjusted_correction_factor'] = merged_prices.apply(adjust_correction_factor, axis=1)

# Aggregate to get single adjusted correction factor per size_slot and rooms_slot
adjusted_correction_factors = merged_prices.groupby(['size_slot', 'rooms_slot'])['adjusted_correction_factor'].mean()

# Display adjusted correction factors (size_slot, rooms_slot, adjusted_correction_factor)
print(adjusted_correction_factors.reset_index())
"""


if __name__ == '__main__':
    orchestrator = Orchestrator(input_path='data/feed_20240714_leboncoin.csv',
                                model_type='naive')
    orchestrator.execute(train_required=True)
    pass
