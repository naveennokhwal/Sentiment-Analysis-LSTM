import pandas as pd

def generate_data(file_path, length):
    data = pd.read_csv(file_path)
    label_counts = data['sentiment'].value_counts()

    # Check if the dataset contains enough data for each sentiment category
    if label_counts.get('negative', 0) < length or label_counts.get('neutral', 0) < length or label_counts.get('positive', 0) < length:
        raise ValueError("The provided length exceeds the number of data points available for any sentiment category.")
    
    # Filter data based on sentiment
    negative = data[data['sentiment'] == 'negative'].iloc[:length]
    neutral = data[data['sentiment'] == 'neutral'].iloc[:length]
    positive = data[data['sentiment'] == 'positive'].iloc[:length]

    # Assign new sentiment values
    negative['sentiment'] = 0
    neutral['sentiment'] = 2
    positive['sentiment'] = 4

    # Combine the modified dataframes back into one
    result = pd.concat([negative, neutral, positive], ignore_index=True)
        
    return result