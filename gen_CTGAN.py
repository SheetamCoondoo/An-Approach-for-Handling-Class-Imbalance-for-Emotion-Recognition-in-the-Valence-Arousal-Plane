import pandas as pd
from ctgan import CTGAN
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess the dataset
def load_data(filepath):
    data = pd.read_csv(filepath)
    print("Original class distribution:")
    print(data['Label'].value_counts())
    return data

def preprocess_data(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# Define and train CTGAN
def train_ctgan(data, class_label):
    cate_feature = ['Label']  #if any categorical features are there
    ctgan = CTGAN(
        epochs=5000,
        batch_size=10,
        generator_lr=0.0002,
        discriminator_lr=0.0002,
        verbose=True
    )
    ctgan.fit(data[data['Label'] == class_label], cate_feature, epochs=5000) #combine with categorical features
    #ctgan.fit(data[data['Label'] == class_label])
    return ctgan

# Generate synthetic data to balance the dataset
def generate_synthetic_data(ctgan, num_samples):
    synthetic_data = ctgan.sample(num_samples)
    return synthetic_data

if __name__ == '__main__':
    filepath = 'F:\\Personal\\AI,ML\\Datasets\\GSR_ECG_EEG_DATA.csv'
    data = load_data(filepath)
    
    # Preprocess features excluding 'Label'
    X = data.drop('Label', axis=1)
    y = data['Label']
    X_scaled, scaler = preprocess_data(X)

    # Class balance
    class_counts = data['Label'].value_counts()
    max_count = class_counts.max()

    # Dictionary to hold synthetic data for each class
    synthetic_pieces = []

    for class_label in class_counts.index:
        count = class_counts[class_label]
        if count < max_count:
            num_to_generate = max_count - count
            ctgan = train_ctgan(data, class_label)
            synthetic_data = generate_synthetic_data(ctgan, num_to_generate)
            synthetic_data = pd.DataFrame(synthetic_data, columns=X.columns)
            synthetic_data['Label'] = class_label
            synthetic_pieces.append(synthetic_data)
    
    # Combine and save the augmented dataset
    synthetic_data_complete = pd.concat(synthetic_pieces, ignore_index=True)
    augmented_data = pd.concat([data, synthetic_data_complete], ignore_index=True)
    augmented_data.to_csv('F:\\Personal\\AI,ML\\AugmentedDataset\\augmented_GSR_ECG_EEG_DATA.csv', index=False)
    
    print("New class distribution after augmentation:")
    print(augmented_data['Label'].value_counts())



