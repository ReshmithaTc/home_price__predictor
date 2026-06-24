import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Function to generate dummy data for the home price prediction model
def generate_dummy_data(num_samples=1000):
    """
    Generates a synthetic dataset for home price prediction.
    Features include square feet, bedrooms, toilets, age, amenities, and city type.
    """
    np.random.seed(42) # for reproducibility

    # Generate numerical features
    square_feet = np.random.normal(loc=2500, scale=800, size=num_samples)
    bedrooms = np.random.randint(1, 6, size=num_samples)
    toilets = np.random.randint(1, 4, size=num_samples)
    age = np.random.randint(0, 100, size=num_samples)

    # Generate categorical features for amenities and city
    amenities = np.random.randint(0, 2, size=(num_samples, 4))
    amenities_df = pd.DataFrame(amenities, columns=['garage', 'swimming_pool', 'fireplace', 'garden'])
    
    city_types = np.random.choice(['urban', 'rural', 'suburban'], size=num_samples)
    city_df = pd.DataFrame({'city': city_types})

    # Combine all features into a single DataFrame
    df = pd.DataFrame({
        'square_feet': square_feet,
        'bedrooms': bedrooms,
        'toilets': toilets,
        'age': age,
    })
    
    df = pd.concat([df, amenities_df, city_df], axis=1)

    # Define base price and add noise
    base_price = (
        df['square_feet'] * 150 +
        df['bedrooms'] * 15000 +
        df['toilets'] * 10000 -
        df['age'] * 500
    )

    # Adjust price based on amenities
    base_price += df['garage'] * 25000
    base_price += df['swimming_pool'] * 50000
    base_price += df['fireplace'] * 15000
    base_price += df['garden'] * 10000

    # Adjust price based on city type
    city_price_map = {'urban': 100000, 'suburban': 50000, 'rural': 20000}
    base_price += df['city'].map(city_price_map)
    
    # Add some random noise to make it more realistic
    noise = np.random.normal(loc=0, scale=50000, size=num_samples)
    df['price'] = base_price + noise

    return df

# Main script to train the model and save it
if __name__ == '__main__':
    # 1. Generate the dummy data
    df = generate_dummy_data()

    # 2. Convert categorical 'city' column to numerical using one-hot encoding
    # We specify the categories to ensure all three city columns are created
    # This prevents errors if one category is accidentally missing from the dataset
    city_categories = ['urban', 'suburban', 'rural']
    df['city'] = pd.Categorical(df['city'], categories=city_categories)
    df = pd.get_dummies(df, columns=['city'], drop_first=True, dtype=int)
    
    # 3. Separate features (X) and target (y)
    X = df.drop('price', axis=1)
    y = df['price']

    # 4. Train the Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # 5. Save the trained model and the list of feature columns
    # The list of columns is crucial for making predictions with new data
    model_columns = list(X.columns)

    with open('model.pkl', 'wb') as file:
        pickle.dump({'model': model, 'columns': model_columns}, file)
    
    print("Model trained and saved successfully!")
    print("Features used:", model_columns)
