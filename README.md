# Restaurant Complaint Management System

This Streamlit application provides a platform for managing restaurant complaints and feedback. It offers functionalities for both customers to submit reviews and restaurant owners to analyze complaint data.

## Features

### For Customers:
- Submit reviews/complaints
- Automatic sentiment analysis of reviews
- Multi-label categorization of complaints
- Generation of auto-responses based on the complaint

### For Restaurant Owners:
- View complaint statistics for a specified number of days
- Analyze complaint categories and their frequencies
- Access detailed review information

## Technologies Used

- Python
- Streamlit
- Pandas
- TensorFlow
- Transformers (BERT)
- TextBlob
- MySQL

## Setup and Installation

1. Clone the repository
2. Install the required packages:

3. Set up a MySQL database named 'restaurant_review'
4. Update the database connection details in the script

## Running the Application

Run the Streamlit app using:

$:streamlit run main.py


## Usage

1. Select user type (Customer or Restaurant Owner) from the sidebar
2. For Customers:
   - Enter your review in the text area
   - Click 'Submit Review' to process and save your feedback
3. For Restaurant Owners:
   - Enter the number of days for which you want to view complaint data
   - Click 'Get Complaint Info' to view statistics and detailed review information

## Model and Data

- The application uses a pre-trained BERT model for multi-label classification
- Ensure that the model weights ('tf_model.h5') and the dataset ('last.csv') are present in the same directory as the script

## Note

- The sentiment analysis and categorization are performed automatically
- The application connects to a MySQL database to store and retrieve review data

## Future Improvements

- Implement user authentication
- Add more detailed analytics for restaurant owners
- Enhance the auto-response generation system
