from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import os

app = Flask(__name__)

# Load data and train models
file_path = os.path.join(os.path.dirname(__file__), 'generated_data_with_gender.csv')
data = pd.read_csv(file_path)
data[['Semester', 'Year']] = data['Semester Name'].str.split(' ', n=1, expand=True)
data['Year'] = pd.to_numeric(data['Year'])
semester_mapping = {'spring': 0, 'summer': 1, 'fall': 2}
data['Semester'] = data['Semester'].str.lower().map(semester_mapping)
data.drop(['Semester Name'], axis=1, inplace=True)
X = data[['Year', 'Semester']]
y_total = data['Total Number of Students']
y_male = data['Male']
y_female = data['Female']

model_total = LinearRegression()
model_male = LinearRegression()
model_female = LinearRegression()
model_total.fit(X, y_total)
model_male.fit(X, y_male)
model_female.fit(X, y_female)

# Helper functions
def validate_semester_input(semester_name):
    try:
        semester_name = semester_name.strip().lower().replace(" ", "")
        if len(semester_name) < 8 or len(semester_name) > 10:
            raise ValueError
        semester, year = semester_name[:-4], semester_name[-4:]
        year = int(year)
        if year < 2024 or year > 2030 or semester not in semester_mapping:
            raise ValueError
    except ValueError:
        return False
    return True

def predict_next_semester(semester_name):
    semester_name = semester_name.strip().lower().replace(" ", "")
    semester, year = semester_name[:-4], semester_name[-4:]
    year = int(year)
    semester = semester_mapping.get(semester)
    next_semester_data = [[year, semester]]
    total_students = int(model_total.predict(next_semester_data)[0])
    male_students = int(model_male.predict(next_semester_data)[0])
    female_students = int(model_female.predict(next_semester_data)[0])
    return total_students, male_students, female_students


# Plotting the linear regression results
def plot_regression(X, y, model, title, ylabel, color):
    year_range = np.linspace(X['Year'].min(), X['Year'].max(), 300)
    semester_range = np.linspace(0, 2, 300)
    X_values = np.array([[year, sem] for year, sem in zip(year_range, semester_range)])

    # Generate predictions for the range
    predictions = model.predict(X_values)

    # Plot actual data
    plt.scatter(X['Year'] + X['Semester']/3, y, color=color, label='Actual data')

    # Plot regression line
    plt.plot(X_values[:, 0] + X_values[:, 1]/3, predictions, color='red', linewidth=2, label='Regression line')
    plt.title(title)
    plt.xlabel('Year + Semester/3')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

plot_regression(X, y_total, model_total, 'Total Students Regression', 'Total Number of Students', 'blue')
plot_regression(X, y_male, model_male, 'Male Students Regression', 'Number of Male Students', 'green')
plot_regression(X, y_female, model_female, 'Female Students Regression', 'Number of Female Students', 'purple')
# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    semester_name = request.form['semester']
    if validate_semester_input(semester_name):
        total_students, male_students, female_students = predict_next_semester(semester_name)
        result = {
            'total_students': total_students,
            'male_students': male_students,
            'female_students': female_students
        }
    else:
        result = {'error': 'Invalid input. Please enter a semester between Spring 2024 and Fall 2030.'}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)