# IBM Employee Attrition Analysis

## Understanding Why Talent Leaves

## Project Overview

This project uses K-Means clustering to identify meaningful employee segments and understand how different workforce profiles relate to employee attrition.

The analysis uses 1,470 IBM employee records and evaluates factors such as job level, compensation, tenure, satisfaction, work-life balance, department, age, education, and gender.

The goal is to help HR leaders move from asking "Who left?" to understanding "What types of employees may be at greater risk of leaving?"

## Business Problem

Employee attrition can create significant costs related to recruiting, onboarding, lost productivity, and knowledge retention.

This project focuses on identifying workforce segments with different attrition patterns so HR leaders can better target retention strategies.

## Analytical Method and Cluster Selection

K-Means clustering was used as the primary analytical method.

Clustering solutions from k=2 through k=10 were evaluated using both the Elbow Method and the Silhouette Score.

### Why k=4 Was Selected

The Silhouette Score was highest at k=2, at approximately 0.27, indicating that two clusters produced the strongest statistical separation. However, the two-cluster solution created employee groups that were too broad to support meaningful HR retention strategies.

The Elbow Method showed that inertia declined substantially through approximately k=4 and then began to flatten, suggesting diminishing improvement from adding more clusters.

The four-cluster solution also produced a local Silhouette Score improvement of approximately 0.197, compared with about 0.187 at k=3 and 0.144 at k=5.

Therefore, k=4 was selected as the best balance between:

- Statistical cluster separation
- The elbow pattern in within-cluster variation
- Business interpretability
- Actionable employee segmentation

## Employee Segments and Attrition Rates

| Employee Segment | Workforce Share | Observed Attrition |
|---|---:|---:|
| Established Mid-Career Core | 28% | 11.1% |
| Experienced External Talent | 9% | 6.8% |
| High-Risk Early-Career Talent | 53% | 21.6% |
| Long-Tenured Leadership Anchors | 9% | 8.8% |

## Key Findings

The highest-risk segment was the **High-Risk Early-Career Talent** group.

This segment represented approximately 53% of the workforce and had an observed attrition rate of 21.6%.

Employees in this group generally had:

- Lower job levels
- Lower monthly income
- Shorter company tenure
- Fewer years in their current role
- Lower job satisfaction
- Younger average age

More experienced and senior employee segments generally showed lower observed attrition.

## Application Features

The interactive Dash application includes:

- Employee segment overview
- Cluster profile comparisons
- Attrition rate analysis
- Elbow Method visualization
- Silhouette Score analysis
- Job level comparisons
- Monthly income analysis
- Tenure and seniority analysis
- Job satisfaction and work-life balance comparisons
- Employee scenario simulator

## Live Application

[View the IBM Employee Attrition Application](https://dazzled-aqua-scallop-e691ac2f.plotly.app/)

## Tools & Technologies

- Python
- Pandas
- Scikit-learn
- K-Means Clustering
- Plotly
- Dash
- Data Visualization
- Unsupervised Machine Learning

## Business Value

The application provides HR leaders with a practical way to understand workforce segments and identify groups that may require greater retention attention.

The employee simulator assigns a hypothetical employee to the closest workforce segment. It does not predict whether an individual employee will leave.

## Important Limitation

This analysis identifies workforce segments and their historical attrition rates. It does not establish causation and should not be used as the sole basis for employment decisions involving individual employees.

## Course

MGMT 59000 - AI for Business Analytics  
Purdue University, Daniels School of Business

## Author

Rabia Fatima  
M.S. Business Analytics  
Purdue University
