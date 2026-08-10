
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.express as px

from dash import Dash, dcc, html, dash_table, Input, Output, State
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import silhouette_score


# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_csv("IBM_Dataset.csv")

X_raw = df.drop(columns=["Attrition"]).copy()
attrition_binary = df["Attrition"].map({"No": 0, "Yes": 1})


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
nominal_features = [
    "Department",
    "Gender"
]

ordinal_features = [
    "Education",
    "JobLevel",
    "JobSatisfaction",
    "WorkLifeBalance"
]

continuous_features = [
    "Age",
    "MonthlyIncome",
    "TotalWorkingYears",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsWithCurrManager"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "nominal",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            nominal_features
        ),
        (
            "ordinal",
            StandardScaler(),
            ordinal_features
        ),
        (
            "continuous",
            StandardScaler(),
            continuous_features
        )
    ],
    remainder="drop"
)

X_processed = preprocessor.fit_transform(X_raw)


# --------------------------------------------------
# Evaluate k = 2 through 10
# --------------------------------------------------
k_values = list(range(2, 11))
inertia_values = []
silhouette_values = []

for k in k_values:
    evaluation_model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    evaluation_labels = evaluation_model.fit_predict(X_processed)

    inertia_values.append(evaluation_model.inertia_)
    silhouette_values.append(
        silhouette_score(
            X_processed,
            evaluation_labels
        )
    )

evaluation_df = pd.DataFrame({
    "K": k_values,
    "Inertia": inertia_values,
    "Silhouette Score": silhouette_values
})


# --------------------------------------------------
# Final K-Means model
# --------------------------------------------------
FINAL_K = 4

kmeans = KMeans(
    n_clusters=FINAL_K,
    random_state=42,
    n_init=20
)

cluster_labels = kmeans.fit_predict(X_processed)

df_clustered = df.copy()
df_clustered["Cluster"] = cluster_labels
df_clustered["AttritionBinary"] = attrition_binary

cluster_name_map = {
    0: "Established Mid-Career Core",
    1: "Experienced External Talent",
    2: "High-Risk Early-Career Talent",
    3: "Long-Tenured Leadership Anchors"
}

df_clustered["ClusterLabel"] = (
    df_clustered["Cluster"]
    .map(cluster_name_map)
)


# --------------------------------------------------
# Cluster profiles
# --------------------------------------------------
cluster_summary = (
    df_clustered
    .groupby("Cluster")
    .agg(
        Employees=("AttritionBinary", "size"),
        AttritionRate=("AttritionBinary", "mean"),
        AverageAge=("Age", "mean"),
        AverageEducation=("Education", "mean"),
        AverageJobLevel=("JobLevel", "mean"),
        AverageIncome=("MonthlyIncome", "mean"),
        TotalExperience=("TotalWorkingYears", "mean"),
        YearsAtIBM=("YearsAtCompany", "mean"),
        YearsInRole=("YearsInCurrentRole", "mean"),
        YearsWithManager=("YearsWithCurrManager", "mean"),
        JobSatisfaction=("JobSatisfaction", "mean"),
        WorkLifeBalance=("WorkLifeBalance", "mean")
    )
    .reset_index()
)

cluster_summary["AttritionRate"] *= 100

cluster_summary["WorkforcePercent"] = (
    cluster_summary["Employees"]
    / len(df_clustered) * 100
)

cluster_summary["ClusterLabel"] = (
    cluster_summary["Cluster"]
    .map(cluster_name_map)
)

summary_display = cluster_summary[
    [
        "Cluster",
        "ClusterLabel",
        "Employees",
        "WorkforcePercent",
        "AttritionRate",
        "AverageAge",
        "AverageJobLevel",
        "AverageIncome",
        "YearsAtIBM",
        "JobSatisfaction",
        "WorkLifeBalance"
    ]
].round(2)


# --------------------------------------------------
# Charts
# --------------------------------------------------
elbow_fig = px.line(
    evaluation_df,
    x="K",
    y="Inertia",
    markers=True,
    title="Elbow Method for K-Means Model Selection"
)

elbow_fig.update_layout(
    xaxis_title="Number of Clusters (k)",
    yaxis_title="Within-Cluster Sum of Squares",
    template="plotly_white"
)

silhouette_fig = px.line(
    evaluation_df,
    x="K",
    y="Silhouette Score",
    markers=True,
    title="Silhouette Scores for K-Means Solutions"
)

silhouette_fig.update_layout(
    xaxis_title="Number of Clusters (k)",
    yaxis_title="Silhouette Score",
    template="plotly_white"
)

attrition_chart = px.bar(
    cluster_summary,
    x="ClusterLabel",
    y="AttritionRate",
    text="AttritionRate",
    title="Observed Attrition Rate by Employee Segment"
)

attrition_chart.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)

attrition_chart.update_layout(
    xaxis_title="Employee Segment",
    yaxis_title="Attrition Rate (%)",
    xaxis_tickangle=-15,
    margin=dict(b=150),
    template="plotly_white"
)

job_level_chart = px.bar(
    cluster_summary,
    x="ClusterLabel",
    y="AverageJobLevel",
    text="AverageJobLevel",
    title="Average Job Level by Employee Segment"
)

job_level_chart.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside"
)

job_level_chart.update_layout(
    xaxis_title="Employee Segment",
    yaxis_title="Average Job Level",
    xaxis_tickangle=-15,
    margin=dict(b=150),
    template="plotly_white"
)

income_chart = px.bar(
    cluster_summary,
    x="ClusterLabel",
    y="AverageIncome",
    text="AverageIncome",
    title="Average Monthly Income by Employee Segment"
)

income_chart.update_traces(
    texttemplate="$%{text:,.0f}",
    textposition="outside"
)

income_chart.update_layout(
    xaxis_title="Employee Segment",
    yaxis_title="Average Monthly Income ($)",
    xaxis_tickangle=-15,
    margin=dict(b=150),
    template="plotly_white"
)

tenure_long = cluster_summary.melt(
    id_vars=["ClusterLabel"],
    value_vars=[
        "TotalExperience",
        "YearsAtIBM",
        "YearsInRole",
        "YearsWithManager"
    ],
    var_name="Seniority Measure",
    value_name="Average Years"
)

tenure_chart = px.bar(
    tenure_long,
    x="ClusterLabel",
    y="Average Years",
    color="Seniority Measure",
    barmode="group",
    title="Experience and Seniority by Employee Segment"
)

tenure_chart.update_layout(
    xaxis_title="Employee Segment",
    yaxis_title="Average Years",
    xaxis_tickangle=-15,
    margin=dict(b=150),
    template="plotly_white"
)

satisfaction_long = cluster_summary.melt(
    id_vars=["ClusterLabel"],
    value_vars=[
        "JobSatisfaction",
        "WorkLifeBalance"
    ],
    var_name="Measure",
    value_name="Average Score"
)

satisfaction_chart = px.bar(
    satisfaction_long,
    x="ClusterLabel",
    y="Average Score",
    color="Measure",
    barmode="group",
    title="Job Satisfaction and Work-Life Balance by Employee Segment"
)

satisfaction_chart.update_layout(
    xaxis_title="Employee Segment",
    yaxis_title="Average Score",
    xaxis_tickangle=-15,
    margin=dict(b=150),
    template="plotly_white"
)


# --------------------------------------------------
# Shared styles
# --------------------------------------------------
input_style = {
    "width": "100%",
    "padding": "10px",
    "marginTop": "5px",
    "marginBottom": "15px",
    "boxSizing": "border-box"
}

card_style = {
    "padding": "20px",
    "border": "1px solid #dddddd",
    "borderRadius": "8px",
    "backgroundColor": "white",
    "marginBottom": "20px"
}

column_style = {
    "width": "32%",
    "display": "inline-block",
    "verticalAlign": "top",
    "padding": "0 1%"
}


# --------------------------------------------------
# Dash application
# --------------------------------------------------
app = Dash(__name__)

app.title = "IBM Employee Attrition Segment Explorer"

app.layout = html.Div(
    style={
        "maxWidth": "1500px",
        "margin": "0 auto",
        "padding": "25px",
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": "#f7f8fa"
    },
    children=[
        html.H1("IBM Employee Attrition Segment Explorer"),

        html.P(
            "This application uses K-Means clustering to identify natural "
            "employee segments and compare their observed attrition patterns."
        ),

        html.P(
            "The simulator performs cluster assignment only. It does not "
            "predict whether an individual employee will leave IBM."
        ),

        dcc.Tabs(
            children=[
                dcc.Tab(
                    label="Executive Overview",
                    children=[
                        html.Div(
                            style=card_style,
                            children=[
                                html.H3("Employee Segment Summary"),

                                dash_table.DataTable(
                                    data=summary_display.to_dict("records"),
                                    columns=[
                                        {
                                            "name": column,
                                            "id": column
                                        }
                                        for column in summary_display.columns
                                    ],
                                    page_size=10,
                                    style_table={
                                        "overflowX": "auto"
                                    },
                                    style_cell={
                                        "padding": "10px",
                                        "textAlign": "left",
                                        "minWidth": "120px"
                                    },
                                    style_header={
                                        "fontWeight": "bold",
                                        "backgroundColor": "#eeeeee"
                                    }
                                )
                            ]
                        ),

                        html.Div(
                            style=card_style,
                            children=[
                                dcc.Graph(
                                    figure=attrition_chart
                                )
                            ]
                        )
                    ]
                ),

                dcc.Tab(
                    label="Model Evaluation",
                    children=[
                        html.Div(
                            style=card_style,
                            children=[
                                dcc.Graph(figure=elbow_fig)
                            ]
                        ),

                        html.Div(
                            style=card_style,
                            children=[
                                dcc.Graph(figure=silhouette_fig)
                            ]
                        ),

                        html.Div(
                            style=card_style,
                            children=[
                                html.H3("K-Means Evaluation Results"),

                                dash_table.DataTable(
                                    data=evaluation_df.round(4).to_dict(
                                        "records"
                                    ),
                                    columns=[
                                        {
                                            "name": column,
                                            "id": column
                                        }
                                        for column in evaluation_df.columns
                                    ],
                                    style_cell={
                                        "padding": "10px",
                                        "textAlign": "left"
                                    },
                                    style_header={
                                        "fontWeight": "bold",
                                        "backgroundColor": "#eeeeee"
                                    }
                                ),

                                html.P(
                                    "The two-cluster solution achieved the "
                                    "highest Silhouette Score. Four clusters "
                                    "were selected because they produced more "
                                    "actionable workforce segments while "
                                    "maintaining reasonable cluster separation."
                                )
                            ]
                        )
                    ]
                ),

                dcc.Tab(
                    label="Cluster Profiles",
                    children=[
                        html.Div(
                            style=card_style,
                            children=[
                                dcc.Graph(figure=job_level_chart)
                            ]
                        ),

                        html.Div(
                            style=card_style,
                            children=[
                                dcc.Graph(figure=income_chart)
                            ]
                        ),

                        html.Div(
                            style=card_style,
                            children=[
                                dcc.Graph(figure=tenure_chart)
                            ]
                        ),

                        html.Div(
                            style=card_style,
                            children=[
                                dcc.Graph(
                                    figure=satisfaction_chart
                                )
                            ]
                        )
                    ]
                ),

                dcc.Tab(
                    label="Employee Segment Simulator",
                    children=[
                        html.Div(
                            style=card_style,
                            children=[
                                html.H3(
                                    "Enter a Hypothetical Employee Profile"
                                ),

                                html.Div(
                                    style=column_style,
                                    children=[
                                        html.Label("Age"),
                                        dcc.Input(
                                            id="age",
                                            type="number",
                                            min=18,
                                            max=60,
                                            value=30,
                                            style=input_style
                                        ),

                                        html.Label("Department"),
                                        dcc.Dropdown(
                                            id="department",
                                            options=[
                                                {
                                                    "label": value,
                                                    "value": value
                                                }
                                                for value in [
                                                    "Human Resources",
                                                    "Research & Development",
                                                    "Sales"
                                                ]
                                            ],
                                            value="Research & Development",
                                            clearable=False,
                                            style={
                                                "marginBottom": "15px"
                                            }
                                        ),

                                        html.Label("Gender"),
                                        dcc.Dropdown(
                                            id="gender",
                                            options=[
                                                {
                                                    "label": value,
                                                    "value": value
                                                }
                                                for value in [
                                                    "Female",
                                                    "Male"
                                                ]
                                            ],
                                            value="Female",
                                            clearable=False
                                        )
                                    ]
                                ),

                                html.Div(
                                    style=column_style,
                                    children=[
                                        html.Label("Education Level"),
                                        dcc.Input(
                                            id="education",
                                            type="number",
                                            min=1,
                                            max=5,
                                            value=3,
                                            style=input_style
                                        ),

                                        html.Label("Job Level"),
                                        dcc.Input(
                                            id="job-level",
                                            type="number",
                                            min=1,
                                            max=5,
                                            value=1,
                                            style=input_style
                                        ),

                                        html.Label("Monthly Income"),
                                        dcc.Input(
                                            id="monthly-income",
                                            type="number",
                                            min=1009,
                                            max=19999,
                                            value=4000,
                                            style=input_style
                                        ),

                                        html.Label("Total Working Years"),
                                        dcc.Input(
                                            id="total-working-years",
                                            type="number",
                                            min=0,
                                            max=40,
                                            value=5,
                                            style=input_style
                                        )
                                    ]
                                ),

                                html.Div(
                                    style=column_style,
                                    children=[
                                        html.Label("Job Satisfaction"),
                                        dcc.Input(
                                            id="job-satisfaction",
                                            type="number",
                                            min=1,
                                            max=4,
                                            value=2,
                                            style=input_style
                                        ),

                                        html.Label("Work-Life Balance"),
                                        dcc.Input(
                                            id="work-life-balance",
                                            type="number",
                                            min=1,
                                            max=4,
                                            value=2,
                                            style=input_style
                                        ),

                                        html.Label("Years at IBM"),
                                        dcc.Input(
                                            id="years-at-company",
                                            type="number",
                                            min=0,
                                            max=40,
                                            value=2,
                                            style=input_style
                                        ),

                                        html.Label(
                                            "Years in Current Role"
                                        ),
                                        dcc.Input(
                                            id="years-in-role",
                                            type="number",
                                            min=0,
                                            max=18,
                                            value=1,
                                            style=input_style
                                        ),

                                        html.Label(
                                            "Years With Current Manager"
                                        ),
                                        dcc.Input(
                                            id="years-with-manager",
                                            type="number",
                                            min=0,
                                            max=17,
                                            value=1,
                                            style=input_style
                                        )
                                    ]
                                ),

                                html.Button(
                                    "Assign Employee Segment",
                                    id="assign-button",
                                    n_clicks=0,
                                    style={
                                        "width": "100%",
                                        "padding": "14px",
                                        "marginTop": "20px",
                                        "fontSize": "16px",
                                        "fontWeight": "bold",
                                        "cursor": "pointer"
                                    }
                                ),

                                html.Div(
                                    id="simulator-output",
                                    style={
                                        "marginTop": "25px"
                                    }
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)


# --------------------------------------------------
# Simulator callback
# --------------------------------------------------
@app.callback(
    Output("simulator-output", "children"),
    Input("assign-button", "n_clicks"),
    State("age", "value"),
    State("department", "value"),
    State("education", "value"),
    State("gender", "value"),
    State("job-level", "value"),
    State("job-satisfaction", "value"),
    State("monthly-income", "value"),
    State("total-working-years", "value"),
    State("work-life-balance", "value"),
    State("years-at-company", "value"),
    State("years-in-role", "value"),
    State("years-with-manager", "value"),
    prevent_initial_call=True
)
def assign_employee_cluster(
    n_clicks,
    age,
    department,
    education,
    gender,
    job_level,
    job_satisfaction,
    monthly_income,
    total_working_years,
    work_life_balance,
    years_at_company,
    years_in_role,
    years_with_manager
):
    values = [
        age,
        department,
        education,
        gender,
        job_level,
        job_satisfaction,
        monthly_income,
        total_working_years,
        work_life_balance,
        years_at_company,
        years_in_role,
        years_with_manager
    ]

    if any(value is None for value in values):
        return html.P(
            "Please complete every field before assigning the segment.",
            style={"color": "red"}
        )

    employee = pd.DataFrame([{
        "Age": age,
        "Department": department,
        "Education": education,
        "Gender": gender,
        "JobLevel": job_level,
        "JobSatisfaction": job_satisfaction,
        "MonthlyIncome": monthly_income,
        "TotalWorkingYears": total_working_years,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_role,
        "YearsWithCurrManager": years_with_manager
    }])

    employee_processed = preprocessor.transform(employee)

    cluster_number = int(
        kmeans.predict(employee_processed)[0]
    )

    cluster_label = cluster_name_map[cluster_number]

    profile = cluster_summary[
        cluster_summary["Cluster"] == cluster_number
    ].iloc[0]

    return html.Div(
        style={
            "padding": "20px",
            "border": "1px solid #cccccc",
            "borderRadius": "8px",
            "backgroundColor": "#ffffff"
        },
        children=[
            html.H2("Assigned Employee Segment"),
            html.P([
                html.Strong("Cluster: "),
                str(cluster_number)
            ]),
            html.P([
                html.Strong("Segment: "),
                cluster_label
            ]),
            html.P([
                html.Strong(
                    "Observed cluster attrition rate: "
                ),
                f"{profile['AttritionRate']:.1f}%"
            ]),
            html.P([
                html.Strong("Average age: "),
                f"{profile['AverageAge']:.1f}"
            ]),
            html.P([
                html.Strong("Average job level: "),
                f"{profile['AverageJobLevel']:.1f}"
            ]),
            html.P([
                html.Strong("Average monthly income: "),
                f"${profile['AverageIncome']:,.0f}"
            ]),
            html.P([
                html.Strong("Average years at IBM: "),
                f"{profile['YearsAtIBM']:.1f}"
            ]),
            html.P([
                html.Strong("Average job satisfaction: "),
                f"{profile['JobSatisfaction']:.2f}"
            ]),
            html.P([
                html.Strong("Average work-life balance: "),
                f"{profile['WorkLifeBalance']:.2f}"
            ]),
            html.Hr(),
            html.P(
                "This simulator assigns the employee to the closest "
                "workforce segment. It does not predict whether an "
                "individual employee will leave IBM and should not be "
                "used for automated employment decisions."
            )
        ]
    )


if __name__ == "__main__":
    app.run(debug=True)
