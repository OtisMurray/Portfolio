# Otis Murray Portfolio

This repository is a collection of data science, machine learning, analytics, and software projects I built through coursework, research, hackathons, and team collaborations. The work spans predictive modeling, environmental forecasting, text classification, clustering, exploratory analysis, and database design across both R and Python workflows.

## Portfolio Highlights

### [FireSight](./FireSight)
An end-to-end wildfire risk forecasting project that combines environmental data engineering, machine learning, and application development.

- Built for the GDG @ Penn State Solution Challenge Hackathon: Innovate for Impact on April 11-12, 2026
- Built forecasting pipelines in Python using weather data, NASA FIRMS fire observations, and gridded spatial features
- Trained and compared tree-based models including Extra Trees and XGBoost
- Exposed predictions through a FastAPI backend
- Built an interactive React and Leaflet frontend for mapping wildfire risk and active fire detections
- Designed as an impact-oriented climate and resilience project aligned with the Google Solution Challenge format

### [PM2.5 Forecasting Model](./PM2.5%20Forecasting%20Model)
A large-scale environmental data processing and forecasting project focused on PM2.5 air pollution over North America.

- Processed NetCDF air quality data into cleaned modeling datasets
- Built forecasting workflows with CatBoost, XGBoost, and linear regression approaches
- Generated figures and supporting analysis scripts for model evaluation and communication

### [Reddit Topic Model](./Reddit%20Topic%20Model)
A text classification project in R that predicts Reddit post categories using embeddings and gradient-boosted models.

- Classified posts into 11 categories
- Performed preprocessing, dimensionality reduction, visualization, and hyperparameter tuning
- Logged model experiments to track tuning decisions and performance across runs

### [XGBOOST MODEL](./XGBOOST%20MODEL)
A regression modeling project that predicts IC50 values for the Omicron variant using patient and covariate data.

- Used R for preprocessing, feature engineering, dimensionality reduction, and modeling
- Applied XGBoost to a biomedical prediction task
- Saved tuning attempts and model artifacts for reproducibility

### [DOG BREEDS MODEL](./DOG%20BREEDS%20MODEL)
An unsupervised learning project in R focused on breed pattern discovery and probabilistic clustering.

- Normalized features with `caret`
- Applied PCA for dimensionality reduction
- Used Gaussian Mixture Models with `ClusterR` to assign probabilistic breed group memberships

### [House_Price](./House_Price)
A structured prediction project in R for estimating house sale prices from training and testing data.

- Cleaned and reformatted raw housing data
- Created group-based price estimates using house quality and condition labels
- Organized work into reproducible `src`, `data`, and `models` folders

### [SQL Project](./SQL%20Project)
A team database project designed to improve volunteer management for the United Helpers Organization.

- Helped design a relational database for volunteers, tasks, assignments, packages, and items
- Wrote SQL for table creation, data integrity rules, and operational queries
- Prepared the system for web access through PostgREST support

### [sleep-health-lifestyle](./sleep-health-lifestyle)
A Python-based exploratory data analysis project studying how occupation, demographics, and lifestyle habits relate to sleep quality and duration.

- Used Pandas and Matplotlib for cleaning, analysis, and visualization
- Investigated behavioral and demographic patterns tied to sleep outcomes

### [NAISS and NDL Bootcamp](./NAISS%20and%20NDL%20Bootcamp)
A set of hands-on notebooks covering practical machine learning workflows and experimentation.

- Data preprocessing and white-box model training
- Random forest modeling for mental health risk analysis
- Customer segmentation
- Classification projects on image and tabular datasets

## Additional Collaborative Work

### [Daily Journal](https://github.com/rmedcraft/Daily-Journal)
Collaborative Penn State Hackathon Fall 2024 project that won Best Use of MongoDB Atlas.

- Built a journaling application designed to flag warning signs of mental illness from user-written entries
- Worked on an applied AI-centered team project with product, data, and software components

## Skills Demonstrated Across This Portfolio

### Machine Learning and Statistics
- Regression and classification
- XGBoost, CatBoost, Extra Trees, Random Forests, Gaussian Mixture Models
- PCA and other dimensionality reduction workflows
- Hyperparameter tuning, model comparison, and evaluation

### Data Engineering and Preparation
- Feature engineering
- Missing value handling, scaling, encoding, and data cleaning
- Working with CSV, NetCDF, JSON, and spatially gridded datasets
- Organizing raw, interim, and processed data for reproducibility

### Software and Visualization
- R, Python, SQL, FastAPI, React, Leaflet, and PostgREST
- Pandas, Matplotlib, caret, ggplot2, xgboost, ClusterR, and scikit-learn workflows
- Interactive map-based visualization and model output communication

### Collaboration
- Individual coursework projects
- Research-style modeling and analysis
- Team software/database builds
- Hackathon development under time constraints

## Repository Structure

Each top-level folder represents a separate project. Some projects are full application or modeling repositories, while others are focused notebooks or academic deliverables. Together they show a range of work across:

- Machine learning
- forecasting
- environmental analytics
- text modeling
- SQL and database systems
- exploratory data analysis

## About Me

I am interested in statistical modeling, data science, machine learning, and applied analytics. This portfolio reflects the way I approach projects: start with the data carefully, build interpretable and well-structured workflows, and turn technical results into something useful and understandable.
