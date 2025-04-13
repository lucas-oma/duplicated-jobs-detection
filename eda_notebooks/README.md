# Exploratory Data Analysis Summary

This document summarizes the key findings and observations from the exploratory data analysis (EDA) of the job postings dataset.

## Key Observations

- Job descriptions are stored in HTML format, requiring parsing for analysis.

- Some fields like finalState have formatting inconsistencies (e.g., trailing commas such as "MI,").

- Employment and Seniority are discrete, categorical variables, which can be useful for classification tasks.

- The dataset contains some null values, but their proportion is relatively low overall.

- String-based unique identifiers are used in the dataset.

- The word count distribution of job descriptions (after HTML text extraction) shows a good distribution, indicating that the parsing approach is effective and accurate for extracting meaningful information.