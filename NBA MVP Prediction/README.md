# NBA MBP Finalist Prediction

This project uses 20 years of NBA statistics to predict the league's MVP finalists in a given season. The project performs extensive data cleaning and pre-processing before training two supervised learning models. The models are then backtested with a proprietary error metric to tune hyperparameters and feature selection.

## Project Structure

- **notebooks/**: Contains Jupyter notebooks for data cleaning and model training.
  - `data_cleaning.ipynb`: Notebook for cleaning and merging various NBA datasets.
  - `modeling.ipynb`: Notebook for training and evaluating supervised learning models.

- **data/**: Contains all data files.
  - `cleaned_data.csv`: Cleaned NBA data from 1991 to 2021, including MVP voting and team statistics.
  - `mvps.csv`: Player statistics for all players recieivng MVP votes between 1991 and 2021.
  - `nicknames.csv`: Contains a mapping of three-letter team nicknames to full team names.
  - `players.csv`: Player statistics for all players between 1991 and 2021.
  - `stats_2022.csv`: Player and team statistics for the 2022-23 NBA season.
  - `teams.csv`: Teams statistics for all teams between 1991 and 2021.

- **.gitignore**: Specifies files and directories to be ignored by Git.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) for details.
