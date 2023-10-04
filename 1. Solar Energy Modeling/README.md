# README
The _solar_weather.csv_ dataset measures the energy production (kW) of solar panels and their surrounding weather at 200,000 points in time. This project performs linear regression modeling to predict a single panel's energy output when given certain weather condition parameters. The project then conducts in-depth error analysis by developing several different error metrics and error visualizations.

Overall, the linear regression model offers a fairly accurate fit to the general shape of the data. However, the dataset also exhibits fairly large variability and a notably large number of outliers (many panels produced a constant of zero kW, even in highly favorable weather conditions). Further data cleaning will likely increase the adequacy of the linear model, but further data exploration is also needed to understand the pressence of these outliers.

__Feature definitions:__ Two features were selected during feature engineering: _GHI_ and _isSun_. _GHI_, or global horizontal irradiance, is a measure of the amount of sunlight that hits a solar panel's face. _isSun_ is a binary variable that measures the presence of sunlight (0 indicates no sun, while 1 indicates sun).

__Libraries:__ This project uses four libraries: pandas, numpy, matplotlib, and sklearn.
