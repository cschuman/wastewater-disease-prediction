# Disclaimer

## Research Software Notice

This software is provided for **research and educational purposes only**. It is not intended for clinical decision-making, public health policy implementation, or any application where errors could result in harm to individuals or populations.

## Accuracy and Limitations

### Model Performance
- Current models achieve Mean Absolute Percentage Error (MAPE) of **16-21%** depending on region and forecast horizon
- Performance varies significantly by geographic region, with rural areas showing higher error rates
- Prediction accuracy degrades for forecast horizons beyond 2 weeks
- Models have not been validated against prospective real-world deployment scenarios

### Known Limitations
1. **Data Latency**: Wastewater surveillance data has inherent reporting delays of 3-7 days
2. **Geographic Coverage**: Not all regions have wastewater monitoring infrastructure; rural areas are underrepresented
3. **Pathogen Evolution**: Models trained on historical data may not accurately predict novel variants
4. **Seasonality**: Performance may vary across flu seasons and during atypical outbreak patterns
5. **Population Changes**: Models assume stable population demographics within prediction windows

## Not for Clinical Use

This software:
- Has **NOT** been reviewed or approved by the FDA, CDC, or any regulatory body
- Should **NOT** be used to make individual patient care decisions
- Should **NOT** be used as the sole basis for public health interventions
- Should **NOT** replace professional epidemiological judgment

## Data Sources

This project uses publicly available data from:
- CDC National Wastewater Surveillance System (NWSS)
- CDC National Healthcare Safety Network (NHSN)
- HealthData.gov

The accuracy and completeness of predictions depend on the quality of these upstream data sources, which may contain errors, revisions, or gaps.

## No Warranty

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED. THE AUTHORS AND CONTRIBUTORS:

- Make no representations about the suitability of this software for any purpose
- Disclaim all warranties including fitness for a particular purpose
- Are not liable for any damages arising from the use of this software
- Do not guarantee the accuracy, completeness, or timeliness of any predictions

## Responsible Use

Users of this software agree to:
1. Validate predictions against independent data sources before any application
2. Clearly communicate uncertainty and limitations when sharing results
3. Not misrepresent model outputs as official forecasts or clinical guidance
4. Report any bugs, errors, or unexpected behavior to the maintainers
5. Cite this project appropriately in any publications or presentations

## Contact

For questions about appropriate use cases or to report concerns, please open an issue on GitHub or contact the maintainers directly.

---

*Last updated: January 2025*
