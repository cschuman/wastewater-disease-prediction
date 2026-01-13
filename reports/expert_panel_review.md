# Expert Panel Review: Wastewater Disease Prediction Project

## Panel Members
- **Dr. Sarah Chen** - CDC Epidemiologist, 20 years WBE experience
- **Dr. Marcus Thompson** - Principal Data Scientist, ML/Time Series Expert
- **Dr. Jennifer Walsh** - Former Hospital CMO, Healthcare Operations
- **Dr. David Kowalski** - Biostatistician, Reproducibility Critic

---

# ROUND 1: Initial Assessments

## Dr. Chen (Epidemiology)
> "The 0.2 correlation between wastewater and hospitalizations is **concerningly low**. Published studies achieve r=0.77-0.96 with proper population normalization. This suggests either data processing issues or alignment problems. The multi-pathogen combined burden forecasting angle IS novel - no published work addresses total respiratory capacity directly. But the core wastewater signal implementation needs investigation."

## Dr. Thompson (ML/Data Science)
> "XGBoost is a reasonable starting choice but **suboptimal for temporal forecasting**. N-BEATS and Temporal Fusion Transformers achieve 34-37% MAE improvement over CDC ensemble models. Your 6% summer MAPE is competitive; 16-21% winter is in the ballpark of state-of-art. The hybrid approach is directionally correct but the fixed thresholds (20% surge detection, 0.4/0.1 blend weights) look like they were tuned post-hoc."

## Dr. Walsh (Healthcare Operations)
> "I need to be blunt: **16-21% forecast error during surges is the difference between adequate care and crisis standards of care**. A 200-bed hospital expecting 160 admissions that gets 192 faces $50-100K in emergency staffing costs and potential patient harm. State-level granularity is wrong - hospitals need facility-level forecasts. As currently designed, **no hospital system would pay for this**."

## Dr. Kowalski (Critical Review)
> "Let me cut to the heart of the problem. Your own feature importance shows **hospitalization lag features contribute 90% of predictive power**. Wastewater contributes 2-5%. You've built an **autoregressive model with decorative wastewater features**. With only 78 weeks covering ONE winter surge, you cannot distinguish model skill from overfitting. This is a well-polished toy."

---

# ROUND 2: Debate - Is the Wastewater Signal Actually Adding Value?

## Dr. Thompson
> "I want to push back on Dr. Kowalski's claim that wastewater is 'decorative.' The 0.2 correlation is weak, but correlation isn't the right metric. We should test **Granger causality** - does wastewater predict hospitalizations after controlling for hospitalization history? And run proper **ablation studies** with statistical tests, not just MAE differences."

## Dr. Kowalski
> "I appreciate the methodological precision, Marcus. But look at the ablation results in the code:
> - COVID (no WW): MAE = 11.07
> - COVID (+ WW): MAE = 13.41
>
> **Adding wastewater made predictions WORSE.** The signal is noise, not value. Until someone demonstrates that wastewater predicts surge ONSET before hospitalization momentum is visible, I remain skeptical."

## Dr. Chen
> "David raises valid points, but I think the problem is implementation, not the wastewater signal itself. The project uses pre-computed percentile data from CDC, which loses precision. Studies achieving r=0.86+ used **raw concentration data with PMMoV normalization**. The signal is there - it's just not being extracted properly."

## Dr. Walsh
> "This is an academic debate. From my operational perspective, even IF the wastewater signal worked perfectly, state-level forecasts don't map to hospital decisions. **The granularity problem is fatal regardless of the signal quality.** County or metro-level would be minimally useful; facility-level is what operations actually need."

---

# ROUND 3: Debate - Is This Novel or Reinventing the Wheel?

## Dr. Chen
> "About 70% reinvention, 30% novel. The XGBoost approach on NWSS/NHSN data has been done. BUT the multi-pathogen combined burden angle is genuinely underexplored. Hospitals care about TOTAL respiratory capacity, not individual pathogens. That framing could be a real contribution."

## Dr. Thompson
> "Agreed on multi-pathogen. I'd add that the **hybrid approach for surges** - blending XGBoost with trend projection - is an interesting idea, even if the current implementation looks ad-hoc. The insight that you need different strategies for stable vs. volatile periods is valid."

## Dr. Kowalski
> "The hybrid model's 20% surge threshold and 0.4/0.1 blend weights scream **post-hoc optimization**. Were these determined before seeing test results? If not, the 23% improvement claim is p-hacked. And multi-pathogen? You still have ONE winter of RSV data. That's not validation."

## Dr. Walsh
> "Novel or not, the question is: **does anyone need this?** CDC already publishes free state-level forecasts. Commercial players like Biobot offer facility-level solutions. Where does this project fit? It's too aggregated for hospitals, not novel enough for research, and duplicates existing public health tools."

---

# ROUND 4: What Would Make This Useful?

## Dr. Chen's Recommendations
1. **Fix the correlation problem** - Use raw concentration data, add PMMoV normalization
2. **Add missing data sources** - Mobility, weather, vaccination coverage, variant proportions
3. **Focus on multi-pathogen as primary contribution** - Compare combined model vs sum of individual models
4. **Add probabilistic forecasting** - Quantile regression or conformal prediction

## Dr. Thompson's Recommendations
1. **Replace XGBoost with N-BEATS or TFT** - Purpose-built for temporal forecasting
2. **Rigorous ablation study** - Permutation importance, Granger causality
3. **Learned blend weights** - Meta-learner instead of fixed thresholds
4. **Submit to CDC Forecast Hub** - External validation against other models

## Dr. Walsh's Recommendations
1. **Partner with a health system** - Get facility-level data to test at actionable granularity
2. **Focus on surge detection** - Binary "surge coming" alert may be more useful than precise counts
3. **Target public health departments** - State-level is appropriate for policy, not hospital operations
4. **Develop decision support layer** - Translate forecasts into specific staffing recommendations

## Dr. Kowalski's Recommendations
1. **Wait for more data** - 3+ flu seasons minimum for credible validation
2. **Pre-register methodology** - Specify architecture and parameters BEFORE seeing test data
3. **Leave-one-season-out CV** - Proper temporal validation
4. **Demonstrate onset prediction** - Can you detect surges 2-4 weeks before hospitalization momentum?

---

# CONSENSUS FINDINGS

## What the Panel Agrees On

| Finding | Consensus Level |
|---------|-----------------|
| 0.2 wastewater correlation is too low | **UNANIMOUS** |
| State-level granularity limits utility | **UNANIMOUS** |
| Single-season validation is insufficient | **UNANIMOUS** |
| Feature importance shows wastewater contributes minimally | **UNANIMOUS** |
| Multi-pathogen angle is potentially novel | **STRONG** (3/4) |
| Hybrid surge approach is directionally interesting | **MODERATE** (2/4) |
| Would not survive real-world deployment as-is | **UNANIMOUS** |

## Key Metrics Comparison

| Metric | This Project | Literature Benchmark | Gap |
|--------|--------------|---------------------|-----|
| WW-Hospitalization Correlation | 0.2 | 0.77-0.96 | SEVERE |
| Surge Period MAPE | 16-21% | 13-14% (N-BEATS) | MODERATE |
| Stable Period MAPE | 6% | 4-6% | ACCEPTABLE |
| WW Feature Importance | 2-5% | Not reported | CONCERNING |
| Validation Seasons | 1 | 3+ required | INSUFFICIENT |

---

# FINAL VERDICT

## Strengths
1. Solid data engineering pipeline
2. Correct identification of multi-pathogen burden as important metric
3. Reasonable ML architecture choice (though suboptimal)
4. Honest backtesting methodology
5. Interactive visualization dashboard

## Critical Weaknesses
1. **Wastewater signal not actually adding predictive value** - Feature importance shows 2-5% contribution
2. **Single-season validation** - Cannot distinguish skill from overfitting
3. **Wrong granularity for end users** - State-level doesn't serve hospital operations
4. **Weak correlation** - 0.2 vs 0.77-0.96 in literature suggests implementation issues
5. **Hybrid model threshold likely overfit** - 20% surge detection, fixed blend weights

## Is This Project Useful?

| Use Case | Assessment |
|----------|------------|
| Academic publication | MAYBE - Multi-pathogen angle with fixes |
| Hospital operations | NO - Wrong granularity, insufficient accuracy |
| Public health surveillance | MARGINAL - Duplicates existing CDC tools |
| Learning project | YES - Good ML engineering practice |
| Production deployment | NO - Fundamental methodology issues |

## Bottom Line

> **"This is a well-executed learning project that correctly identifies an important public health problem. However, in its current form, it is an autoregressive hospitalization model with decorative wastewater features. The premise that wastewater enables early warning is not demonstrated by the evidence. Single-season validation on 78 weeks of data is insufficient to make claims about model utility. The multi-pathogen combined burden angle is the most promising direction, but would require additional flu seasons and proper temporal validation to be credible."**

---

# Recommendations for Next Steps

## If Goal is Publication
1. Wait for 2025-26 flu season data (prospective validation)
2. Focus exclusively on multi-pathogen contribution
3. Add proper uncertainty quantification
4. Submit to CDC Forecast Hub for external benchmark
5. Be honest about limitations in the paper

## If Goal is Real-World Impact
1. Partner with a health system for facility-level data
2. Pivot to surge detection (binary alert) rather than precise forecasting
3. Target state epidemiologists, not hospital administrators
4. Integrate with existing surveillance workflows

## If Goal is Technical Improvement
1. Replace XGBoost with N-BEATS or Temporal Fusion Transformer
2. Add mobility, weather, and variant data
3. Use raw wastewater concentrations with PMMoV normalization
4. Implement proper ablation with Granger causality testing
5. Pre-register methodology before additional testing

---

*Panel review conducted: January 2026*
*Methodology: Independent analysis followed by structured debate*
