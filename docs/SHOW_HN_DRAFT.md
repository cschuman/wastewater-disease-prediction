# Show HN Draft

## Title Options (pick one)

**Option A (Impact-focused):**
> Show HN: Predicting hospital surges from sewage – 2 weeks before patients arrive

**Option B (Technical):**
> Show HN: Open source wastewater surveillance forecasting using CDC data

**Option C (Direct):**
> Show HN: I built an open source tool to predict respiratory hospitalizations from wastewater

**Recommended: Option A** — Creates curiosity, implies novel approach

---

## Post Body

```
Show HN: Predicting hospital surges from sewage – 2 weeks before patients arrive

GitHub: https://github.com/cschuman/wastewater-disease-prediction

I built an open source forecasting tool that predicts respiratory disease
hospitalizations using wastewater surveillance data from the CDC.

**Why wastewater?**

When you're sick, viral RNA shows up in sewage days before you feel symptoms
and weeks before you might go to a hospital. The CDC's National Wastewater
Surveillance System (NWSS) tracks SARS-CoV-2, Influenza, and RSV concentrations
at treatment plants across the US.

This gives public health departments a 10-17 day early warning signal—enough
time to staff up, pre-position supplies, or issue public advisories.

**What the project does:**

- Fetches data from CDC public APIs (NWSS wastewater + NHSN hospitalizations)
- Trains XGBoost and ARIMA models to forecast weekly hospitalizations by state
- Includes health equity analysis using CDC's Social Vulnerability Index
- Provides a SvelteKit dashboard for exploring county-level data

**Technical details:**

- Python 3.11+, scikit-learn, XGBoost, statsmodels
- 127 tests, GitHub Actions CI, OpenSSF Best Practices certified
- MIT licensed

**What I'm looking for:**

The model currently achieves reasonable accuracy but there's room for improvement.
I'm especially interested in:

- Better feature engineering (weather, mobility data, vaccination rates)
- Ensemble approaches or neural architectures
- Validation against CDC FluSight forecasts
- Public health domain expertise

I'm also looking for co-maintainers if anyone wants to help build this into
something public health departments could actually use.

Happy to answer questions about the data, methodology, or wastewater surveillance
in general.
```

---

## Timing Tips

**Best times to post Show HN:**
- Tuesday-Thursday
- 8-10 AM Eastern (catches US morning + Europe afternoon)
- Avoid Mondays (buried by weekend backlog) and Fridays (lower engagement)

**First hour matters:**
- Be ready to respond to comments immediately
- Upvotes in first hour determine front page placement

---

## Anticipated Questions & Answers

**Q: How accurate is it?**
> Currently seeing MAE of ~X hospitalizations/100k at 2-week horizon. The baseline
> (just using last week's value) gets ~Y, so there's meaningful signal in the
> wastewater data. Accuracy varies by state based on wastewater site coverage.

**Q: Why not just use the CDC's existing forecasts?**
> CDC FluSight focuses on single pathogens. Hospitals manage total bed capacity,
> not individual diseases. This combines COVID + Flu + RSV into a "total respiratory
> burden" forecast. Also, the CDC doesn't publish wastewater-based hospitalization
> forecasts—they publish the data, but the forecasting is left to researchers.

**Q: Is anyone actually using this?**
> Not yet in production. This started as a research project to see if the signal
> was there. The goal is to get it good enough that a health department could
> actually deploy it. Looking for collaborators to help get there.

**Q: How does the equity analysis work?**
> We join forecasts with CDC's Social Vulnerability Index to identify counties
> that are both high-risk (predicted surge) and high-vulnerability (socioeconomic
> factors). The idea is to help prioritize resource allocation.

**Q: What's the latency on the data?**
> CDC publishes wastewater data weekly, typically with a 1-2 week lag. Hospital
> data has similar latency. So "real-time" is really "near-real-time weekly batches."

**Q: Can I run this locally?**
> Yes: `pip install -r requirements.txt`, then run the fetch scripts. Full data
> pipeline takes ~10 minutes. Dashboard requires Node.js but can run `npm run dev`
> to see it locally.

---

## Follow-up Strategy

**If it gains traction:**
1. Respond to every comment thoughtfully
2. Note feature requests in GitHub issues (link them)
3. Thank people who star/contribute
4. Post update comment after 24 hours with any interesting findings from discussion

**If it doesn't:**
- Don't despair—HN is fickle
- Try r/datascience or r/epidemiology instead
- Repost in 2-3 months with new features
