# WNBA Women's Health-Informed Betting
### By: David Zhong

## Description
Automated system predicting WNBA player performance by analyzing historical points and menstrual/cyclical patterns, generating betting picks, and storing results in a database.

## Disclaimer
**==THE SUBJECT MATTER OF THIS PROJECT INVOLVES CONCEPTS THAT ARE INHERENTLY UNETHICAL, INVASIVE, AND POTENTIALLY INAPPROPRIATE.==**
**==THERE IS NO CREDIBLE OR PEER-REVIEWED SCIENTIFIC EVIDENCE ESTABLISHING ANY CORRELATION BETWEEN BASKETBALL PERFORMANCE AND THE MENSTRUAL CYCLE.==**
**==THIS PROJECT IS PRESENTED SOLELY AS AN ACADEMIC EXPLORATION AND SHOULD NOT BE INTERPRETED AS ENDORSING, SUPPORTING, OR PROMOTING ANY SUCH PRACTICES.==**


## Features
- Pulls NFL game information, including team matchups, start times, and betting odds (spread, moneyline, totals), by querying an AWS Lambda API with data cached in S3 every 12 hours.
- Betting odds are from DraftKings US provided by The Odds API.
- Users can place bets or parlays on games with a wager cap of $15k.
- Bets are then recorded on a Google Sheet (had a MongoDB implentation but found that Google Sheets was more practical for my usage).

## Technologies
- React
- AWS Lambda
- AWS S3
- Node.js
- Rest API
- Tailwind CSS
- Google Apps Scripts
- Google Sheets

## Future Features
- Automatic results check
