# WNBA Women's Health-Informed Betting
### By: David Zhong

## Description
Automated system predicting WNBA player performance by analyzing historical points and menstrual/cyclical patterns, generating betting picks, and storing results in a database.

## Disclaimer
**THE SUBJECT MATTER OF THIS PROJECT INVOLVES CONCEPTS THAT ARE INHERENTLY UNETHICAL, INVASIVE, AND POTENTIALLY INAPPROPRIATE.**
**THERE IS NO CREDIBLE OR PEER-REVIEWED SCIENTIFIC EVIDENCE ESTABLISHING ANY CORRELATION BETWEEN BASKETBALL PERFORMANCE AND THE MENSTRUAL CYCLE.**
**THIS PROJECT IS PRESENTED SOLELY AS AN ACADEMIC EXPLORATION AND SHOULD NOT BE INTERPRETED AS ENDORSING, SUPPORTING, OR PROMOTING ANY SUCH PRACTICES.**

## Actions
- 7 AM EST: Last night's WNBA player statistics are downloaded from the [SportsDataVerse wehoop WNBA data GitHub repository](https://github.com/sportsdataverse/wehoop-wnba-data).
- 8 AM EST: Python script goes through the playerbox, updating the player's points scored, team name, and updating fields such as result (won/lost), points differential (actual - predicted), and profit (+/-).
- 10 AM EST: Today's WNBA player point props are scraped from [ScoresAndOdds.com](https://www.scoresandodds.com/wnba/props), and recorded in props.json.
- 11 AM EST: Various AI generated prediction models will try and determine whether the player will score over or under the line set by the books. Their bet (over/under), performance note, and bet amount will be recorded along with the date, player name, model name, predicted points, actual points, over line, under line, over odds, and under odds.

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
