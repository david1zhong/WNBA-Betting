# WNBA Women's Health-Informed Betting
### By: David Zhong

## Description
Automated system predicting WNBA player performance by analyzing historical points and menstrual/cyclical patterns, generating betting picks, and storing results in a database.

## Disclaimer
**THE SUBJECT MATTER OF THIS PROJECT INVOLVES CONCEPTS THAT ARE INHERENTLY UNETHICAL, INVASIVE, AND POTENTIALLY INAPPROPRIATE.**
**THERE IS NO CREDIBLE OR PEER-REVIEWED SCIENTIFIC EVIDENCE ESTABLISHING ANY CORRELATION BETWEEN BASKETBALL PERFORMANCE AND THE MENSTRUAL CYCLE.**
**THIS PROJECT IS PRESENTED SOLELY AS AN ACADEMIC EXPLORATION AND SHOULD NOT BE INTERPRETED AS ENDORSING, SUPPORTING, OR PROMOTING ANY SUCH PRACTICES.**

## Aim of the Project
To see if the top AI models are able to generate potentially profitable insights, no matter how unethical or ridiculous, as well as to investigate an interesting theoretical speculation. As someone interested in data analysis and sports betting (with a bonus of promoting and indulging in women's sport), I'm testing how well these tools are when pointed at something truly degenerate.

## What I Learned (So Far)
- This is most likely not profitable.
- There is no clear correlation between the menstrual cycle and basketball performance.
- Some players may not even menstruate.
- Leveraging AI is as useful as a coin-flip.

## Actions
- 7 AM EST: Last night's WNBA player statistics are downloaded from the [SportsDataVerse wehoop WNBA data GitHub repository](https://github.com/sportsdataverse/wehoop-wnba-data).
- 8 AM EST: Python script goes through the playerbox, updating the player's points scored, team name, and updating fields such as result (won/lost), points differential (actual - predicted), and profit (+/-).
- 10 AM EST: Today's WNBA player point props are scraped from [ScoresAndOdds.com](https://www.scoresandodds.com/wnba/props), and recorded in props.json.
- 11 AM EST: Various AI generated prediction models will try and determine whether the player will score over or under the line set by the books. Their bet (over/under), performance note, and bet amount will be recorded along with the date, player name, model name, predicted points, actual points, over line, under line, over odds, and under odds.

### Models
1. model_CL1 is generated from Claude Sonnet 4. The prompt is simply to focus on pattern recognition; with yearly playerbox data, go through each month and try to recognize repeated dips in point production at a similar time per month. The model tries to determine what type of game (very bad game, good game, average game, etc) the player would have today. This model is unable to generate wager amounts.
2. model_CL2 is generated from Claude Sonnet 4. The prompt is again, to focus on pattern recognition, but instead of just analyzing points, also analyze field goal percentage over each game from every year. 

## Technologies
- Python
- Pandas
- PostgreSQL (Supabase)
- SQLAlchemy
- BeautifulSoup
- Streamlit
- GitHub Actions

## Future Features
- Imagine you placed $1 on every bet placed. How much profit would you make?
- Let the models train off the data the other models and itself have already predicted in the past, to make more informed predictions. For example, I've noticed that again and again, the models seem to underestimate and always bet the under on Las Vegas Aces' star player A'ja Wilson. Letting the models learn off of previous predictions can help them to recognize not to underestimate some players, and start to bet more in favour of the over instead of the under and vice-versa for some players.
