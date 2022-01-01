library(tidyverse)
library(dplyr)
library(lubridate)
plays_field_goals_1220 <- filter(plays,specialTeamsPlayType %in% c('Field Goal'))
plays_field_goals_1220 <- plays_field_goals_1220 %>% mutate(scoreDifference = preSnapHomeScore - preSnapVisitorScore,secondsRemain = 0)
plays_field_goals_1220$secondsRemain <- ifelse(plays_field_goals_1220$quarter <= 4,
                                               period_to_seconds(hms(plays_field_goals_1220$gameClock))/60 + (4 - plays_field_goals_1220$quarter) * 900,
                                               -(period_to_seconds(hms(plays_field_goals_1220$gameClock))/60))
plays_field_goals_1220$Result <- ifelse(plays_field_goals_1220$specialTeamsResult == "Kick Attempt Good",1,0)
plays_field_goals_1220 <- merge(plays_field_goals_1220,games, by = "gameId", all= TRUE)
plays_field_goals_1220 <- filter(plays_field_goals_1220, !is.na(playId))
plays_field_goals_1220$Home <- ifelse(plays_field_goals_1220$homeTeamAbbr == plays_field_goals_1220$possessionTeam,1,0)
plays_field_goals_1220 <- plays_field_goals_1220 %>% select(gameId,playId,playDescription,quarter,possessionTeam,Home,Result,kickerId,gameClock,kickLength,preSnapHomeScore,preSnapVisitorScore,scoreDifference,secondsRemain)
write.csv(plays_field_goals_1220,"plays_field_goals_1220.csv")

plays_field_goals_1220_tidy <- plays_field_goals_1220 %>% select(gameId,playId,possessionTeam,Home,Result,kickerId,kickLength,scoreDifference,secondsRemain)
write.csv(plays_field_goals_1220_tidy,"plays_field_goals_1220_tidy.csv")