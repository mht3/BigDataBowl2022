library(tidyverse)
library(dplyr)
library(lubridate)
plays_field_goals <- filter(plays,specialTeamsPlayType %in% c('Field Goal'))
plays_field_goals_regularTime <- plays_field_goals %>% filter(quarter <= 4)
plays_field_goals_regularTime <- plays_field_goals_regularTime %>% mutate(scoreDifference = preSnapHomeScore - preSnapVisitorScore, secondsRemain =  period_to_seconds(hms(gameClock))/60 + (4 - quarter) * 900)
plays_field_goals_regularTime <- plays_field_goals_regularTime %>% select(gameId,playId,playDescription,quarter,possessionTeam,specialTeamsResult,kickerId,gameClock,kickLength,preSnapHomeScore,preSnapVisitorScore,scoreDifference,secondsRemain)
plays_field_goals_regularTime$Result <- ifelse(plays_field_goals_regularTime$specialTeamsResult == "Kick Attempt Good",1,0)
write.csv(plays_field_goals_regularTime,"plays_field_goals_regularTime1.csv")

plays_field_goals_overTime <- plays_field_goals %>% filter(quarter == 5)
plays_field_goals_overTime <- plays_field_goals_overTime %>% mutate(scoreDifference = preSnapHomeScore - preSnapVisitorScore, secondsRemain =  period_to_seconds(hms(gameClock))/60)
plays_field_goals_overTime <- plays_field_goals_overTime %>% select(gameId,playId,playDescription,quarter,possessionTeam,specialTeamsResult,kickerId,gameClock,kickLength,preSnapHomeScore,preSnapVisitorScore,scoreDifference,secondsRemain)
plays_field_goals_overTime$Result <- ifelse(plays_field_goals_overTime$specialTeamsResult == "Kick Attempt Good",1,0)
write.csv(plays_field_goals_overTime,"plays_field_goals_overTime1.csv")

plays_regularTime_team <- merge(plays_field_goals_regularTime,games, by = "gameId", all= TRUE)
plays_regularTime_team <- filter(plays_regularTime_team, !is.na(playId))
plays_regularTime_team$Home <- ifelse(plays_regularTime_team$homeTeamAbbr == plays_regularTime_team$possessionTeam,1,0)

plays_overTime_team <- merge(plays_field_goals_overTime,games, by = "gameId", all= TRUE)
plays_overTime_team <- filter(plays_overTime_team, !is.na(playId))
plays_overTime_team$Home <- ifelse(plays_overTime_team$homeTeamAbbr == plays_overTime_team$possessionTeam,1,0)

plays_field_goals_regularTime_tidy <- plays_regularTime_team %>% select(gameId,playId,possessionTeam,Home,Result,kickerId,kickLength,scoreDifference,secondsRemain)
plays_field_goals_overTime_tidy <- plays_overTime_team %>% select(gameId,playId,possessionTeam,Home,Result,kickerId,kickLength,scoreDifference,secondsRemain)

write.csv(plays_field_goals_regularTime_tidy,"plays_regularTime_tidy.csv")
write.csv(plays_field_goals_overTime_tidy,"plays_overTime_tidy.csv")

write_csv(plays_regularTime_team,"plays_regularTime.csv")
write_csv(plays_overTime_team,"plays_overTime.csv")