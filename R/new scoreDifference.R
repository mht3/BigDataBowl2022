library(tidyverse)
plays_field_goals_weather <- plays_field_goals_weather_220101
plays_field_goals_weather_full <- merge(plays_field_goals_weather,games, by = "gameId")
plays_field_goals_weather_full <- filter(plays_field_goals_weather_full,!is.na(playId)) 
plays_field_goals_weather_full <- select(plays_field_goals_weather_full,-c(X,season,week,gameDate,gameTimeEastern))
plays_field_goals_weather_full <- mutate(plays_field_goals_weather_full,Home = (possessionTeam == homeTeamAbbr),score_difference_new = 0)
i<-1
while(i<=2641){
  if(plays_field_goals_weather_full[i,6] == FALSE){
    plays_field_goals_weather_full[i,25] = plays_field_goals_weather_full[i,13] * (-1)
  }else{
    plays_field_goals_weather_full[i,25] = plays_field_goals_weather_full[i,13]
  }
  i <- i + 1
}
plays_field_goals_weather_full <- select(plays_field_goals_weather_full,-c(homeTeamAbbr,visitorTeamAbbr,scoreDifference))
plays_field_goals_weather_scoreDifferencecheck <- select(plays_field_goals_weather_full,gameId,possessionTeam,preSnapHomeScore,preSnapVisitorScore,homeTeamAbbr,score_difference_new)
write.csv(plays_field_goals_weather_full,"plays_field_goals_weather_220103.csv")