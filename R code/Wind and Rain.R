library(tidyverse)
library(dplyr)
library(lubridate)
games_weather_windrain <- mutate(games_weather_post2018_startend,condition = -1,real_WindSpeed = WindSpeed,quarter = -1)
games_weather_windrain <- select(games_weather_windrain,game_id,StadiumName,RoofType,TimeMeasure,TimeStartGame,TimeEndGame,WindSpeed,EstimatedCondition,condition,real_WindSpeed,quarter)
games_weather_windrain$EstimatedCondition <- games_weather_windrain$EstimatedCondition %>% replace_na('Clear')
row_length = count(games_weather_windrain,"game_id")

i <- 1
while(i <= 6285){
  if(games_weather_windrain[i,8] == "Clear") {
    games_weather_windrain[i,9] = 0
  }
  if(games_weather_windrain[i,8] == "Light Rain" | games_weather_windrain[i,8] == "Light Snow"){
    games_weather_windrain[i,9] = 1
  }
  
  if(games_weather_windrain[i,8] == "Moderate Rain" | games_weather_windrain[i,8] == "Moderate Snow"){
    games_weather_windrain[i,9] = 2
  }
  
  if(games_weather_windrain[i,8] == "Heavy Rain"){
    games_weather_windrain[i,9] = 3
  }
  
  if(games_weather_windrain[i,3] == "Indoor" | games_weather_windrain[i,3] == "Retractable"){
    games_weather_windrain[i,9] = 0
  }
  
  i <- i + 1
}

i <- 1
while(i <= 6285){
  if(games_weather_windrain[i,3] == "Indoor"){
    games_weather_windrain[i,10] = 0
  }
  i <- i + 1
}


games_weather_windrain_tmp <- mutate(games_weather_windrain,weather_time_hour = 0, weather_time_minute = 0, start_time_hour = 0, start_time_minute = 0, end_time_hour = 0, end_time_minute = 0)
i <- 1
x <- games_weather_windrain_tmp
hour_count <- 0
while(i<=6285){
  weather_completetime = x[i,4]
  start_completetime = x[i,5]
  end_completetime = x[i,6]
  #print(temperature_completetime)
  #print(start_completetime)
  #print(end_completetime)
  weather_time <- substring(weather_completetime, nchar(weather_completetime)-4,nchar(weather_completetime))
  start_time <- substring(start_completetime,nchar(start_completetime)-4,nchar(start_completetime))
  end_time <- substring(end_completetime,nchar(end_completetime)-4,nchar(end_completetime))
  start_time_hour = strtoi(substring(start_time,0,2))
  start_time_minute = strtoi(substring(start_time,4,5))
  if(is.na(start_time_minute)){
    start_time_minute = strtoi(substring(start_time,5,5))
  }
  end_time_hour = strtoi(substring(end_time,0,2))
  end_time_minute = strtoi(substring(end_time,4,5))
  if(is.na(end_time_minute)){
    end_time_minute = strtoi(substring(end_time,5,5))
  }
  weather_time_hour = strtoi(substring(weather_time,0,2))
  weather_time_minute = strtoi(substring(weather_time,4,5))
  if(end_time_hour == 0){
    #print("change 00 to 24")
    end_time_hour = 24
  }
  
  if(end_time_hour == 1){
    end_time_hour = 25
  }
  
  if(weather_time_hour == 0){
    #print("change 00 to 24")
    weather_time_hour = 24
  }
  
  if(weather_time_hour == 1){
    weather_time_hour = 25
  }
  
  x[i,12] = weather_time_hour
  x[i,13] = weather_time_minute
  x[i,14] = start_time_hour
  x[i,15] = start_time_minute
  x[i,16] = end_time_hour
  x[i,17] = end_time_minute
  i <- i + 1 
}
games_weather_windrain_tmp <- x

games_weather_windrain_tmp <- filter(games_weather_windrain_tmp,weather_time_minute == 0)
games_weather_windrain_tmp <- mutate(games_weather_windrain_tmp,gameId = game_id)
games_weather_windrain_tmp <- subset(games_weather_windrain_tmp,select = -c(game_id))
games_weather_windrain_tmp <- merge(games_weather_windrain_tmp,plays_field_goals_overtime,by="gameId",all = TRUE)
games_weather_windrain_tmp <- filter(games_weather_windrain_tmp,!is.na(overtime))

games_weather_windrain_tmp <- mutate(games_weather_windrain_tmp,game_length = (end_time_hour - start_time_hour)*60 + (end_time_minute - start_time_minute))
games_weather_windrain_tmp <- filter(games_weather_windrain_tmp,!is.na(game_length))
i<-1
while(i<=5034){
  weather_time_hour = games_weather_windrain_tmp[i,12]
  start_time_hour = games_weather_windrain_tmp[i,14]
  overtime = games_weather_windrain_tmp[i,18]
  if(weather_time_hour == start_time_hour){
    games_weather_windrain_tmp[i,11] = 1
  }
  
  if(weather_time_hour - start_time_hour == 1){
    games_weather_windrain_tmp[i,11] = 2
  }
  
  if(weather_time_hour - start_time_hour == 2){
    games_weather_windrain_tmp[i,11] = 3
  }
  
  if(weather_time_hour - start_time_hour == 3){
    games_weather_windrain_tmp[i,11] = 4
  }
  
  if(weather_time_hour - start_time_hour == 4 && overtime == 1){
    games_weather_windrain_tmp[i,11] = 5
  }
  
  i<- i + 1
}

games_weather_windrain_final <- select(games_weather_windrain_tmp,gameId,StadiumName,RoofType,TimeMeasure,TimeStartGame,TimeEndGame,real_WindSpeed,condition,quarter)
games_weather_windrain_final <- filter(games_weather_windrain_final,quarter != -1)

plays_field_goals_weather_tmp <- merge(plays_field_goals_weather,games_weather_windrain_final,by=c("gameId","quarter"),all = TRUE)
plays_field_goals_weather_tmp <- filter(plays_field_goals_weather_tmp,!is.na(playId))
write.csv(plays_field_goals_weather_tmp,"plays_field_goals_weather_211231.csv")

