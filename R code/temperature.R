library(tidyverse)
library(dplyr)
library(lubridate)
games_weather_post2018 <- filter(games_weather,game_id >= 2018000000)
#games_weather_post2018_countgames <- count(games_weather_post2018,game_id)

games_post2018 <- filter(games,game_id >= 2018000000)

games_weather_post2018_startend <- merge(games_weather_post2018,games_post2018, by = "game_id", all = TRUE)
games_weather_post2018_startend <- merge(games_weather_post2018_startend,stadium_coordinates, by = "StadiumName", all = TRUE)
games_weather_post2018_startend <- filter(games_weather_post2018_startend, !is.na(Temperature))
games_weather_post2018_startend <- games_weather_post2018_startend[order(games_weather_post2018_startend$game_id,decreasing = F),]
games_weather_average_temperature <- mutate(games_weather_post2018_startend,average_temperature = 0)

i <- 1
x <- games_weather_post2018_startend
temperature <- 0
hour_count <- 0
current_game_id <- 0
while(i<=6285){
  if(current_game_id == 0){
    current_game_id <- x[i,2]
  }
  temperature_completetime = x[i,5]
  start_completetime = x[i,15]
  end_completetime = x[i,16]
  #print(temperature_completetime)
  #print(start_completetime)
  #print(end_completetime)
  temperature_time <- substring(temperature_completetime, nchar(temperature_completetime)-4,nchar(temperature_completetime))
  start_time <- substring(start_completetime,nchar(start_completetime)-4,nchar(start_completetime))
  end_time <- substring(end_completetime,nchar(end_completetime)-4,nchar(end_completetime))
  start_time_hour = substring(start_time,0,2)
  end_time_hour = substring(end_time,0,2)
  temperature_time_hour = substring(temperature_time,0,2)
  #print(end_time_hour)
  if(current_game_id == x[i,2]){
    if(temperature_time_hour >= start_time_hour){
      if(temperature_time_hour <= end_time_hour){
        temperature <- temperature + x[i,6]
        hour_count <- hour_count + 1
        #print(temperature)
      }
    }
  }else{
    #average_temperature = format(round(temperature/hour_count,2),nsmall = 2)
    average_temperature = temperature/hour_count
    print(x[i-1,2])
    roof_type = x[i-1,19]
    if(roof_type == "Indoor"){
       average_temperature = 70
    }
    j <- 1
    while(i-j >= 1 && current_game_id == x[i-j,2]){
      games_weather_average_temperature[i-j,23] = average_temperature
      j <- j + 1
    }
    current_game_id = x[i,2]
    temperature <- 0
    hour_count <- 0 
  }
  #print(i)
  #print(temperature)
  #print(hour_count)
  i <- i + 1 
}

average_temperature = temperature/hour_count
j <- 6279
while(j <= 6285){
  games_weather_average_temperature[j,23] = average_temperature
  j <- j + 1
}

games_weather_average_temperature <- games_weather_average_temperature %>% select(game_id,StadiumName,RoofType,TimeMeasure,Temperature,TimeStartGame,TimeEndGame,average_temperature)
#print(games_weather_average_temperature[1,8])
plays_field_goals_singlegame <- data.frame(matrix(ncol = 2, nrow = 804))
col <- c("gameId","average_temperature")
colnames(plays_field_goals_singlegame) <- col

i <- 1
game_count <- 1
current_game_id <- games_weather_average_temperature[1,1]
while(i<=6285){
  if(games_weather_average_temperature[i,1] != current_game_id){
    plays_field_goals_singlegame[game_count,1] = current_game_id
    plays_field_goals_singlegame[game_count,2] = games_weather_average_temperature[i-1,8]
    game_count <- game_count + 1
    current_game_id = games_weather_average_temperature[i,1]
  }
  i<- i + 1
}

plays_field_goals_singlegame[804,1] = 2021020700
plays_field_goals_singlegame[804,2] = 60.9800000

plays_field_goals_weather <- merge(plays_field_goals_1228,plays_field_goals_singlegame, by = "gameId", all = TRUE)
plays_field_goals_weather <- filter(plays_field_goals_weather,!is.na(playId))
plays_field_goals_weather <- subset(plays_field_goals_weather,select = -c(V15))
write.csv(plays_field_goals_weather,"plays_field_goals_weather.csv")

