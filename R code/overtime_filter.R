library(tidyverse)
library(dplyr)
library(lubridate)
plays_field_goals_weather_tmp <- mutate(plays_field_goals_weather,overtime = 0)
plays_field_goals_overtime <- filter(plays_field_goals_weather,quarter == 5)
i<- 1 
while(i<=2643){
  j<- 1
  while(j<=28){
    if(plays_field_goals_weather_tmp[i,1] == plays_field_goals_overtime[j,1]){
      plays_field_goals_weather_tmp[i,16] = 1
    }
    j<- j + 1
  }
  i<- i + 1
}
plays_field_goals_overtime <- select(plays_field_goals_weather_tmp,gameId,overtime)

plays_field_goals_overtime <- distinct(plays_field_goals_overtime)