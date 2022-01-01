library(tidyverse)
library(dplyr)
library(lubridate)
plays_field_goals_1220 %>% count(kickerId,sort=TRUE)
plays_field_goals_1220 <- filter(plays_field_goals_1220,!is.na(kickerId))
x <- plays_field_goals_1220[][8]
x <- unique(x)
x[1][1]
i <- 1
while(i <= 59){
  filename = sprintf('plays_field_goals_%s.csv', x[i,1])
  plays_field_goals_tmp <- filter(plays_field_goals_1220, kickerId == x[i,1] )
  write.csv(plays_field_goals_tmp,filename)
  i <- i+1
}