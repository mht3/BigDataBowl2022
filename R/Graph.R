library(tidyverse)
library(ggplot2)
library(ggrepel)
clutch_scores_by_distance_tmp <- clutch_scores_by_distance
clutch_scores_by_distance_tmp <- clutch_scores_by_distance_tmp %>% group_by(Kick_Category) %>% mutate_at(vars(KER),list(KER_scaled = scale))
clutch_scores_by_distance_tmp <- ungroup(clutch_scores_by_distance_tmp)
clutch_scores_by_distance_tmp2 <- filter(clutch_scores_by_distance_tmp, (KER_scaled >= 1 | KER_scaled <= -1))

clutch_scores_by_distance_graph <- ggplot(data = clutch_scores_by_distance_tmp)
clutch_scores_by_distance_graph <- clutch_scores_by_distance_graph + 
  geom_point(aes(x = Kick_Category,y = KER_scaled, color = Name)) + geom_label_repel(aes(x = Kick_Category,y = KER_scaled,label = ifelse((KER_scaled >= 1 | KER_scaled <= -1),as.character(Name),'')),point.padding = 0.5) + theme_classic()
clutch_scores_by_distance_graph <- clutch_scores_by_distance_graph + labs(y = "KER_scaled",title = "                                                                                      Scaled KER by Distance")

clutch_scores_by_scoreDifference_tmp <- clutch_scores_by_score_difference
clutch_scores_by_scoreDifference_tmp <- clutch_scores_by_scoreDifference_tmp %>% group_by(Score_Category) %>% mutate_at(vars(KER),list(KER_scaled = scale))
clutch_scores_by_scoreDifference_tmp <- ungroup(clutch_scores_by_scoreDifference_tmp)
clutch_scores_by_scoreDifference_tmp2 <- filter(clutch_scores_by_scoreDifference_tmp, (KER_scaled >= 1 | KER_scaled <= -1))

clutch_scores_by_scoreDifference_graph <- ggplot(data = clutch_scores_by_scoreDifference_tmp)
clutch_scores_by_scoreDifference_graph <- clutch_scores_by_scoreDifference_graph + 
  geom_point(aes(x = Score_Category,y = KER_scaled, color = Name)) + geom_label_repel(aes(x = Score_Category,y = KER_scaled,label = ifelse((KER_scaled >= 1 | KER_scaled <= -1),as.character(Name),'')),point.padding = 0.5) + theme_classic()
clutch_scores_by_scoreDifference_graph <- clutch_scores_by_scoreDifference_graph + labs(y = "KER_scaled",
                                                                          title = "                                                                                      Scaled KER by Score Difference")


clutch_scores_by_scoreDifference_graph 
clutch_scores_by_distance_graph 
