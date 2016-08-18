
quartz()

library(car)
ga_data <- read.csv("trans.csv", header = TRUE, sep = ",")
scatterplotMatrix(~ ga_data$centile + ga_data$rank + ga_data$income, 
                  data=ga_data, spread=FALSE,
                  lty.smooth=2, main="Relation Matrix")
                  
Sys.sleep(1000)