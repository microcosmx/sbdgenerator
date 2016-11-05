
#par(mfrow=c(1,2))

library(car)

# 3d plot
library(scatterplot3d)

#pdf("plotchart.pdf")
quartz()
#png("plotcharts.png")

par(mfrow=c(2,2))

ga_data <- read.csv("/Users/admin/work/workspace_new/workspace_scala/sbdgenerator/server/data2/trans.csv", header = TRUE, sep = ",")
features <- c("adultpop","income", "households","income", "rank","income")
dim(features) <- c(2,3)
lapply(1:3, function(i) {
  #png(paste0("relation-chart-",i,".png"))
  
  plot(ga_data[[paste0(features[1,i])]], ga_data[[paste0(features[2,i])]], 
       main = paste0(features[1,i],"-",features[2,i]," Relation"), 
       xlab = paste0(features[1,i]," Value"), ylab = paste0(features[2,i]," Value"))
  abline(lm(ga_data[[paste0(features[2,i])]] ~ ga_data[[paste0(features[1,i])]]), col="red", lwd=2, lty=1) 
  lines(lowess(ga_data[[paste0(features[1,i])]], ga_data[[paste0(features[2,i])]]), col="blue", lwd=2, lty=2) 
  
})


s3d <-scatterplot3d(ga_data$income, ga_data$adultpop, ga_data$rank, 
                    pch=16, 
                    highlight.3d=TRUE,
                    type="h", 
                    main="3D Scatter Plot with Score relationship")
fit <- lm(ga_data$income ~ ga_data$adultpop+ga_data$rank) 
s3d$plane3d(fit)
