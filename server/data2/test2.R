
x <- 1:10
y <- sin(x)
#x11() ## or windows(), default is pdf()
quartz()
plot(x,y)
Sys.sleep(10)
