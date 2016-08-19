
x <- 1:50
y <- sin(x)
z <- cos(x)
#x11() ## or windows(), default is pdf()
quartz()
plot(x,y)
plot(x,z)
Sys.sleep(10)
