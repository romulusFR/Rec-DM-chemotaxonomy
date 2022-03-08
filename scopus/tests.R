# https://journal.r-project.org/archive/2014/RJ-2014-014/RJ-2014-014.pdf
# https://github.com/randy3k/radian
# source("tests.R")

require(MRCV)


farmer2 <- read.table("data/2_mrcv.txt", header = TRUE)

head(farmer2, n = 3)
tail(farmer2, n = 3)

irt <- item.response.table(data = farmer2, I = 3, J = 4)


set.seed(102211) # Set seed to replicate bootstrap results
# MI.test(data = farmer2, I = 3, J = 4, type = "all", B = 1999, plot.hist = TRUE)

df <- item.response.table(data = farmer2, I = 3, J = 4, create.dataframe = TRUE)


set.seed(499077) # Set seed to replicate bootstrap results
# mod.fit <- genloglin(data = farmer2, I = 3, J = 4, model = "y.main", B = 1999, print.status = TRUE)
# summary(mod.fit)

anova(object = mod.fit, model.HA = "saturated", type = "all")


mod.fit.w3y1 <- genloglin(data = farmer2, I = 3, J = 4, model = count ~ -1 + W:Y +
    wi %in% W:Y + yj %in% W:Y + wi:yj + wi:yj %in% Y +
    wi:yj %in% W3:Y1, B = 1999)